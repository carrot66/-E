"""Full-MOSEI cross-modal interaction experiment for Problem 2.

This experiment is deliberately isolated from the selected model.  It uses
the leakage-safe train/valid pickle created from the user-provided complete
CMU-MOSEI package, never reads the complete test split, and saves both the
best and last checkpoints together with hashes and validation records.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from q2_experiment import (
    Fusion,
    ROOT as Q2_ROOT,
    TextEncoder,
    fit_normalization,
    metrics,
    normalize,
    predict,
    prepare_split,
    scenarios,
    seed_all,
    sha,
    tensor_batch,
    write_csv,
)
from q2_full_mosei_experiment import profile
from q2_scattered_gap_experiment import augment_batch, get_banks, make_scattered_suite


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "question2" else SCRIPT_DIR
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
if (OUTPUT_ROOT / "question2").exists():
    OUTPUT_ROOT = OUTPUT_ROOT / "question2"
DEFAULT_SAFE_DATA = PROJECT_ROOT / "work/full_mosei/safe_train_valid.pkl"
DEFAULT_OUT = OUTPUT_ROOT / "问题2_Q1对齐质量融合完整优化"
DEFAULT_TEACHER = OUTPUT_ROOT / "问题2_完整MOSEI训练实验/完整数据_缺失蒸馏学生_种子2026.pt"


class AlignmentAwareFusion(nn.Module):
    """Modality-token attention followed by word-level temporal attention."""

    def __init__(self, hidden: int = 128, dropout: float = 0.2, layers: int = 2):
        super().__init__()
        self.hidden = hidden
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU())
            for dim in (768, 74, 35)
        ])
        self.modality_embedding = nn.Parameter(torch.randn(3, hidden) * 0.02)
        self.word_position = nn.Parameter(torch.randn(1, 50, hidden) * 0.02)
        # Q1-derived alignment descriptors: observed flags (3), global
        # coverage (3), modality count, all-modality agreement, per-modality
        # local continuity (3), normalized word position, and valid flag.
        self.quality_projection = nn.Sequential(
            nn.Linear(13, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        cross_layer = nn.TransformerEncoderLayer(
            hidden, 8, hidden * 4, dropout, batch_first=True,
            norm_first=True, activation="gelu"
        )
        self.cross_modal = nn.TransformerEncoder(
            cross_layer, layers, enable_nested_tensor=False
        )
        temporal_layer = nn.TransformerEncoderLayer(
            hidden, 8, hidden * 4, dropout, batch_first=True,
            norm_first=True, activation="gelu"
        )
        self.temporal = nn.TransformerEncoder(
            temporal_layer, 2, enable_nested_tensor=False
        )
        self.attention = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 96), nn.GELU(),
            nn.Dropout(dropout)
        )
        self.cls = nn.Linear(96, 3)
        self.reg = nn.Linear(96, 1)

    def forward(self, t, a, v, mask, valid):
        # h: batch x word x modality x hidden.
        h = torch.stack([
            proj(x) * mask[:, :, j, None]
            for j, (proj, x) in enumerate(zip(self.projections, (t, a, v)))
        ], dim=2)
        observed = mask.float()
        fractions = observed.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        prev = torch.cat([torch.zeros_like(observed[:, :1]), observed[:, :-1]], 1)
        nxt = torch.cat([observed[:, 1:], torch.zeros_like(observed[:, :1])], 1)
        continuity = observed * (prev + nxt) / 2.0
        count = observed.sum(-1, keepdim=True) / 3.0
        agreement = (observed.prod(-1, keepdim=True)).float()
        position = torch.arange(50, device=t.device, dtype=torch.float32)[None, :, None] / 49.0
        position = position.expand(t.shape[0], -1, -1)
        valid_flag = valid.float().unsqueeze(-1)
        quality = torch.cat([
            observed,
            fractions[:, None].expand(-1, 50, -1),
            count, agreement, continuity, position, valid_flag
        ], -1)
        quality_word = self.quality_projection(quality)

        # Modality-major order: T[0:50], A[50:100], V[100:150].
        tokens = h.permute(0, 2, 1, 3).reshape(t.shape[0], 150, self.hidden)
        token_mask = mask.permute(0, 2, 1).reshape(t.shape[0], 150)
        modality = self.modality_embedding.repeat_interleave(50, dim=0)[None]
        position = self.word_position.repeat(1, 3, 1)
        quality_token = quality_word[:, None].expand(-1, 3, -1, -1)
        quality_token = quality_token.reshape(t.shape[0], 150, self.hidden)
        tokens = tokens + modality + position + quality_token
        padding = ~token_mask
        # Transformer attention cannot receive a row whose every key is masked.
        # The dummy token is excluded from the output pool below.
        all_hidden = padding.all(1)
        if all_hidden.any():
            padding = padding.clone()
            padding[all_hidden, 0] = False
        tokens = self.cross_modal(tokens, src_key_padding_mask=padding)

        word_tokens = tokens.reshape(t.shape[0], 3, 50, self.hidden).permute(0, 2, 1, 3)
        weight = observed / observed.sum(-1, keepdim=True).clamp_min(1)
        word = (word_tokens * weight[..., None]).sum(2) + quality_word
        word = self.temporal(word, src_key_padding_mask=~valid)
        pool_mask = valid & mask.any(-1)
        attention = self.attention(word).squeeze(-1).masked_fill(~pool_mask, -1e4)
        attention = torch.softmax(attention, 1) * pool_mask
        attention = attention / attention.sum(1, keepdim=True).clamp_min(1e-8)
        pooled = (word * attention[..., None]).sum(1)
        pooled = self.head(pooled)
        return self.cls(pooled), 3 * torch.tanh(self.reg(pooled).squeeze(-1) / 3)


def load_teacher(path: Path, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    teacher = Fusion(gated=True).to(device)
    teacher.load_state_dict(checkpoint["model"])
    teacher.eval()
    return teacher, checkpoint


def save_checkpoint(path: Path, model, payload: dict):
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save({"model": model.state_dict(), **payload}, temp)
    os.replace(temp, path)


def train(args):
    device = args.device
    seed_all(args.seed)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = Path(args.safe_data).resolve(strict=True)
    source_hash = sha(source)
    teacher_path = Path(args.teacher).resolve(strict=True)
    teacher, teacher_checkpoint = load_teacher(teacher_path, device)
    started = time.time()

    with source.open("rb") as stream:
        safe = pickle.load(stream)
    if set(safe) != {"train", "valid"}:
        raise ValueError("safe_train_valid.pkl must contain train and valid only")
    if len(safe["train"]["id"]) != 16326 or len(safe["valid"]["id"]) != 1871:
        raise ValueError("Unexpected safe full-MOSEI train/valid size")

    encoder = TextEncoder(device)
    cache = Q2_ROOT / "work/问题2_特征缓存"
    train_split = prepare_split(safe["train"], encoder, cache, "cross_full_train", source_hash)
    valid_split = prepare_split(safe["valid"], encoder, cache, "cross_full_valid", source_hash)
    stats = fit_normalization(train_split)
    teacher_stats = teacher_checkpoint.get("stats")
    if teacher_stats is not None:
        for key in ("t", "a", "v"):
            if not np.allclose(stats[key]["mean"], teacher_stats[key]["mean"], atol=1e-5):
                raise ValueError("Normalization differs from saved teacher checkpoint")
            if not np.allclose(stats[key]["std"], teacher_stats[key]["std"], atol=1e-5):
                raise ValueError("Normalization differs from saved teacher checkpoint")
    normalize(train_split, stats)
    normalize(valid_split, stats)
    del safe

    quick = scenarios(valid_split, encoder, cache, stats, source_hash)
    scattered = make_scattered_suite(valid_split, encoder, cache, stats, source_hash)
    banks = get_banks(train_split, encoder, cache, stats, source_hash)
    teacher_prob, teacher_reg = predict(teacher, train_split, device)
    teacher_eligible = (
        (teacher_prob.argmax(1) == train_split["y"]) &
        (np.abs(teacher_reg - train_split["r"]) < 1.0)
    )

    config = {
        "experiment": "full_mosei_q1_alignment_quality_token_interaction",
        "source_sha256": source_hash,
        "teacher_sha256": sha(teacher_path),
        "script_sha256": sha(Path(__file__).resolve()),
        "train_samples": len(train_split["ids"]),
        "valid_samples": len(valid_split["ids"]),
        "architecture": f"three modality tokens per word + Q1 alignment quality descriptors (13 dims), 2-layer 8-head cross-modal Transformer, 2-layer temporal Transformer, hidden={args.hidden}, dropout={args.dropout}",
        "loss": "sqrt-frequency weighted CE + 0.6 SmoothL1 + teacher KL(T=2) and regression distillation on eligible augmented samples",
        "optimizer": f"AdamW lr={args.lr:g}, weight_decay=1e-3, batch={args.batch_size}, clip=1, patience={args.patience}",
        "augmentation": "35% clean, 30% contiguous modality subsets, 35% scattered shared TAV gaps",
        "selection": "full valid clean + contiguous + scattered profile from the predeclared q2 score",
        "data_policy": "safe train/valid only; full test and attachment-3 labels are never loaded",
    }
    (out / "问题2_Q1对齐质量融合实验配置.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model = AlignmentAwareFusion(hidden=args.hidden, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    weights = torch.as_tensor(
        np.sqrt(len(train_split["y"]) / (3 * np.bincount(train_split["y"], minlength=3))),
        dtype=torch.float32, device=device
    )
    best_score = -float("inf")
    stale = 0
    logs = []
    best_path = out / "问题2_Q1对齐质量融合完整优化_best.pt"
    last_path = out / "问题2_Q1对齐质量融合完整优化_last.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(train_split["ids"]))
        losses = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            mask, text, changed = augment_batch(train_split, indices, *banks)
            logits, regression = model(*tensor_batch(train_split, indices, device, text, mask))
            y = torch.as_tensor(train_split["y"][indices], device=device)
            r = torch.as_tensor(train_split["r"][indices], device=device)
            loss = F.cross_entropy(logits, y, weight=weights) + 0.6 * F.smooth_l1_loss(regression, r)
            eligible = changed & teacher_eligible[indices]
            if eligible.any():
                selected = torch.as_tensor(eligible, device=device)
                target_prob = torch.as_tensor(teacher_prob[indices][eligible], device=device)
                target_prob = F.softmax(torch.log(target_prob.clamp_min(1e-8)) / 2.0, -1)
                loss = loss + 0.2 * 4 * F.kl_div(
                    F.log_softmax(logits[selected] / 2.0, -1),
                    target_prob, reduction="batchmean"
                )
                loss = loss + 0.1 * F.smooth_l1_loss(
                    regression[selected],
                    torch.as_tensor(teacher_reg[indices][eligible], device=device)
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        result, old_rows, sparse_rows = profile(
            "cross_modal", model, valid_split, quick, scattered, device
        )
        row = {
            "epoch": epoch, "train_loss": float(np.mean(losses)), **result,
            "clean_Accuracy": float(old_rows[0]["Accuracy"]),
            "contiguous_30pct_mean_Accuracy": float(np.mean([r["Accuracy"] for r in old_rows[1:]])),
            "scattered_TAV_mean_Accuracy": float(np.mean([r["Accuracy"] for r in sparse_rows])),
            "best_checkpoint": str(best_path),
        }
        logs.append(row)
        write_csv(out / "问题2_Q1对齐质量融合训练记录.csv", logs)
        write_csv(out / "问题2_Q1对齐质量融合验证场景.csv",
                  [{"stage": "epoch", **r} for r in old_rows + sparse_rows])
        save_checkpoint(last_path, model, {
            "epoch": epoch, "selection_score": result["selection_score"],
            "stats": stats, "config": config, "class_names": ["Negative", "Neutral", "Positive"]
        })
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if result["selection_score"] > best_score + 1e-4:
            best_score = result["selection_score"]
            stale = 0
            save_checkpoint(best_path, model, {
                "epoch": epoch, "selection_score": best_score,
                "stats": stats, "config": config,
                "class_names": ["Negative", "Neutral", "Positive"]
            })
        else:
            stale += 1
        if stale >= args.patience:
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    result, old_rows, sparse_rows = profile(
        "cross_modal", model, valid_split, quick, scattered, device
    )
    write_csv(out / "问题2_Q1对齐质量融合最佳验证场景.csv",
              [{"stage": "best", **r} for r in old_rows + sparse_rows])
    baseline = float(args.baseline_score)
    final = {
        "selected": "cross_modal" if result["selection_score"] >= baseline + args.min_gain else "previous_full_student",
        "cross_modal_profile": result,
        "previous_full_student_selection_score": baseline,
        "acceptance_rule": f"cross-modal score >= previous + {args.min_gain}; no automatic replacement otherwise",
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha(best_path),
        "last_checkpoint": str(last_path),
        "last_checkpoint_sha256": sha(last_path),
        "source_sha256": source_hash,
        "teacher_sha256": sha(teacher_path),
        "elapsed_seconds": time.time() - started,
        "test_or_attachment3_labels_used": False,
        "model_saved": True,
    }
    (out / "问题2_Q1对齐质量融合实验结论.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "问题2_Q1对齐质量融合模型哈希.json").write_text(
        json.dumps({"best": sha(best_path), "last": sha(last_path),
                    "source": source_hash, "teacher": sha(teacher_path)},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(final, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-data", default=DEFAULT_SAFE_DATA)
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--baseline-score", type=float, default=0.5628739263399123)
    parser.add_argument("--min-gain", type=float, default=0.005)
    train(parser.parse_args())

