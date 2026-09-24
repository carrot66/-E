"""Predict every unlabeled Annex 3 sample with the selected Q2 adaptive ensemble."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from q2_crossmodal_full_experiment import CrossModalFusion
from q2_deberta_robust_ensemble import dynamic_blend, reconstruct_texts
from q2_deberta_text_experiment import DebertaSentiment, tokenize
from q2_experiment import (
    MODEL_ID, MODEL_REVISION, ROOT, TextEncoder, normalize, predict,
    prepare_split, runtime, sha,
)


EXPECTED_CROSS_SHA = "31fcae7dddea5f9008a1d42918dc0fc2daadb7115ccf1096b1424ac1ceb13d0f"
EXPECTED_DEBERTA_SHA = "565a34365754f12fca2ac6a87fdd705aeec241ee059978d7f4ddedb517584ab4"
CLASSES = ("Negative", "Neutral", "Positive")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def model_strength(prob: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Keep raw regression and offer a separate, class-consistent submission value."""
    label = prob.argmax(1)
    result = np.asarray(raw, dtype=np.float32).copy()
    result[label == 0] = np.minimum(result[label == 0], -0.01)
    result[label == 1] = 0.0
    result[label == 2] = np.maximum(result[label == 2], 0.01)
    return result


@torch.inference_mode()
def infer_text(model, tokenizer, text: str, device: str) -> tuple[np.ndarray, np.ndarray]:
    ids, attention = tokenize(tokenizer, [text], 96)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        logits, regression = model(
            torch.as_tensor(ids, device=device),
            torch.as_tensor(attention, device=device),
        )
    return logits.softmax(-1).float().cpu().numpy(), regression.float().cpu().numpy()


def run(args: argparse.Namespace) -> None:
    runtime(args.device)
    annex_dir = Path(args.annex3_dir).resolve(strict=True)
    out = Path(args.out).resolve()
    cross_path = Path(args.cross_checkpoint).resolve(strict=True)
    text_path = Path(args.deberta_checkpoint).resolve(strict=True)
    calibration_path = Path(args.calibration).resolve(strict=True)
    text_config_path = Path(args.deberta_config).resolve(strict=True)
    cross_hash, text_hash = sha(cross_path), sha(text_path)
    if (cross_hash, text_hash) != (EXPECTED_CROSS_SHA, EXPECTED_DEBERTA_SHA):
        raise ValueError("Checkpoint hashes differ from the GitHub selected-best README")

    params = json.loads(calibration_path.read_text(encoding="utf-8"))["parameters"]
    text_config = json.loads(text_config_path.read_text(encoding="utf-8"))
    model_path = text_config["model_path"]
    cross_checkpoint = torch.load(cross_path, map_location="cpu", weights_only=False)
    cross = CrossModalFusion(hidden=128, dropout=.2, layers=2).to(args.device)
    missing, unexpected = cross.load_state_dict(cross_checkpoint["model"], strict=False)
    if set(missing) != {"ordinal.weight", "ordinal.bias"} or unexpected:
        raise RuntimeError(f"Cross-modal checkpoint mismatch: {missing}, {unexpected}")
    cross.eval()

    text = DebertaSentiment(model_path, dropout=.15, train_layers=3).to(args.device)
    text_checkpoint = torch.load(text_path, map_location="cpu", weights_only=False)
    text.load_state_dict(text_checkpoint["model"])
    text.eval()
    bert_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    deberta_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoder = TextEncoder(args.device)

    files = sorted(annex_dir.glob("*.pkl"))
    if len(files) != 30:
        raise ValueError(f"Expected 30 aligned Annex 3 files; found {len(files)}")
    rows: list[dict] = []
    file_hashes: dict[str, str] = {}
    repeat_identical = True
    cache = ROOT / "work/问题2_特征缓存"
    for file in files:
        file_hashes[file.stem] = sha(file)
        with file.open("rb") as stream:
            container = pickle.load(stream)
        if set(container) != {"test"} or set(container["test"]) != {"text_bert", "audio", "vision"}:
            raise ValueError(f"Unexpected Annex 3 structure: {file.name}")
        sample = container["test"]
        if len(sample["audio"]) != 1:
            raise ValueError(f"Expected one sample in {file.name}")
        sample["id"] = [file.stem]
        prepared = normalize(
            prepare_split(sample, encoder, cache, "annex3_best_fusion", file_hashes[file.stem]),
            cross_checkpoint["stats"],
        )
        observed_text = prepared["mask"][:, :, 0]
        recovered = reconstruct_texts(prepared["b"], observed_text, bert_tokenizer)
        cross_p, cross_r = predict(cross, prepared, args.device)
        text_p, text_r = infer_text(text, deberta_tokenizer, recovered[0], args.device)
        valid_count = prepared["valid"].sum(1).clip(min=1)
        availability = observed_text.sum(1) / valid_count
        p, effective = dynamic_blend(
            cross_p, text_p, params["weight_deberta"], availability,
            params["neutral_bias"], params["positive_bias"],
        )
        r = (1 - effective) * cross_r + effective * text_r
        # Repeated forward passes catch accidental train-mode or random inference.
        cross_p2, cross_r2 = predict(cross, prepared, args.device)
        text_p2, text_r2 = infer_text(text, deberta_tokenizer, recovered[0], args.device)
        repeat_identical &= all(np.array_equal(x, y) for x, y in (
            (cross_p, cross_p2), (cross_r, cross_r2), (text_p, text_p2), (text_r, text_r2)
        ))
        if not np.isfinite(p).all() or not np.isfinite(r).all() or not np.allclose(p.sum(1), 1, atol=1e-6):
            raise ValueError(f"Invalid prediction: {file.name}")
        label = int(p.argmax(1)[0])
        strength = float(model_strength(p, r)[0])
        n_joint_unknown = int((prepared["valid"] & (prepared["b"][:, 0] == 100) & ~observed_text).sum())
        row = {
            "样本编号": file.stem,
            "预测极性": CLASSES[label],
            "提交强度_类别一致后处理": strength,
            "模型原始回归强度": float(r[0]),
            "预测最大概率": float(p[0, label]),
            "负向概率": float(p[0, 0]),
            "中性概率": float(p[0, 1]),
            "正向概率": float(p[0, 2]),
            "跨模态分支预测": CLASSES[int(cross_p.argmax(1)[0])],
            "文本分支预测": CLASSES[int(text_p.argmax(1)[0])],
            "文本分支实际融合权重": float(effective[0]),
            "文本不可用比例": float(1 - availability[0]),
            "语音不可用比例": float(1 - prepared["mask"][:, :, 1].sum(1)[0] / valid_count[0]),
            "视觉不可用比例": float(1 - prepared["mask"][:, :, 2].sum(1)[0] / valid_count[0]),
            "联合UNK屏蔽位置数": n_joint_unknown,
            "有效内容位置数": int(valid_count[0]),
        }
        rows.append(row)
        write_csv(out / "逐样本预测" / f"{file.stem}.csv", [row])
        print(json.dumps({"sample": file.stem, "prediction": row["预测极性"], "confidence": row["预测最大概率"]}, ensure_ascii=False), flush=True)

    if len(rows) != 30 or len({r["样本编号"] for r in rows}) != 30:
        raise ValueError("Annex 3 sample count or ID uniqueness failed")
    write_csv(out / "问题2_附件3全量预测.csv", rows)
    write_csv(out / "问题2_附件3提交版预测.csv", [
        {"样本编号": row["样本编号"],
         "预测类别编码_负0中1正2": CLASSES.index(row["预测极性"]),
         "预测极性": row["预测极性"],
         "预测强度": row["提交强度_类别一致后处理"]}
        for row in rows
    ])
    counts = Counter(row["预测极性"] for row in rows)
    confidence = np.asarray([row["预测最大概率"] for row in rows])
    branch_disagreement = [row["样本编号"] for row in rows if row["跨模态分支预测"] != row["文本分支预测"]]
    mean_missing = {k: float(np.mean([row[k] for row in rows])) for k in ("文本不可用比例", "语音不可用比例", "视觉不可用比例")}
    analysis = {
        "method": "GitHub selected best: DeBERTa-v3-small + cross-modal token interaction, availability-weighted calibrated log-probability blend",
        "样本数": len(rows),
        "类别计数": {name: counts.get(name, 0) for name in CLASSES},
        "类别比例": {name: counts.get(name, 0) / len(rows) for name in CLASSES},
        "平均最大类别概率": float(confidence.mean()),
        "最大类别概率中位数": float(np.median(confidence)),
        "低置信度样本_最大概率小于0.5": [row["样本编号"] for row in rows if row["预测最大概率"] < .5],
        "低置信度样本_最大概率小于0.6": [row["样本编号"] for row in rows if row["预测最大概率"] < .6],
        "两分支预测不一致样本": branch_disagreement,
        "平均模态不可用比例": mean_missing,
        "联合UNK屏蔽位置总数": sum(row["联合UNK屏蔽位置数"] for row in rows),
        "文本分支平均实际融合权重": float(np.mean([row["文本分支实际融合权重"] for row in rows])),
        "融合参数": {k: float(params[k]) for k in ("weight_deberta", "neutral_bias", "positive_bias")},
        "验证集结果_非附件3": json.loads(calibration_path.read_text(encoding="utf-8"))["ensemble_full_fixed_parameters"],
        "附件3标签可用": False,
        "附件3Accuracy或F1可计算": False,
        "提交强度规则": "Negative<=-0.01, Neutral=0, Positive>=0.01; raw model regression is separately retained",
    }
    audit = {
        "count": len(rows), "unique_ids": len({row["样本编号"] for row in rows}),
        "repeat_inference_identical": bool(repeat_identical),
        "finite_probabilities_and_regression": True, "probabilities_sum_to_one": True,
        "labels_loaded": False, "joint_unknown_masked_positions": analysis["联合UNK屏蔽位置总数"],
        "cross_checkpoint_sha256": cross_hash, "deberta_checkpoint_sha256": text_hash,
        "input_file_sha256": file_hashes, "bert_revision": encoder.revision,
        "calibration_sha256": sha(calibration_path),
        "device": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "cpu",
    }
    write_json(out / "问题2_附件3预测分析.json", analysis)
    write_json(out / "问题2_附件3推理核验.json", audit)
    write_csv(out / "问题2_附件3类别概率汇总.csv", [
        {"类别": name, "数量": counts.get(name, 0), "比例": counts.get(name, 0) / len(rows)} for name in CLASSES
    ])
    print(json.dumps({"finished": True, "counts": analysis["类别计数"], "mean_confidence": analysis["平均最大类别概率"], "repeat_identical": repeat_identical}, ensure_ascii=False), flush=True)
    del cross, text, encoder
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-checkpoint", default=ROOT / "outputs/question2/问题2_跨模态交互完整优化/问题2_跨模态交互完整优化_best.pt")
    parser.add_argument("--deberta-checkpoint", default=ROOT / "outputs/question2/问题2_DeBERTa文本增强_v1/问题2_DeBERTa文本最佳.pt")
    parser.add_argument("--deberta-config", default=ROOT / "outputs/question2/问题2_DeBERTa文本增强_v1/问题2_DeBERTa实验配置.json")
    parser.add_argument("--calibration", default=ROOT / "outputs/question2/最终最优结果/当前最优_DeBERTa自适应融合/问题2_视频组隔离融合结果.json")
    parser.add_argument("--annex3-dir", default=ROOT / "E题数据/E题数据/附件3-模态缺失特征样本/对齐版本")
    parser.add_argument("--out", default=ROOT / "outputs/question2/问题2_附件3最终最优预测")
    parser.add_argument("--device", default="cuda:1")
    run(parser.parse_args())
