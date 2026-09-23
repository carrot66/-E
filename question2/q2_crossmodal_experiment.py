"""Independent train/validation experiment for Problem 2.

This script never evaluates attachment-2 test or predicts attachment 3. It
compares a modality-specific, attention-fusion, consistency-trained candidate
against the already fixed single-seed baseline on identical validation masks.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from q2_experiment import (
    MODEL_ID, MODEL_REVISION, MODS, ROOT, Fusion, TextEncoder, block_mask, derive_masks,
    dump, fit_normalization, metrics, norm_text, normalize,
    runtime, scenarios, seed_all, sha, tensor_batch,
    text_features, write_csv,
)


class TemporalModality(nn.Module):
    """Local temporal context is learned separately before modality fusion."""

    def __init__(self, input_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.depthwise = nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden)
        self.pointwise = nn.Conv1d(hidden, hidden, 1)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        x = self.project(x) * observed.unsqueeze(-1)
        residual = self.pointwise(F.gelu(self.depthwise(x.transpose(1, 2)))).transpose(1, 2)
        return self.norm(x + self.dropout(residual)) * observed.unsqueeze(-1)


class CrossModalFusion(nn.Module):
    """Each time step attends over observed T/A/V evidence and a null token."""

    def __init__(self, hidden: int = 96, dropout: float = .2):
        super().__init__()
        self.hidden = hidden
        self.encoders = nn.ModuleList(TemporalModality(d, hidden, dropout) for d in (768, 74, 35))
        self.null = nn.Parameter(torch.randn(1, 1, hidden) * .02)
        self.cross = nn.MultiheadAttention(hidden, 4, dropout=dropout, batch_first=True)
        self.quality = nn.Linear(6, hidden)
        self.position = nn.Parameter(torch.randn(1, 50, hidden) * .01)
        self.pre_norm = nn.LayerNorm(hidden)
        layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout,
                                           batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.pool = nn.Linear(hidden, 1)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 64), nn.GELU(),
                                  nn.Dropout(dropout))
        self.cls = nn.Linear(64, 3)
        self.reg = nn.Linear(64, 1)
        self.aux = nn.ModuleList(nn.Linear(hidden, 3) for _ in range(3))

    def forward(self, t, a, v, mask, valid):
        batch, length = mask.shape[:2]
        encoded = [encoder(x, mask[:, :, i]) for i, (encoder, x) in
                   enumerate(zip(self.encoders, (t, a, v)))]
        stacked = torch.stack(encoded, dim=2)
        observed = mask.float()
        mean = stacked.sum(2) / observed.sum(2, keepdim=True).clamp_min(1.)
        keys = stacked.reshape(batch * length, 3, self.hidden)
        null = self.null.expand(batch * length, -1, -1)
        keys = torch.cat((keys, null), 1)
        key_pad = torch.cat((~mask.reshape(batch * length, 3),
                             torch.zeros(batch * length, 1, dtype=torch.bool,
                                         device=mask.device)), 1)
        query = mean.reshape(batch * length, 1, self.hidden) + null
        fused, _ = self.cross(query, keys, keys, key_padding_mask=key_pad,
                              need_weights=False)
        fractions = observed.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        quality = torch.cat((observed, fractions[:, None].expand(-1, length, -1)), -1)
        z = self.pre_norm(fused.reshape(batch, length, self.hidden) + mean +
                          self.quality(quality) + self.position[:, :length])
        z = self.temporal(z, src_key_padding_mask=~valid)
        has_observation = valid & mask.any(-1)
        w = torch.softmax(self.pool(z).squeeze(-1).masked_fill(~has_observation, -1e4), 1)
        w = w * has_observation
        w = w / w.sum(1, keepdim=True).clamp_min(1e-8)
        pooled = self.head((z * w.unsqueeze(-1)).sum(1))
        auxiliary = []
        for i, x in enumerate(encoded):
            m = mask[:, :, i].float()
            unary = (x * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp_min(1.)
            auxiliary.append(self.aux[i](unary))
        return self.cls(pooled), 3 * torch.tanh(self.reg(pooled).squeeze(-1) / 3), auxiliary


def inputs(split, indices, device, text=None, mask=None):
    return tensor_batch(split, indices, device, text, mask)


def prepare_split_safe(source, encoder, cache, name, source_hash):
    """Treat content-position BERT [UNK] as unavailable before encoding."""
    bert, valid, mask = derive_masks(source)
    mask[:, :, 0] &= bert[:, 0] != 100
    return {'b': bert, 'valid': valid, 'mask': mask,
            't': text_features(encoder, bert, mask[:, :, 0], cache, name + '_unk_safe', source_hash),
            'a': np.asarray(source['audio'], np.float32),
            'v': np.asarray(source['vision'], np.float32),
            'y': np.asarray(source['classification_labels'], np.int64),
            'r': np.asarray(source['regression_labels'], np.float32),
            'ids': list(map(str, source['id'])),
            'raw': list(map(str, source['raw_text']))}


@torch.inference_mode()
def predict(model, split, device, scenario=None):
    model.eval()
    all_prob, all_reg = [], []
    for start in range(0, len(split['ids']), 128):
        idx = np.arange(start, min(start + 128, len(split['ids'])))
        text = scenario['t'][idx] if scenario and scenario.get('t') is not None else None
        mask = scenario['mask'][idx] if scenario else None
        logits, intensity, _ = model(*inputs(split, idx, device, text, mask))
        all_prob.append(logits.softmax(-1).cpu().numpy())
        all_reg.append(intensity.cpu().numpy())
    return np.concatenate(all_prob), np.concatenate(all_reg)


def validation_rows(model, split, suite, device):
    rows = []
    for scenario in suite:
        prob, intensity = predict(model, split, device, scenario)
        rows.append({**{k: scenario[k] for k in ('subset', 'rate', 'position', 'replicate')},
                     **metrics(split['y'], split['r'], prob, intensity)})
    return rows


@torch.inference_mode()
def invariance_checks(model, split, device):
    model.eval()
    original = inputs(split, np.arange(min(8, len(split['ids']))), device)
    reference = model(*original)
    perturbed = [value.clone() for value in original]
    for modality in range(3):
        perturbed[modality][~original[3][:, :, modality]] = 12345.
    changed = model(*perturbed)
    masked_invariant = all(torch.allclose(x, y, atol=1e-5)
                           for x, y in zip(reference[:2], changed[:2]))
    empty = [value.clone() for value in original]
    empty[3].zero_()
    zero_logits, zero_reg, _ = model(*empty)
    all_missing_finite = bool(torch.isfinite(zero_logits).all() and
                              torch.isfinite(zero_reg).all())
    if not masked_invariant or not all_missing_finite:
        raise AssertionError('Cross-modal masking invariants failed')
    return {'masked_value_invariance': masked_invariant,
            'all_missing_finite': all_missing_finite}


def score(rows):
    clean = rows[0]
    missing = [r for r in rows[1:] if r['position'] == 'random']
    return (.5 * clean['Macro_F1'] + .5 * np.mean([r['Macro_F1'] for r in missing])
            - .1 * (clean['MAE'] + np.mean([r['MAE'] for r in missing])) / 2)


def short_gap_mask(valid, rate, seed):
    """Shared temporal point/short-block loss; never select padding positions."""
    rng = np.random.default_rng(seed)
    hidden = np.zeros_like(valid)
    for sample, row in enumerate(valid):
        index = np.flatnonzero(row)
        target = max(1, min(len(index), int(round(rate * len(index)))))
        selected = np.zeros(len(index), dtype=bool)
        for _ in range(len(index) * 4):
            if selected.sum() >= target:
                break
            start = int(rng.integers(0, len(index)))
            width = int(rng.integers(1, 4))
            selected[start:min(start + width, len(index))] = True
        if selected.sum() < target:
            remaining = np.flatnonzero(~selected)
            selected[rng.choice(remaining, size=target - selected.sum(), replace=False)] = True
        hidden[sample, index[selected]] = True
    return hidden


def short_gap_scenarios(split, encoder, cache, stats, source_hash):
    stress = []
    for subset in ('T', 'TAV'):
        for rate in (.1, .2, .3):
            mask = split['mask'].copy()
            hide = short_gap_mask(split['valid'], rate, 91000 + int(rate * 100))
            for channel in subset:
                mask[:, :, MODS.index(channel)] &= ~hide
            text = None
            if 'T' in subset:
                encoded = text_features(encoder, split['b'], mask[:, :, 0], cache,
                                        'validation_short_gap', source_hash)
                text = norm_text(encoded, mask[:, :, 0], stats)
            stress.append({'subset': subset, 'rate': rate, 'position': 'scattered-short',
                           'replicate': 0, 'mask': mask, 't': text})
    return stress


def main(args):
    # The source pickle contains all three official splits, but this experiment
    # deliberately accesses only train and valid objects and never evaluates test.
    runtime(args.device)
    seed_all(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(args.data_root) / '附件2-数据集特征文件/aligned_50.pkl'
    source_hash = sha(source)
    with source.open('rb') as handle:
        data = pickle.load(handle)
    for name in ('train', 'valid'):
        derive_masks(data[name])
        y = np.asarray(data[name]['classification_labels'])
        r = np.asarray(data[name]['regression_labels'])
        if not np.isfinite(r).all() or not np.array_equal(y, np.sign(r) + 1):
            raise ValueError(f'{name}: label inconsistency')
    if set(map(str, data['train']['id'])) & set(map(str, data['valid']['id'])):
        raise ValueError('train/valid sample ID overlap')
    encoder = TextEncoder(args.device)
    cache = ROOT / 'work/问题2_特征缓存'
    tr = prepare_split_safe(data['train'], encoder, cache, 'train', source_hash)
    va = prepare_split_safe(data['valid'], encoder, cache, 'valid', source_hash)
    stats = fit_normalization(tr)
    normalize(tr, stats)
    normalize(va, stats)
    banks = []
    for rate in (.1, .3, .5):
        hide = block_mask(tr['valid'], rate, 51000 + int(rate * 100))
        observed = tr['mask'][:, :, 0] & ~hide
        features = text_features(encoder, tr['b'], observed, cache, 'train_aug', source_hash)
        banks.append({'hide': hide, 't': norm_text(features, observed, stats), 'pattern': 'block'})
    for rate in (.1, .2, .3):
        hide = short_gap_mask(tr['valid'], rate, 61000 + int(rate * 100))
        observed = tr['mask'][:, :, 0] & ~hide
        features = text_features(encoder, tr['b'], observed, cache, 'train_short_gap', source_hash)
        banks.append({'hide': hide, 't': norm_text(features, observed, stats), 'pattern': 'scattered-short'})
    quick = scenarios(va, encoder, cache, stats, source_hash)
    model = CrossModalFusion().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    counts = np.bincount(tr['y'], minlength=3)
    class_weights = torch.as_tensor(np.sqrt(len(tr['y']) / (3 * counts)),
                                    dtype=torch.float32, device=args.device)
    checkpoint = out / f'问题2_交叉模态一致性_种子{args.seed}.pt'
    cfg = {'model': 'modality-specific depthwise temporal convolution + masked cross-modal attention',
           'bert_model': MODEL_ID, 'bert_revision': MODEL_REVISION, 'data_sha256': source_hash,
           'code_sha256': sha(__file__), 'seed': args.seed, 'batch_size': 64, 'max_epochs': args.epochs,
           'optimizer': 'AdamW', 'learning_rate': 3e-4, 'weight_decay': 1e-3,
           'dropout': .2, 'hidden': 96, 'augmentation_probability': .65,
           'augmentation_rates': {'contiguous': [.1, .3, .5], 'scattered-short': [.1, .2, .3]},
           'missingness_simulation': 'one shared temporal mask across selected modalities; short gaps of 1-3 aligned positions',
           'main_loss': 'class-weighted CE + 0.6 SmoothL1',
           'full_view_weight': .3, 'auxiliary_weight': .08,
           'consistency': '0.1 KL(T=2) + 0.03 SmoothL1 on correct high-confidence full views',
           'selection': '0.5 clean F1 + 0.5 quick missing F1 - 0.1 mean MAE',
           'acceptance_rule': 'score >= original distilled seed-2026 score + 0.01 and clean F1 >= original clean F1 - 0.005',
           'native_unknown_token_rule': 'at future inference, text_bert ID100 inside content must be unavailable before BERT and fusion; this script does not access attachment3',
           'data_policy': 'fit train; select valid; no test or attachment3 access'}
    dump(out / '实验预注册.json', cfg)
    best, stale, history = -float('inf'), 0, []
    began = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = np.random.permutation(len(tr['ids']))
        losses = []
        for start in range(0, len(permutation), 64):
            idx = permutation[start:start + 64]
            mask = tr['mask'][idx].copy()
            text = tr['t'][idx].copy()
            changed = np.zeros(len(idx), dtype=bool)
            for k, sample in enumerate(idx):
                if np.random.rand() > .65:
                    continue
                subset = np.random.choice(('T', 'A', 'V', 'TA', 'TV', 'AV', 'TAV'))
                bank = banks[np.random.randint(len(banks))]
                hide = bank['hide'][sample]
                for channel in subset:
                    mask[k, :, MODS.index(channel)] &= ~hide
                if 'T' in subset:
                    text[k] = bank['t'][sample]
                changed[k] = True
            logits, intensity, aux = model(*inputs(tr, idx, args.device, text, mask))
            y = torch.as_tensor(tr['y'][idx], device=args.device)
            r = torch.as_tensor(tr['r'][idx], device=args.device)
            loss = F.cross_entropy(logits, y, weight=class_weights) + .6 * F.smooth_l1_loss(intensity, r)
            aux_losses = []
            for modality, auxiliary_logits in enumerate(aux):
                eligible = torch.as_tensor(mask[:, :, modality].any(1), device=args.device)
                if eligible.any():
                    aux_losses.append(F.cross_entropy(auxiliary_logits[eligible], y[eligible],
                                                      weight=class_weights))
            if aux_losses:
                loss = loss + .08 * torch.stack(aux_losses).mean()
            changed_t = torch.as_tensor(changed, device=args.device)
            if changed_t.any():
                full_logits, full_intensity, _ = model(*inputs(tr, idx, args.device))
                loss = loss + .3 * (F.cross_entropy(full_logits, y, weight=class_weights) +
                                    .6 * F.smooth_l1_loss(full_intensity, r))
                with torch.no_grad():
                    full_prob = full_logits.softmax(-1)
                    reliable = changed_t & (full_prob.argmax(-1) == y) & (full_prob.max(-1).values > .6)
                if reliable.any():
                    kd = F.kl_div(F.log_softmax(logits[reliable] / 2, -1),
                                  F.softmax(full_logits[reliable].detach() / 2, -1),
                                  reduction='batchmean') * 4
                    consistency = F.smooth_l1_loss(intensity[reliable],
                                                   full_intensity[reliable].detach())
                    loss = loss + .1 * kd + .03 * consistency
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(float(loss.detach()))
        rows = validation_rows(model, va, quick, args.device)
        value = float(score(rows))
        record = {'epoch': epoch, 'train_loss': float(np.mean(losses)), 'clean_Macro_F1': rows[0]['Macro_F1'],
                  'clean_MAE': rows[0]['MAE'],
                  'quick_missing_Macro_F1': float(np.mean([r['Macro_F1'] for r in rows[1:]])),
                  'quick_missing_MAE': float(np.mean([r['MAE'] for r in rows[1:]])),
                  'selection_score': value, 'elapsed_seconds': time.time() - began}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        write_csv(out / '训练记录.csv', history)
        if value > best + 1e-4:
            best, stale = value, 0
            torch.save({'model': model.state_dict(), 'seed': args.seed, 'epoch': epoch,
                        'score': value, 'stats': stats, 'configuration': cfg}, checkpoint)
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=False)['model'])
    model.eval()
    checks = invariance_checks(model, va, args.device)
    detailed = scenarios(va, encoder, cache, stats, source_hash, full=True, repeats=2)
    detailed.extend(short_gap_scenarios(va, encoder, cache, stats, source_hash))
    rows = validation_rows(model, va, detailed, args.device)
    write_csv(out / '验证集缺失消融.csv', rows)
    selected_score = score(rows)
    summary = {'seed': args.seed, 'best_epoch': int(max(history, key=lambda r:r['selection_score'])['epoch']),
               'quick_selection_score': best, 'full_validation_score': float(selected_score),
               'clean': rows[0], 'mean_random_missing_F1': float(np.mean([r['Macro_F1'] for r in rows[1:] if r['position'] == 'random'])),
               'mean_random_missing_MAE': float(np.mean([r['MAE'] for r in rows[1:] if r['position'] == 'random'])),
               'checkpoint_sha256': sha(checkpoint), 'invariance_checks': checks,
               'note': 'No attachment2 test or attachment3 used.'}
    baseline_path = Path(args.baseline_checkpoint)
    baseline_state = torch.load(baseline_path, map_location=args.device, weights_only=False)
    if baseline_state.get('method') != 'distilled' or baseline_state.get('seed') != args.seed:
        raise ValueError('The comparison checkpoint must be the matching distilled seed.')
    baseline = Fusion(gated=True).to(args.device)
    baseline.load_state_dict(baseline_state['model'])
    baseline.eval()
    baseline_quick = []
    for scenario in quick:
        from q2_experiment import predict as baseline_predict
        prob, intensity = baseline_predict(baseline, va, args.device, scenario)
        baseline_quick.append({**metrics(va['y'], va['r'], prob, intensity),
                               'subset': scenario['subset'], 'rate': scenario['rate'],
                               'position': scenario['position'], 'replicate': scenario['replicate']})
    baseline_score = float(score(baseline_quick))
    summary['baseline_quick_score'] = baseline_score
    summary['baseline_clean_Macro_F1'] = baseline_quick[0]['Macro_F1']
    summary['acceptance_passed'] = bool(best >= baseline_score + .01 and
                                        rows[0]['Macro_F1'] >= baseline_quick[0]['Macro_F1'] - .005)
    stress_rows = [r for r in rows if r['position'] == 'scattered-short']
    summary['short_gap_mean_F1'] = float(np.mean([r['Macro_F1'] for r in stress_rows]))
    baseline_stress = []
    for scenario in detailed:
        if scenario['position'] != 'scattered-short':
            continue
        prob, intensity = baseline_predict(baseline, va, args.device, scenario)
        baseline_stress.append({**metrics(va['y'], va['r'], prob, intensity),
                                'subset': scenario['subset'], 'rate': scenario['rate'],
                                'position': scenario['position'], 'replicate': scenario['replicate']})
    summary['baseline_short_gap_mean_F1'] = float(np.mean([r['Macro_F1'] for r in baseline_stress]))
    write_csv(out / '原版单种子相同场景基线.csv', baseline_quick + baseline_stress)
    dump(out / '验证结论.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='E题数据/E题数据')
    parser.add_argument('--out', default='outputs/question2/问题2_交叉模态一致性实验')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--baseline-checkpoint', default='outputs/question2/问题2_首轮实验结果/问题2_门控蒸馏_种子2026.pt')
    main(parser.parse_args())
