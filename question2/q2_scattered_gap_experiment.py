"""问题2独立实验：用短而分散的共同缺口增强现有门控蒸馏模型。

Only attachment-2 train is fitted and only valid is used for model selection.
The existing checkpoints, official predictions, test split and GitHub are untouched.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from q2_ensemble import load_models, summarize
from q2_experiment import (ROOT, Fusion, TextEncoder, block_mask, dump,
                           fit_normalization, norm_text, normalize, prepare_split,
                           predict, runtime, scenarios, seed_all, sha,
                           tensor_batch, text_features, write_csv)


def scattered_mask(valid: np.ndarray, rate: float, seed: int) -> np.ndarray:
    """Mask a target proportion as mostly separate 1--4-content-token runs."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(valid, dtype=bool)
    for i, row in enumerate(valid):
        positions = np.flatnonzero(row)
        length = len(positions)
        target = min(length, max(1, round(rate * length))) if rate else 0
        hidden = np.zeros(length, dtype=bool)
        while hidden.sum() < target:
            remaining = target - int(hidden.sum())
            run = min(remaining, int(rng.choice([1, 2, 3, 4], p=[.37, .31, .22, .10])))
            candidates = []
            for start in range(length - run + 1):
                end = start + run
                if hidden[start:end].any():
                    continue
                if (start and hidden[start - 1]) or (end < length and hidden[end]):
                    continue
                candidates.append(start)
            if not candidates:
                # Short utterances may have too little room for separated runs.
                free = np.flatnonzero(~hidden)
                hidden[int(rng.choice(free))] = True
                continue
            start = int(rng.choice(candidates))
            hidden[start:start + run] = True
        out[i, positions[hidden]] = True
    return out


def get_banks(train, encoder, cache, stats, source_hash):
    blocks = {}
    scattered = {}
    for rate in (.1, .3, .5):
        hide = block_mask(train['valid'], rate, 51000 + int(rate * 100))
        obs = train['mask'][:, :, 0] & ~hide
        text = text_features(encoder, train['b'], obs, cache, 'train_aug', source_hash)
        blocks[rate] = {'hide': hide, 'text': norm_text(text, obs, stats)}
        print(f'prepared contiguous {rate:.1f}', flush=True)
    for rate in (.1, .2, .3):
        hide = scattered_mask(train['valid'], rate, 91200 + int(rate * 100))
        obs = train['mask'][:, :, 0] & ~hide
        text = text_features(encoder, train['b'], obs, cache, 'train_scattered', source_hash)
        scattered[rate] = {'hide': hide, 'text': norm_text(text, obs, stats)}
        print(f'prepared scattered {rate:.1f}', flush=True)
    return blocks, scattered


def make_scattered_suite(valid, encoder, cache, stats, source_hash):
    suite = []
    for rate in (.1, .2, .3):
        for rep in (0, 1):
            hide = scattered_mask(valid['valid'], rate, 121000 + 1009 * rep + int(rate * 100))
            mask = valid['mask'].copy()
            mask &= ~hide[:, :, None]  # all three modalities share the same short gaps
            observed = mask[:, :, 0]
            encoded = text_features(encoder, valid['b'], observed, cache,
                                    'valid_scattered', source_hash)
            text = norm_text(encoded, observed, stats)
            suite.append({'subset': 'TAV', 'rate': rate, 'position': 'scattered',
                          'replicate': rep, 'mask': mask, 't': text})
    return suite


def augment_batch(train, indices, blocks, scattered):
    """Fixed 35% clean, 30% original contiguous, 35% scattered TAV mixture."""
    mask = train['mask'][indices].copy()
    text = train['t'][indices].copy()
    changed = np.zeros(len(indices), bool)
    for row, sample in enumerate(indices):
        draw = np.random.random()
        if draw < .35:
            continue
        if draw < .65:
            bank = blocks[random.choice((.1, .3, .5))]
            subset = random.choice(('T', 'A', 'V', 'TA', 'TV', 'AV', 'TAV'))
        else:
            bank = scattered[random.choice((.1, .2, .3))]
            subset = 'TAV'
        hide = bank['hide'][sample]
        for modality in subset:
            mask[row, :, 'TAV'.index(modality)] &= ~hide
        if 'T' in subset:
            text[row] = bank['text'][sample]
        changed[row] = True
    if np.any(text[~mask[:, :, 0]] != 0):
        raise AssertionError('Text remained nonzero behind a missingness mask')
    return mask, text, changed


def score(quick_rows, scattered_rows):
    """Predeclared score: F1 40/30/30, MAE penalty 0.1 with same weights."""
    clean = quick_rows[0]
    old_f1 = float(np.mean([r['Macro_F1'] for r in quick_rows[1:]]))
    old_mae = float(np.mean([r['MAE'] for r in quick_rows[1:]]))
    new_f1 = float(np.mean([r['Macro_F1'] for r in scattered_rows]))
    new_mae = float(np.mean([r['MAE'] for r in scattered_rows]))
    weighted_f1 = .4 * clean['Macro_F1'] + .3 * old_f1 + .3 * new_f1
    weighted_mae = .4 * clean['MAE'] + .3 * old_mae + .3 * new_mae
    return weighted_f1 - .1 * weighted_mae


def profile(name, model, valid, quick, scattered, device):
    old, _ = summarize([name], {name: model}, valid, quick, device)
    new, _ = summarize([name], {name: model}, valid, scattered, device)
    return {'selection_score': score(old, new),
            'clean_F1': old[0]['Macro_F1'], 'clean_MAE': old[0]['MAE'],
            'contiguous_30pct_mean_F1': float(np.mean([r['Macro_F1'] for r in old[1:]])),
            'contiguous_30pct_mean_MAE': float(np.mean([r['MAE'] for r in old[1:]])),
            'scattered_TAV_mean_F1': float(np.mean([r['Macro_F1'] for r in new])),
            'scattered_TAV_mean_MAE': float(np.mean([r['MAE'] for r in new]))}, old, new


def check_text_leakage(train, encoder, scattered):
    hide = scattered[.3]['hide'][0]
    observed = train['mask'][0:1, :, 0] & ~hide[None]
    original = train['b'][0:1].copy()
    corrupted = original.copy()
    corrupted[0, 0, hide] = 2023
    if not np.array_equal(encoder.encode(original, observed, 1),
                          encoder.encode(corrupted, observed, 1)):
        raise AssertionError('Masked text token leaked into BERT encoding')
    return {'hidden_text_id_mutation_invariant': True,
            'hidden_text_features_zero': bool(np.all(
                scattered[.3]['text'][0:1][~observed] == 0))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=ROOT / 'E题数据/E题数据')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/问题2_短点状共同缺失实验')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    runtime(args.device)
    seed_all(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.data_root / '附件2-数据集特征文件/aligned_50.pkl'
    source_hash = sha(source)
    original_dir = ROOT / 'outputs/问题2_首轮实验结果'
    teacher_path = original_dir / f'问题2_普通融合_种子{args.seed}.pt'
    baseline_path = original_dir / f'问题2_门控蒸馏_种子{args.seed}.pt'
    models, checkpoints = load_models({'teacher': teacher_path,
                                       'baseline': baseline_path}, args.device)
    teacher, baseline = models['teacher'], models['baseline']
    stats = checkpoints['teacher']['stats']
    if checkpoints['baseline']['stats'] != stats:
        raise ValueError('Reference normalization differs')
    for checkpoint in checkpoints.values():
        if checkpoint['provenance']['data_sha256'] != source_hash:
            raise ValueError('Reference data SHA-256 differs')
    encoder = TextEncoder(args.device,
                          revision=checkpoints['teacher']['provenance']['bert_revision'])
    with source.open('rb') as stream:
        data = pickle.load(stream)
    cache = ROOT / 'work/问题2_特征缓存'
    train = prepare_split(data['train'], encoder, cache, 'train', source_hash)
    valid = prepare_split(data['valid'], encoder, cache, 'valid', source_hash)
    if fit_normalization(train) != stats:
        raise ValueError('Training normalization differs from reference')
    normalize(train, stats)
    normalize(valid, stats)

    policy = {'source_sha256': source_hash, 'script_sha256': sha(__file__),
              'teacher_sha256': sha(teacher_path), 'baseline_sha256': sha(baseline_path),
              'bert_revision': encoder.revision, 'seed': args.seed,
              'python': sys.version.split()[0], 'torch': torch.__version__,
              'training': '35% clean, 30% original contiguous 10/30/50% random subset, '
                          '35% scattered shared TAV 10/20/30% runs mostly 1-4 tokens',
              'architecture': 'unchanged Fusion(gated=True), hidden=96, dropout=0.2',
              'loss': 'weighted CE + 0.6 SmoothL1; eligible clean teacher prediction: '
                      '0.2*4 KL(T=2) + 0.1 SmoothL1',
              'optimizer': 'AdamW lr=3e-4 wd=1e-3 batch64 patience7 clip1',
              'selection_score': '0.4 clean F1 + 0.3 original quick mean F1 + '
                                 '0.3 scattered TAV mean F1 - 0.1 same-weight MAE',
              'validation': 'complete, T/A/V/AV/TAV 30% contiguous, '
                            'TAV scattered 10/20/30% x2 masks',
              'predeclared_acceptance': 'score +0.005; clean F1 >=baseline-0.01; '
                                        'contiguous mean F1 >=baseline-0.01; '
                                        'scattered mean F1 >=baseline+0.02',
              'test_or_attachment3_used_for_selection': False}
    dump(args.out / '问题2_实验配置.json', policy)
    blocks, scattered = get_banks(train, encoder, cache, stats, source_hash)
    leakage_checks = check_text_leakage(train, encoder, scattered)
    quick = scenarios(valid, encoder, cache, stats, source_hash)
    sparse_suite = make_scattered_suite(valid, encoder, cache, stats, source_hash)
    before, before_old, before_new = profile('baseline', baseline, valid, quick,
                                              sparse_suite, args.device)
    dump(args.out / '问题2_基线预评估.json', before)
    write_csv(args.out / '问题2_验证集固定场景_基线.csv',
              [{'model': 'existing_distilled_2026', **r} for r in before_old + before_new])

    teacher_prob, teacher_reg = predict(teacher, train, args.device)
    teacher_eligible = (teacher_prob.argmax(1) == train['y']) & (
        np.abs(teacher_reg - train['r']) < 1.)
    model = Fusion(gated=True).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    weights = torch.as_tensor(np.sqrt(len(train['y']) /
                                     (3 * np.bincount(train['y'], minlength=3))),
                              dtype=torch.float32, device=args.device)
    best = -float('inf')
    best_epoch = 0
    bad = 0
    logs = []
    checkpoint_path = args.out / f'问题2_短点状共同缺失蒸馏_种子{args.seed}.pt'
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(train['ids']))
        losses = []
        for start in range(0, len(order), 64):
            idx = order[start:start + 64]
            mask, text, changed = augment_batch(train, idx, blocks, scattered)
            logits, reg = model(*tensor_batch(train, idx, args.device, text, mask))
            y = torch.as_tensor(train['y'][idx], device=args.device)
            r = torch.as_tensor(train['r'][idx], device=args.device)
            loss = F.cross_entropy(logits, y, weight=weights) + .6 * F.smooth_l1_loss(reg, r)
            eligible = changed & teacher_eligible[idx]
            if eligible.any():
                selected = torch.as_tensor(eligible, device=args.device)
                probability = torch.as_tensor(teacher_prob[idx][eligible], device=args.device)
                soft = F.softmax(torch.log(probability.clamp_min(1e-8)) / 2, -1)
                loss += .2 * 4 * F.kl_div(F.log_softmax(logits[selected] / 2, -1),
                                           soft, reduction='batchmean')
                loss += .1 * F.smooth_l1_loss(reg[selected],
                    torch.as_tensor(teacher_reg[idx][eligible], device=args.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        current, original_rows, sparse_rows = profile('candidate', model, valid,
                                                       quick, sparse_suite, args.device)
        logs.append({'epoch': epoch, 'training_loss': float(np.mean(losses)), **current})
        print(json.dumps(logs[-1], ensure_ascii=False), flush=True)
        if current['selection_score'] > best + 1e-4:
            best = current['selection_score']
            best_epoch = epoch
            bad = 0
            torch.save({'model': model.state_dict(), 'method': 'distilled',
                        'seed': args.seed, 'epoch': epoch,
                        'selection_score': best, 'stats': stats,
                        'provenance': policy, 'architecture': {'hidden': 96,
                        'dropout': .2}}, checkpoint_path)
        else:
            bad += 1
        write_csv(args.out / '问题2_训练记录.csv', logs)
        if bad >= 7:
            break
    model.load_state_dict(torch.load(checkpoint_path, map_location=args.device,
                                     weights_only=False)['model'])
    model.eval()
    after, after_old, after_new = profile('candidate', model, valid, quick,
                                          sparse_suite, args.device)
    write_csv(args.out / '问题2_验证集固定场景_对照.csv',
              [{'model': 'existing_distilled_2026', **r} for r in before_old + before_new] +
              [{'model': 'scattered_gap_distilled_2026', **r} for r in after_old + after_new])
    delta = {key: after[key] - before[key] for key in before}
    accepted = (delta['selection_score'] >= .005 and delta['clean_F1'] >= -.01
                and delta['contiguous_30pct_mean_F1'] >= -.01
                and delta['scattered_TAV_mean_F1'] >= .02)
    dump(args.out / '问题2_短点状共同缺失实验结论.json',
         {'baseline': before, 'candidate': after, 'delta': delta,
          'best_epoch': best_epoch, 'accepted': bool(accepted),
          'acceptance_rule': policy['predeclared_acceptance'],
          'text_leakage_checks': leakage_checks,
          'test_or_attachment3_used_for_selection': False})
    print('FINISHED', json.dumps({'accepted': accepted, 'baseline': before,
                                 'candidate': after, 'delta': delta},
                                ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
