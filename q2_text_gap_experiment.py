"""Q2 train/valid experiment: long and complete text gaps during training.

The existing clean teacher, selected ensemble and attachment-3 predictions are
left untouched. Attachment-2 test labels are never used for training or model
selection. This is one predeclared single-seed augmentation comparison.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from q2_ensemble import load_models, selection_score, summarize
from q2_experiment import (ROOT, Fusion, TextEncoder, block_mask, dump,
                           fit_normalization, norm_text, normalize, prepare_split,
                           predict, runtime, scenarios, seed_all, sha,
                           tensor_batch, text_features, write_csv)


def text_banks(train, encoder, cache, stats, source_hash):
    banks = {}
    for rate in (.1, .3, .5, .7):
        hidden = block_mask(train['valid'], rate, 51000 + int(rate * 100))
        observed = train['mask'][:, :, 0] & ~hidden
        encoded = text_features(encoder, train['b'], observed, cache,
                                'train_long_gap', source_hash)
        value = norm_text(encoded, observed, stats)
        if not np.all(value[~observed] == 0):
            raise AssertionError('Hidden text has a nonzero feature vector')
        banks[rate] = {'hide': hidden, 'text': value}
        print(f'prepared train text gap {rate:.1f}', flush=True)
    return banks


def check_no_text_leakage(train, encoder, banks):
    """Hidden lexical IDs may change, but encoded observed tokens must not."""
    hidden = banks[.7]['hide'][0]
    observed = train['mask'][0:1, :, 0] & ~hidden[None]
    original = train['b'][0:1].copy()
    altered = original.copy()
    altered[0, 0, hidden] = 2023
    a = encoder.encode(original, observed, batch=1)
    b = encoder.encode(altered, observed, batch=1)
    if not np.array_equal(a, b):
        raise AssertionError('Masked lexical IDs leaked through BERT context')
    return {'hidden_token_mutation_invariant': True,
            'hidden_feature_vectors_zero': True,
            'full_text_blackout_zero_and_masked': True}


def augment_batch(train, indices, banks):
    """Predeclared 40% clean, 25% ordinary, 20% long T, 15% full T."""
    mask = train['mask'][indices].copy()
    text = train['t'][indices].copy()
    changed = np.zeros(len(indices), dtype=bool)
    for j, sample in enumerate(indices):
        draw = np.random.rand()
        if draw < .25:
            rate = random.choice((.1, .3, .5))
            subset = random.choice(('T', 'A', 'V', 'TA', 'TV', 'AV', 'TAV'))
            hidden = banks[rate]['hide'][sample]
            for modality in subset:
                mask[j, :, 'TAV'.index(modality)] &= ~hidden
            if 'T' in subset:
                text[j] = banks[rate]['text'][sample]
            changed[j] = True
        elif draw < .45:
            rate = random.choice((.5, .7))
            hidden = banks[rate]['hide'][sample]
            mask[j, :, 0] &= ~hidden
            text[j] = banks[rate]['text'][sample]
            changed[j] = True
        elif draw < .60:
            mask[j, :, 0] = False
            text[j] = 0.
            changed[j] = True
    if not np.all(text[~mask[:, :, 0]] == 0):
        raise AssertionError('Text hidden from the mask remains nonzero')
    return mask, text, changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=ROOT / 'E题数据/E题数据')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/问题2_长文本缺失训练实验')
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
    comparison_path = original_dir / f'问题2_门控蒸馏_种子{args.seed}.pt'
    existing, checkpoints = load_models({'teacher': teacher_path,
                                         'comparison': comparison_path}, args.device)
    teacher, comparison = existing['teacher'], existing['comparison']
    stats = checkpoints['teacher']['stats']
    if checkpoints['comparison']['stats'] != stats:
        raise ValueError('Reference checkpoints have different train normalization')
    for checkpoint in checkpoints.values():
        if checkpoint['provenance']['data_sha256'] != source_hash:
            raise ValueError('Reference checkpoint source-data SHA-256 mismatch')
    encoder = TextEncoder(args.device,
                          revision=checkpoints['teacher']['provenance']['bert_revision'])
    # This pickle holds all splits, but the experiment only selects train/valid.
    with source.open('rb') as f:
        data = pickle.load(f)
    cache = ROOT / 'work/问题2_特征缓存'
    train = prepare_split(data['train'], encoder, cache, 'train', source_hash)
    valid = prepare_split(data['valid'], encoder, cache, 'valid', source_hash)
    fitted_stats = fit_normalization(train)
    if fitted_stats != stats:
        raise ValueError('Train-only normalization differs from existing model')
    normalize(train, stats)
    normalize(valid, stats)
    banks = text_banks(train, encoder, cache, stats, source_hash)
    leakage_checks = check_no_text_leakage(train, encoder, banks)
    quick = scenarios(valid, encoder, cache, stats, source_hash)
    teacher_prob, teacher_reg = predict(teacher, train, args.device)
    teacher_eligible = (teacher_prob.argmax(1) == train['y']) & (
        np.abs(teacher_reg - train['r']) < 1.)
    model = Fusion(gated=True).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    weights = torch.as_tensor(np.sqrt(len(train['y']) / (3 * np.bincount(train['y'], minlength=3))),
                              dtype=torch.float32, device=args.device)
    provenance = {'source_sha256': source_hash, 'script_sha256': sha(__file__),
                  'teacher_sha256': sha(teacher_path), 'comparison_sha256': sha(comparison_path),
                  'bert_revision': encoder.revision, 'seed': args.seed,
                  'python': sys.version.split()[0], 'torch': torch.__version__,
                  'training': '40% clean, 25% standard 10/30/50% gaps, '
                              '20% text 50/70% contiguous gaps, 15% whole-text blackout',
                  'same_as_existing': 'Fusion architecture, weighted CE + 0.6 SmoothL1, '
                                      'conditional KD, AdamW 3e-4, wd 1e-3, batch64, '
                                      'patience7, same validation selection score',
                  'evaluation_predeclared': 'clean, T 50/70%, TAV 50/70%, '
                                            'all 56 missing scenarios',
                  'selection_data': 'validation only; attachment-2 test and attachment-3 '
                                    'are not used for design, training or selection'}
    dump(args.out / '问题2_实验配置.json', provenance)
    checkpoint_path = args.out / f'问题2_长文本缺失蒸馏_种子{args.seed}.pt'
    best_score = -float('inf')
    best_epoch = None
    bad_epochs = 0
    logs = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = np.random.permutation(len(train['ids']))
        losses = []
        for start in range(0, len(permutation), 64):
            idx = permutation[start:start + 64]
            mask, text, changed = augment_batch(train, idx, banks)
            logits, reg = model(*tensor_batch(train, idx, args.device, text, mask))
            y = torch.as_tensor(train['y'][idx], device=args.device)
            r = torch.as_tensor(train['r'][idx], device=args.device)
            loss = F.cross_entropy(logits, y, weight=weights) + .6 * F.smooth_l1_loss(reg, r)
            eligible = changed & teacher_eligible[idx]
            if eligible.any():
                tm = torch.as_tensor(eligible, device=args.device)
                probability = torch.as_tensor(teacher_prob[idx][eligible], device=args.device)
                soft = F.softmax(torch.log(probability.clamp_min(1e-8)) / 2, -1)
                loss += .2 * 4 * F.kl_div(F.log_softmax(logits[tm] / 2, -1),
                                           soft, reduction='batchmean')
                loss += .1 * F.smooth_l1_loss(reg[tm],
                    torch.as_tensor(teacher_reg[idx][eligible], device=args.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        rows, _ = summarize(['new'], {'new': model}, valid, quick, args.device)
        score = selection_score(rows)
        logs.append({'epoch': epoch, 'train_loss': float(np.mean(losses)),
                     'validation_score': score,
                     'clean_Macro_F1': rows[0]['Macro_F1'],
                     'missing_30pct_Macro_F1': float(np.mean([x['Macro_F1'] for x in rows[1:]])),
                     'clean_MAE': rows[0]['MAE']})
        print(json.dumps(logs[-1], ensure_ascii=False), flush=True)
        if score > best_score + 1e-4:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save({'model': model.state_dict(), 'method': 'long_text_gap_distilled',
                        'seed': args.seed, 'epoch': epoch, 'selection_score': score,
                        'stats': stats, 'provenance': provenance,
                        'architecture': {'hidden': 96, 'dropout': .2}}, checkpoint_path)
        else:
            bad_epochs += 1
        write_csv(args.out / '问题2_训练记录.csv', logs)
        if bad_epochs >= 7:
            break
    model.load_state_dict(torch.load(checkpoint_path, map_location=args.device,
                                     weights_only=False)['model'])
    model.eval()
    # If text is fully masked, arbitrary text tensor values must have no effect.
    sample_idx = np.arange(min(8, len(valid['ids'])))
    full_text_mask = valid['mask'][sample_idx].copy()
    full_text_mask[:, :, 0] = False
    inputs = tensor_batch(valid, sample_idx, args.device,
                          np.zeros_like(valid['t'][sample_idx]), full_text_mask)
    with torch.inference_mode():
        original_output = model(*inputs)
        changed_inputs = [x.clone() for x in inputs]
        changed_inputs[0].fill_(12345.)
        changed_output = model(*changed_inputs)
        if not all(torch.allclose(a, b, atol=1e-6)
                   for a, b in zip(original_output, changed_output)):
            raise AssertionError('Full-text blackout allows text values to affect predictions')
    leakage_checks['full_text_prediction_value_invariant'] = True

    full_suite = scenarios(valid, encoder, cache, stats, source_hash,
                           full=True, repeats=2)
    results = []
    for name, candidate in [('existing_distilled_2026', comparison),
                            ('long_text_gap_distilled_2026', model)]:
        rows, _ = summarize([name], {name: candidate}, valid,
                            full_suite, args.device)
        results.extend([{'model': name, **row} for row in rows])
    write_csv(args.out / '问题2_验证集完整缺失消融对照.csv', results)

    def report(name):
        rows = [x for x in results if x['model'] == name]
        key = lambda subset, rate: [r['Macro_F1'] for r in rows if
            r['subset'] == subset and r['rate'] == rate and
            r['position'] == 'random']
        clean = next(r for r in rows if r['subset'] == 'none')
        quick_rows, _ = summarize([name], {name: comparison if name.startswith('existing') else model},
                                  valid, quick, args.device)
        return {'selection_score': selection_score(quick_rows),
                'clean_Macro_F1': clean['Macro_F1'], 'clean_MAE': clean['MAE'],
                'T_50pct_F1': float(np.mean(key('T', .5))),
                'T_70pct_F1': float(np.mean(key('T', .7))),
                'TAV_50pct_F1': float(np.mean(key('TAV', .5))),
                'TAV_70pct_F1': float(np.mean(key('TAV', .7))),
                'all_56_missing_Macro_F1': float(np.mean([r['Macro_F1'] for r in rows
                    if r['position'] == 'random' and r['rate'] > 0]))}
    before = report('existing_distilled_2026')
    after = report('long_text_gap_distilled_2026')
    decision = {'best_epoch': best_epoch, 'before': before, 'after': after,
                'deltas_after_minus_before': {k: after[k] - before[k] for k in before},
                'predeclared_adoption_rule': 'selection_score at least +0.005; '
                    'clean F1 no worse than -0.01; T70 F1 at least +0.02; '
                    'all56 F1 at least +0.01',
                'meets_adoption_rule': bool(after['selection_score'] >= before['selection_score'] + .005
                    and after['clean_Macro_F1'] >= before['clean_Macro_F1'] - .01
                    and after['T_70pct_F1'] >= before['T_70pct_F1'] + .02
                    and after['all_56_missing_Macro_F1'] >= before['all_56_missing_Macro_F1'] + .01),
                'text_leakage_checks': leakage_checks,
                'test_or_attachment3_used_for_selection': False}
    dump(args.out / '问题2_长文本缺失训练结论.json', decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
