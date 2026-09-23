"""Validation-only, video-grouped cross-fit of one neutral-class logit offset.

This diagnostic does not retrain models, read attachment-2 test, or produce
attachment-3 predictions. Existing selected-model files remain untouched.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from q2_ensemble import load_models
from q2_experiment import (ROOT, TextEncoder, dump, metrics, normalize,
                           prepare_split, predict, runtime, scenarios, sha,
                           write_csv)


GRID = np.round(np.arange(-1.0, 1.0001, .1), 5)


def adjust(prob, regression, bias):
    """Multiply only the neutral probability by exp(bias), then renormalize."""
    q = prob.copy()
    q[:, 1] *= np.exp(bias)
    q /= q.sum(axis=1, keepdims=True)
    r = regression.copy()
    cls = q.argmax(1)
    r[cls == 1] = 0.
    r[cls == 0] = np.minimum(r[cls == 0], -.01)
    r[cls == 2] = np.maximum(r[cls == 2], .01)
    return q, r


def summarize(y, r, banks, idx, bias):
    rows = []
    for p, pr in banks:
        q, rr = adjust(p[idx], pr[idx], bias)
        rows.append(metrics(y[idx], r[idx], q, rr))
    f1_missing = float(np.mean([row['Macro_F1'] for row in rows[1:]]))
    mae_missing = float(np.mean([row['MAE'] for row in rows[1:]]))
    score = .5 * (rows[0]['Macro_F1'] + f1_missing) - .05 * (rows[0]['MAE'] + mae_missing)
    return {'score': score, 'clean_Macro_F1': rows[0]['Macro_F1'],
            'missing_Macro_F1': f1_missing, 'clean_MAE': rows[0]['MAE'],
            'missing_MAE': mae_missing, 'clean_Accuracy': rows[0]['Accuracy']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=ROOT / 'E题数据/E题数据')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'outputs/问题2_优化实验结果')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/问题2_中性偏置交叉验证')
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    runtime(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.data_root / '附件2-数据集特征文件/aligned_50.pkl'
    source_hash = sha(source)
    manifest = json.loads((args.model_dir / '问题2_集成选模结论.json').read_text(encoding='utf-8'))
    if manifest['selected'] != 'distilled' or manifest['data_sha256'] != source_hash:
        raise ValueError('Existing ensemble selection or source-data SHA-256 does not match')
    paths = {name: args.model_dir / '模型参数' / f'{name}.pt' for name in manifest['members']}
    if any(sha(path) != manifest['checkpoint_sha256'][name] for name, path in paths.items()):
        raise ValueError('Model checkpoint SHA-256 does not match manifest')
    models, checkpoints = load_models(paths, args.device)
    stats = next(iter(checkpoints.values()))['stats']
    if any(ck['stats'] != stats for ck in checkpoints.values()):
        raise ValueError('Normalization statistics differ between checkpoints')
    encoder = TextEncoder(args.device, revision=manifest['bert_revision'])
    with source.open('rb') as f:
        original = pickle.load(f)['valid']  # The pickle holds every split; use valid only.
    valid = normalize(prepare_split(original, encoder, ROOT / 'work/问题2_特征缓存',
                                    'valid', source_hash), stats)
    suite = scenarios(valid, encoder, ROOT / 'work/问题2_特征缓存', stats, source_hash)
    banks = []
    for scenario in suite:
        predictions = [predict(models[name], valid, args.device, scenario)
                       for name in manifest['members']]
        banks.append((np.mean([x[0] for x in predictions], axis=0),
                      np.mean([x[1] for x in predictions], axis=0)))
    y, r = valid['y'], valid['r']
    ids = np.arange(len(y))
    groups = np.asarray([sid.split('$_$')[0] for sid in valid['ids']])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
    fold_rows, curves = [], []
    bias_by_row = np.zeros(len(y), dtype=np.float32)
    for fold, (fit_idx, hold_idx) in enumerate(splitter.split(ids, y, groups), 1):
        if set(groups[fit_idx]) & set(groups[hold_idx]):
            raise AssertionError('A video group appears on both sides of a fold')
        curve = [(float(b), summarize(y, r, banks, fit_idx, float(b))) for b in GRID]
        choice = max(curve, key=lambda z: (z[1]['score'], -abs(z[0])))
        bias = choice[0]
        bias_by_row[hold_idx] = bias
        before = summarize(y, r, banks, hold_idx, 0.)
        after = summarize(y, r, banks, hold_idx, bias)
        fold_rows.append({'fold': fold, 'holdout_n': len(hold_idx),
                          'holdout_video_groups': len(set(groups[hold_idx])),
                          'holdout_class_counts': np.bincount(y[hold_idx], minlength=3).tolist(),
                          'fit_selected_neutral_bias': bias,
                          'fit_score': choice[1]['score'],
                          'holdout_score_before': before['score'],
                          'holdout_score_after': after['score'],
                          'holdout_clean_F1_before': before['clean_Macro_F1'],
                          'holdout_clean_F1_after': after['clean_Macro_F1'],
                          'holdout_missing_F1_before': before['missing_Macro_F1'],
                          'holdout_missing_F1_after': after['missing_Macro_F1']})
        curves.extend({'fold': fold, 'bias': b, **row} for b, row in curve)

    oof_banks = []
    for p, pr in banks:
        q = np.zeros_like(p)
        rr = np.zeros_like(pr)
        for b in np.unique(bias_by_row):
            subset = bias_by_row == b
            q[subset], rr[subset] = adjust(p[subset], pr[subset], float(b))
        oof_banks.append((q, rr))
    before = summarize(y, r, banks, ids, 0.)
    # Reuse summarize on already adjusted OOF predictions with zero extra bias.
    after = summarize(y, r, oof_banks, ids, 0.)
    q_before, _ = adjust(*banks[0], 0.)
    q_after, _ = adjust(*oof_banks[0], 0.)
    recall_before = [float(np.mean(q_before.argmax(1)[y == k] == k)) for k in range(3)]
    recall_after = [float(np.mean(q_after.argmax(1)[y == k] == k)) for k in range(3)]
    result = {'purpose': 'validation-only neutral-logit calibration diagnostic',
              'data_sha256': source_hash, 'model_checkpoint_sha256': manifest['checkpoint_sha256'],
              'grid': GRID.tolist(), 'grouping': 'video ID, 5-fold stratified grouped cross-fit',
              'folds': fold_rows, 'OOF_before': before, 'OOF_after': after,
              'OOF_recall_before_Negative_Neutral_Positive': recall_before,
              'OOF_recall_after_Negative_Neutral_Positive': recall_after,
              'test_used_for_selection_or_metrics': False,
              'attachment3_read': False,
              'interpretation_limit': 'The underlying checkpoint was itself selected on full validation, so cross-fit offset diagnostics are not a new independent holdout.'}
    dump(args.out / '问题2_中性偏置交叉验证结论.json', result)
    write_csv(args.out / '问题2_分组折外表现.csv', fold_rows)
    write_csv(args.out / '问题2_拟合折偏置扫描.csv', curves)
    print(json.dumps({'OOF_before': before, 'OOF_after': after,
                      'biases': [x['fit_selected_neutral_bias'] for x in fold_rows],
                      'recall_before': recall_before, 'recall_after': recall_after},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
