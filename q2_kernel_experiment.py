"""问题2的独立浅层对照：冻结BERT特征上的缺失增强与核特征分类器。

本脚本只用附件2 train 拟合，valid 比较；不读取 test 或附件3来选型。
在模拟文本缺失时，先遮挡 token 再重跑 BERT，避免上下文泄漏。
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import StandardScaler

from q2_experiment import (ROOT, TextEncoder, fit_normalization, metrics, normalize,
                           prepare_split, runtime, scenarios, seed_all, sha, write_csv)
from q2_ensemble import selection_score


def pooling(split: dict, scenario: dict) -> np.ndarray:
    mask = scenario['mask']
    text = split['t'] if scenario.get('t') is None else scenario['t']
    vectors = []
    for index, x in enumerate((text, split['a'], split['v'])):
        present = mask[:, :, index].astype(np.float32)
        count = np.maximum(present.sum(axis=1, keepdims=True), 1.)
        array = np.asarray(x, np.float32)
        mean = np.sum(array * present[:, :, None], axis=1) / count
        if index:
            square = np.sum(array * array * present[:, :, None], axis=1) / count
            std = np.sqrt(np.maximum(square - mean * mean, 0.))
            vectors.extend((mean, std))
        else:
            vectors.append(mean)
    quality = mask.sum(axis=1) / np.maximum(split['valid'].sum(axis=1, keepdims=True), 1)
    vectors.append(quality.astype(np.float32))
    return np.concatenate(vectors, axis=1).astype(np.float32)


def consistent_regression(class_index: np.ndarray, raw: np.ndarray) -> np.ndarray:
    value = np.asarray(raw, np.float32).copy()
    value[class_index == 1] = 0.
    value[class_index == 0] = np.minimum(value[class_index == 0], -.01)
    value[class_index == 2] = np.maximum(value[class_index == 2], .01)
    return np.clip(value, -3, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=ROOT / 'E题数据/E题数据')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/问题2_核特征浅层对照')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runtime(args.device)
    seed_all(2026)
    source = args.data_root / '附件2-数据集特征文件/aligned_50.pkl'
    source_hash = sha(source)
    with source.open('rb') as stream:
        data = pickle.load(stream)
    # 只从这个容器取 train/valid 两个划分，绝不访问 test 的样本或标签。
    encoder = TextEncoder(args.device)
    cache = ROOT / 'work/问题2_特征缓存'
    train = prepare_split(data['train'], encoder, cache, 'train', source_hash)
    valid = prepare_split(data['valid'], encoder, cache, 'valid', source_hash)
    stats = fit_normalization(train)
    normalize(train, stats); normalize(valid, stats)
    train_suite = scenarios(train, encoder, cache, stats, source_hash)
    valid_suite = scenarios(valid, encoder, cache, stats, source_hash)
    clean = pooling(train, train_suite[0])
    # 每条训练样本另取一个预先固定的30%连续缺失情形，避免重复同一句的全部五种变体。
    rng = np.random.default_rng(2026)
    choices = rng.integers(1, len(train_suite), len(train['ids']))
    augmented = np.empty_like(clean)
    for number in range(1, len(train_suite)):
        indices = np.flatnonzero(choices == number)
        if len(indices):
            augmented[indices] = pooling(train, train_suite[number])[indices]
    X = np.concatenate([clean, augmented])
    y = np.tile(train['y'], 2)
    intensity = np.tile(train['r'], 2)
    scale = StandardScaler().fit(clean)
    X = np.clip(scale.transform(X), -8, 8).astype(np.float32)
    # 三个预先给定的候选；比较时只用 valid 的固定 quick 情形。
    variants = [('linear', None, False), ('linear_balanced', None, True),
                ('nystroem_rbf', Nystroem(kernel='rbf', gamma=1. / X.shape[1],
                                         n_components=384, random_state=2026), True)]
    rows = []
    for name, transform, balanced in variants:
        Z = X if transform is None else transform.fit_transform(X)
        weights = None
        if balanced:
            counts = np.bincount(train['y'], minlength=3)
            weights = np.sqrt(len(train['y']) / (3. * counts))[y]
        classifier = RidgeClassifier(alpha=10., class_weight=None)
        classifier.fit(Z, y, sample_weight=weights)
        regressor = Ridge(alpha=20.).fit(Z, intensity)
        scenario_rows = []
        for scenario in valid_suite:
            query = np.clip(scale.transform(pooling(valid, scenario)), -8, 8).astype(np.float32)
            query = query if transform is None else transform.transform(query)
            logits = classifier.decision_function(query)
            pred = np.argmax(logits, axis=1)
            prob = np.eye(3, dtype=np.float32)[pred]
            estimate = consistent_regression(pred, regressor.predict(query))
            result = {'candidate': name, 'subset': scenario['subset'], 'rate': scenario['rate']}
            result.update(metrics(valid['y'], valid['r'], prob, estimate))
            rows.append(result); scenario_rows.append(result)
        score = selection_score(scenario_rows)
        print(json.dumps({'candidate': name, 'score': score, 'clean_F1': scenario_rows[0]['Macro_F1'],
                          'missing_F1': float(np.mean([x['Macro_F1'] for x in scenario_rows[1:]]))}), flush=True)
    write_csv(args.out / '问题2_核特征候选验证.csv', rows)
    scores = {name: selection_score([r for r in rows if r['candidate'] == name]) for name, _, _ in variants}
    best = max(scores, key=scores.get)
    (args.out / '问题2_核特征结论.json').write_text(json.dumps({
        'scores': scores, 'best': best, 'training_samples': len(train['ids']),
        'validation_samples': len(valid['ids']), 'train_views_per_sample': 2,
        'model_selection_source': 'valid only', 'attachment2_test_used': False,
        'attachment3_used': False, 'feature_rule': 'missing text reencoded by BERT after masking',
        'data_sha256': source_hash, 'bert_revision': encoder.revision,
    }, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
