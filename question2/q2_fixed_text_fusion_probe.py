"""Predeclared, label-free probability/intensity fusion on full MOSEI valid.

Compares the supervised BERT text classifier, frozen-BERT full-data Fusion
student, and fixed 50/20% text mixture. All text-gap scenarios are re-encoded
after redaction in BOTH encoders. The source contains train/valid only.
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import torch

from q2_ensemble import load_models
from q2_experiment import (ROOT, TextEncoder, dump, metrics, normalize,
                           predict, prepare_split, runtime, scenarios, seed_all,
                           sha, write_csv)
from q2_full_bert_text_experiment import (CONFIG_PATH as TEXT_CONFIG,
                                           DEFAULT_OUT as TEXT_OUT,
                                           FULL_SOURCE, SAFE_MANIFEST,
                                           SAFETY_AUDIT,
                                           SupervisedTextEncoder,
                                           load_delta, prepare_text,
                                           read_safe_manifest,
                                           read_safety_audit, tensors)
from q2_scattered_gap_experiment import make_scattered_suite

OUT = ROOT / 'outputs/question2/问题2_固定文本融合验证'
FULL_STUDENT = (ROOT / 'outputs/question2/问题2_完整MOSEI训练实验/'
                '完整数据_缺失蒸馏学生_种子2026.pt')


def consistency(probability, intensity):
    """Problem-defined deterministic class/intensity coherence, no labels."""
    intensity = intensity.copy()
    cls = probability.argmax(1)
    intensity[cls == 1] = 0.
    intensity[cls == 0] = np.minimum(intensity[cls == 0], -.01)
    intensity[cls == 2] = np.maximum(intensity[cls == 2], .01)
    return probability, intensity


def mixture(text_pred, fusion_pred, observation_ratio, zero_aware=False):
    if zero_aware:
        text_weight = np.where(observation_ratio == 0, 0.,
                               np.where(observation_ratio >= .5, .5, .2))
    else:
        text_weight = np.where(observation_ratio >= .5, .5, .2)
    probability = (text_weight[:, None] * text_pred[0] +
                   (1 - text_weight[:, None]) * fusion_pred[0])
    intensity = text_weight * text_pred[1] + (1 - text_weight) * fusion_pred[1]
    return consistency(probability, intensity), text_weight


@torch.inference_mode()
def predict_text(model, raw, observed, device, batch_size=64):
    """Re-encode given observed positions before BERT self-attention."""
    model.eval()
    probabilities = []
    intensities = []
    for start in range(0, len(raw['ids']), batch_size):
        indices = np.arange(start, min(start + batch_size, len(raw['ids'])))
        inputs = tensors(raw, indices, observed[indices], device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=device.startswith('cuda')):
            logits, intensity = model(*inputs)
        probabilities.append(logits.float().softmax(-1).cpu().numpy())
        intensities.append(intensity.float().cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(intensities)


def validation_score(rows):
    clean = next(r for r in rows if r['subset'] == 'none')
    contiguous = [r for r in rows if r['position'] == 'random' and
                  r['subset'] != 'none']
    scattered = [r for r in rows if r['position'] == 'scattered']
    if len(contiguous) != 5 or len(scattered) != 6:
        raise ValueError('Expected clean + five contiguous + six scattered scenarios')
    weighted_f1 = (.4 * clean['Macro_F1'] +
                   .3 * np.mean([r['Macro_F1'] for r in contiguous]) +
                   .3 * np.mean([r['Macro_F1'] for r in scattered]))
    weighted_mae = (.4 * clean['MAE'] +
                    .3 * np.mean([r['MAE'] for r in contiguous]) +
                    .3 * np.mean([r['MAE'] for r in scattered]))
    return float(weighted_f1 - .1 * weighted_mae)


def evaluate_scenario(text_model, fusion_model, text_raw, fusion_split,
                      scenario, device):
    observed = scenario['mask'][:, :, 0]
    text_raw_pred = predict_text(text_model, text_raw, observed, device)
    # Frozen BERT text features in scenario['t'] were also computed from
    # the redacted token sequence by scenarios/make_scattered_suite.
    fusion_raw_pred = predict(fusion_model, fusion_split, device, scenario)
    text_result = consistency(*text_raw_pred)
    fusion_result = consistency(*fusion_raw_pred)
    ratio = observed.sum(1) / fusion_split['valid'].sum(1).clip(min=1)
    mixed, weights = mixture(text_raw_pred, fusion_raw_pred, ratio)
    zero_aware, zero_weights = mixture(text_raw_pred, fusion_raw_pred, ratio,
                                       zero_aware=True)
    return ({'text_only': text_result, 'fusion_only': fusion_result,
             'fixed_50_20': mixed, 'zero_aware_fixed': zero_aware},
            {'text_weight_mean': float(weights.mean()),
             'text_weight_20pct_count': int((weights == .2).sum()),
             'text_observed_zero_count': int((ratio == 0).sum()),
             'zero_aware_text_weight_mean': float(zero_weights.mean())})


def run(args):
    runtime(args.device)
    seed_all(2026)
    args.out.mkdir(parents=True, exist_ok=True)
    source_hash = sha(args.safe_source)
    safety_hash = read_safety_audit(args.safety_audit)
    manifest_hash, manifest = read_safe_manifest(args.safe_manifest,
                                                  args.safe_source, source_hash,
                                                  safety_hash)
    text_model, text_checkpoint = load_delta(args.text_delta, args.device)
    text_provenance = text_checkpoint['provenance']
    if (text_provenance['full_source_sha256'] != source_hash or
            text_provenance['safe_train_valid_manifest_sha256'] != manifest_hash or
            text_provenance['independent_safety_audit_sha256'] != safety_hash):
        raise ValueError('Text delta has a different safe source or audit')
    fusion_models, fusion_checkpoints = load_models(
        {'fusion': args.fusion_checkpoint}, args.device)
    fusion_model = fusion_models['fusion']
    frozen = fusion_checkpoints['fusion']
    if frozen['method'] != 'distilled':
        raise ValueError('Expected the full-data missingness-trained student')
    frozen_source = frozen['provenance']['source_sha256']
    if frozen_source not in (source_hash, manifest['source_sha256']):
        raise ValueError('Frozen Fusion student used a different source')
    with args.safe_source.open('rb') as stream:
        data = pickle.load(stream)
    if set(data) != {'train', 'valid'}:
        raise ValueError('Evaluation source must physically exclude full test')
    raw_valid = data['valid']
    del data
    gc.collect()
    text_raw = prepare_text(raw_valid, 'valid')
    frozen_encoder = TextEncoder(args.device,
                                 revision=frozen['provenance']['bert_model'])
    cache = ROOT / 'work/问题2_特征缓存'
    # The safe pickle contains the original train/valid examples. Using the
    # original source digest allows exact reuse of frozen-BERT caches.
    frozen_valid = normalize(prepare_split(raw_valid, frozen_encoder, cache,
                                           'full_valid', frozen_source),
                             frozen['stats'])
    del raw_valid
    gc.collect()
    if text_raw['ids'] != frozen_valid['ids'] or not np.array_equal(
            text_raw['b'], frozen_valid['b']) or not np.array_equal(
            text_raw['observed'], frozen_valid['mask'][:, :, 0]):
        raise ValueError('Text and Fusion sample order or observed masks differ')
    quick = scenarios(frozen_valid, frozen_encoder, cache, frozen['stats'],
                      frozen_source)
    scattered = make_scattered_suite(frozen_valid, frozen_encoder, cache,
                                     frozen['stats'], frozen_source)
    rows = []
    weight_rows = []
    for scenario in quick + scattered:
        output, weights = evaluate_scenario(text_model, fusion_model, text_raw,
                                            frozen_valid, scenario, args.device)
        spec = {key: scenario[key] for key in ('subset', 'rate', 'position', 'replicate')}
        for name, (probability, intensity) in output.items():
            rows.append({'model': name, **spec,
                         **metrics(text_raw['y'], text_raw['r'],
                                   probability, intensity)})
        weight_rows.append({**spec, **weights})
        print('SCENARIO', json.dumps(spec, ensure_ascii=False), flush=True)
    write_csv(args.out / '问题2_固定文本融合完整验证场景.csv', rows)
    write_csv(args.out / '问题2_固定文本融合权重与可观测比例.csv', weight_rows)
    scores = {}
    clean_f1 = {}
    sparse_f1 = {}
    for name in ('text_only', 'fusion_only', 'fixed_50_20', 'zero_aware_fixed'):
        subset = [row for row in rows if row['model'] == name]
        scores[name] = validation_score(subset)
        clean_f1[name] = next(r['Macro_F1'] for r in subset if r['subset'] == 'none')
        sparse_f1[name] = float(np.mean([r['Macro_F1'] for r in subset
                                        if r['position'] == 'scattered']))
    report = {'validation_scores': scores,
              'clean_Macro_F1': clean_f1, 'scattered_TAV_mean_Macro_F1': sparse_f1,
              'fixed_rule': 'T observed fraction >=0.5: text/fusion=0.5/0.5; otherwise=0.2/0.8',
              'zero_aware_rule': 'same fixed rule, except 0 observed text gives text weight 0',
              'rules_fixed_before_label_evaluation': True,
              'source_sha256': source_hash, 'safe_manifest_sha256': manifest_hash,
              'safety_audit_sha256': safety_hash,
              'text_delta_sha256': sha(args.text_delta),
              'fusion_checkpoint_sha256': sha(args.fusion_checkpoint),
              'text_missing_before_both_bert_encoders': True,
              'full_test_or_annex3_labels_read': False}
    dump(args.out / '问题2_固定文本融合验证结论.json', report)
    print('FINISHED', json.dumps(report, ensure_ascii=False), flush=True)


def smoke():
    """CPU-only checks for weights, masking, score, and model output shape."""
    runtime('cpu')
    config = json.loads(TEXT_CONFIG.read_text(encoding='utf-8'))
    model = SupervisedTextEncoder(config).cpu().eval()
    b = np.zeros((2, 3, 50), np.int64)
    b[:, 0, :6] = [101, 2023, 100, 2024, 2025, 102]
    b[:, 1, :6] = 1
    raw = {'b': b, 'valid': np.zeros((2, 50), bool),
           'observed': np.zeros((2, 50), bool), 'ids': ['a', 'b']}
    raw['valid'][:, 1:5] = True
    raw['observed'][:, 1:5] = True
    raw['observed'][1, 2] = False
    clean = predict_text(model, raw, raw['observed'], 'cpu', batch_size=2)
    changed = b.copy()
    changed[1, 0, 2] = 2047
    altered = predict_text(model, {**raw, 'b': changed}, raw['observed'],
                           'cpu', batch_size=2)
    if not np.array_equal(clean[0][1], altered[0][1]):
        raise AssertionError('Missing lexical ID changed text classifier output')
    fake_text = (np.array([[.2, .3, .5], [.3, .3, .4]], np.float32),
                 np.array([.5, .2], np.float32))
    fake_fusion = (np.array([[.4, .4, .2], [.4, .5, .1]], np.float32),
                   np.array([-.2, .3], np.float32))
    mixed, weight = mixture(fake_text, fake_fusion, np.array([.6, .2]))
    if not np.allclose(weight, [.5, .2]) or not np.allclose(
            mixed[0][0], .5 * fake_text[0][0] + .5 * fake_fusion[0][0]):
        raise AssertionError('Fixed mixture weight calculation is wrong')
    zero, zero_weight = mixture(fake_text, fake_fusion, np.array([.6, 0.]),
                                zero_aware=True)
    if zero_weight[1] != 0 or not np.array_equal(zero[0][1], fake_fusion[0][1]):
        raise AssertionError('Zero-observation guard is wrong')
    print('SMOKE_OK', json.dumps({'masked_id_invariance': True,
                                  'fixed_weights': [float(x) for x in weight],
                                  'zero_observation_text_weight': 0.,
                                  'probability_shape': list(clean[0].shape)},
                                 ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('evaluate', 'smoke'), required=True)
    parser.add_argument('--safe-source', type=Path, default=FULL_SOURCE)
    parser.add_argument('--safe-manifest', type=Path, default=SAFE_MANIFEST)
    parser.add_argument('--safety-audit', type=Path, default=SAFETY_AUDIT)
    parser.add_argument('--text-delta', type=Path,
                        default=TEXT_OUT / '问题2_完整数据BERT文本编码器_最优delta.pt')
    parser.add_argument('--fusion-checkpoint', type=Path, default=FULL_STUDENT)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    if args.mode == 'smoke':
        smoke()
    else:
        run(args)


if __name__ == '__main__':
    main()
