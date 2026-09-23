"""附件3最终候选：监督文本与完整数据鲁棒 Fusion 的固定无标签融合。

固定规则在完整 valid 标签评估前写入实验配置：文本可用比例不少于
0.5 时取文本/三模态预测各50%，否则取20%/80%。本程序只读取附件3
特征和已冻结的权重，不读取完整 MOSEI test 或任何标签。
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from q2_ensemble import load_models
from q2_experiment import (ROOT, TextEncoder, dump, normalize, predict,
                           prepare_split, runtime, save_predictions, sha,
                           write_csv)
from q2_full_bert_text_experiment import (FULL_SOURCE, SAFETY_AUDIT,
                                           SAFE_MANIFEST, load_delta,
                                           prepare_text, read_safe_manifest,
                                           read_safety_audit, tensors)


def coherence(probability, intensity):
    probability = np.asarray(probability)
    intensity = np.asarray(intensity).copy()
    cls = probability.argmax(1)
    intensity[cls == 1] = 0.
    intensity[cls == 0] = np.minimum(intensity[cls == 0], -.01)
    intensity[cls == 2] = np.maximum(intensity[cls == 2], .01)
    return probability, intensity


@torch.inference_mode()
def predict_text(model, raw, observed, device):
    probs, regs = [], []
    model.eval()
    for start in range(0, len(raw['ids']), 64):
        idx = np.arange(start, min(start + 64, len(raw['ids'])))
        logits, reg = model(*tensors(raw, idx, observed[idx], device))
        probs.append(logits.float().softmax(-1).cpu().numpy())
        regs.append(reg.float().cpu().numpy())
    return coherence(np.concatenate(probs), np.concatenate(regs))


def fixed_mix(text_pred, fusion_pred, text_ratio):
    weight = np.where(text_ratio >= .5, .5, .2).astype(np.float32)
    prob = weight[:, None] * text_pred[0] + (1. - weight[:, None]) * fusion_pred[0]
    reg = weight * text_pred[1] + (1. - weight) * fusion_pred[1]
    return coherence(prob, reg), weight


def run(args):
    runtime(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    safety_hash = read_safety_audit(Path(args.safety_audit))
    safe_hash = sha(Path(args.safe_source))
    manifest_hash, manifest = read_safe_manifest(Path(args.safe_manifest),
                                                  Path(args.safe_source),
                                                  safe_hash, safety_hash)
    text_delta = Path(args.text_delta)
    text_model, text_ckpt = load_delta(text_delta, args.device)
    text_prov = text_ckpt['provenance']
    if (text_prov['full_source_sha256'] != safe_hash or
            text_prov['safe_train_valid_manifest_sha256'] != manifest_hash or
            text_prov['independent_safety_audit_sha256'] != safety_hash):
        raise ValueError('监督BERT delta与安全训练源不一致')
    fusion_path = Path(args.fusion_checkpoint)
    models, checkpoints = load_models({'fusion': fusion_path}, args.device)
    fusion = models['fusion']; fusion_ckpt = checkpoints['fusion']
    if fusion_ckpt['method'] != 'distilled':
        raise ValueError('需要完整数据缺失蒸馏学生模型')
    fusion_source = fusion_ckpt['provenance']['source_sha256']
    if fusion_source not in (safe_hash, manifest['source_sha256']):
        raise ValueError('Fusion权重与安全训练源不一致')
    frozen_encoder = TextEncoder(args.device,
                                 revision=fusion_ckpt['provenance']['bert_model'])
    files = sorted(Path(args.annex3_dir).glob('*.pkl'))
    if len(files) != 30:
        raise ValueError(f'附件3文件数应为30，实际{len(files)}')
    cache = ROOT / 'work/问题2_特征缓存'
    rows = []
    hidden_unk = 0
    for path in files:
        with path.open('rb') as stream:
            container = pickle.load(stream)
        if set(container) != {'test'}:
            raise ValueError(f'{path.name}外层结构异常')
        sample = container['test']
        if set(sample) != {'text_bert', 'audio', 'vision'}:
            raise ValueError(f'{path.name}字段异常')
        sample['id'] = [path.stem]
        raw_text = prepare_text(sample, 'annex3')
        frozen_item = normalize(prepare_split(sample, frozen_encoder, cache,
                                                'annex3_fixed_frozen', sha(path)),
                                fusion_ckpt['stats'])
        observed = frozen_item['mask'][:, :, 0]
        if not np.array_equal(raw_text['observed'], observed):
            raise AssertionError(f'{path.name}文本共同缺失判据不一致')
        hidden_unk += int((frozen_item['valid'] & (frozen_item['b'][:, 0] == 100) &
                           ~observed).sum())
        text_pred = predict_text(text_model, raw_text, observed, args.device)
        fusion_pred = coherence(*predict(fusion, frozen_item, args.device))
        mixed, weight = fixed_mix(text_pred, fusion_pred,
                                  observed.sum(1) /
                                  frozen_item['valid'].sum(1).clip(min=1))
        mixed2, weight2 = fixed_mix(text_pred, fusion_pred,
                                    observed.sum(1) /
                                    frozen_item['valid'].sum(1).clip(min=1))
        if not np.array_equal(mixed[0], mixed2[0]) or not np.array_equal(mixed[1], mixed2[1]):
            raise AssertionError(f'{path.name}重复推理不一致')
        if not np.allclose(mixed[0].sum(1), 1., atol=1e-6):
            raise AssertionError(f'{path.name}概率未归一')
        row = save_predictions(out / '逐样本预测', path.stem + '.csv',
                               frozen_item, mixed[0], mixed[1])[0]
        row['模型'] = '固定文本-三模态融合'
        row['文本融合权重'] = float(weight[0])
        rows.append(row)
    if len(rows) != 30 or len({row['样本编号'] for row in rows}) != 30:
        raise AssertionError('附件3结果数量/编号检查失败')
    if hidden_unk != 131:
        raise AssertionError(f'附件3共同缺失UNK数={hidden_unk}，预期131')
    write_csv(out / '问题2_附件3全量预测.csv', rows)
    dump(out / '问题2_固定文本融合附件3推理核验.json', {
        'count': 30, 'rule': 'text ratio >=0.5: 0.5; otherwise: 0.2',
        'full_test_labels_used': False, 'masked_joint_unknown_positions': hidden_unk,
        'safe_source_sha256': safe_hash, 'safe_manifest_sha256': manifest_hash,
        'safety_audit_sha256': safety_hash, 'text_delta_sha256': sha(text_delta),
        'fusion_checkpoint_sha256': sha(fusion_path), 'repeat_inference_identical': True,
        'probabilities_sum_to_one': True,
    })
    print(f'FINISHED {len(rows)} samples, joint UNK masked={hidden_unk}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--safe-source', default=FULL_SOURCE)
    parser.add_argument('--safe-manifest', default=SAFE_MANIFEST)
    parser.add_argument('--safety-audit', default=SAFETY_AUDIT)
    parser.add_argument('--text-delta', default=ROOT / 'outputs/问题2_完整数据BERT文本编码实验/问题2_完整数据BERT文本编码器_最优delta.pt')
    parser.add_argument('--fusion-checkpoint', default=ROOT / 'outputs/问题2_完整MOSEI训练实验/完整数据_缺失蒸馏学生_种子2026.pt')
    parser.add_argument('--annex3-dir', default=ROOT / 'E题数据/E题数据/附件3-模态缺失特征样本/对齐版本')
    parser.add_argument('--out', default=ROOT / 'outputs/问题2_固定文本融合附件3结果')
    parser.add_argument('--device', default='cuda:1')
    run(parser.parse_args())
