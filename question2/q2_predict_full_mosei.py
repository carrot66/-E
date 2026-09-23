"""问题2：经验证选定的完整数据模型对附件3逐条无标签推理。

只读取附件3的30个特征文件；完整数据test及标签不进入本进程。
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from q2_ensemble import load_models, predict_ensemble
from q2_experiment import (
    ROOT, TextEncoder, dump, normalize, prepare_split, runtime, save_predictions,
    sha, write_csv,
)


def run(args):
    runtime(args.device)
    result_dir = Path(args.model_dir)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    conclusion_path = result_dir / '完整数据_验证选模结论.json'
    conclusion = json.loads(conclusion_path.read_text(encoding='utf-8'))
    selected = conclusion['selected']
    if selected not in ('teacher', 'student'):
        raise ValueError(f'未知模型选择: {selected}')
    ckpt_path = result_dir / conclusion[f'{selected}_checkpoint']
    if sha(ckpt_path) != conclusion[f'{selected}_sha256']:
        raise ValueError('模型权重 SHA-256 与验证选模记录不一致')
    models, checkpoints = load_models({selected: ckpt_path}, args.device)
    checkpoint = checkpoints[selected]
    provenance = checkpoint['provenance']
    manifest_path = Path(args.safe_manifest)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if provenance['source_sha256'] != manifest['source_sha256']:
        raise ValueError('模型训练来源与已审计完整数据不一致')
    safety_path = Path(args.safety_audit)
    safety = json.loads(safety_path.read_text(encoding='utf-8'))
    if safety.get('annex3_unique_matches') != 30 or safety.get('annex3_match_split_counts') != {'test': 30}:
        raise ValueError('独立附件3重合审计未通过')
    encoder = TextEncoder(args.device, revision=provenance['bert_model'])
    files = sorted(Path(args.annex3_dir).glob('*.pkl'))
    if len(files) != 30:
        raise ValueError(f'附件3文件数应为30，实际{len(files)}')
    cache = ROOT / 'work/问题2_特征缓存'
    rows = []
    unk_hidden = 0
    for file in files:
        with file.open('rb') as stream:
            container = pickle.load(stream)
        if set(container) != {'test'} or set(container['test']) != {'text_bert', 'audio', 'vision'}:
            raise ValueError(f'{file.name}输入结构异常')
        sample = container['test']
        if len(sample['audio']) != 1:
            raise ValueError(f'{file.name}预期恰有1条样本')
        sample['id'] = [file.stem]
        prepared = normalize(
            prepare_split(sample, encoder, cache, 'annex3_full_model', sha(file)),
            checkpoint['stats'])
        b = prepared['b']
        missing_unk = prepared['valid'] & (b[:, 0] == 100) & ~prepared['mask'][:, :, 0]
        unk_hidden += int(missing_unk.sum())
        prob1, reg1 = predict_ensemble(models, [selected], prepared, args.device, None)
        prob2, reg2 = predict_ensemble(models, [selected], prepared, args.device, None)
        if not np.array_equal(prob1, prob2) or not np.array_equal(reg1, reg2):
            raise AssertionError(f'{file.name}重复推理不一致')
        if not np.isfinite(prob1).all() or not np.isfinite(reg1).all():
            raise AssertionError(f'{file.name}存在非有限预测')
        if not np.allclose(prob1.sum(1), 1., atol=1e-6):
            raise AssertionError(f'{file.name}类别概率不归一')
        item = save_predictions(output / '逐样本预测', file.stem + '.csv',
                                prepared, prob1, reg1)[0]
        item['模型'] = '完整数据缺失蒸馏学生' if selected == 'student' else '完整数据干净教师'
        rows.append(item)
    if len({x['样本编号'] for x in rows}) != 30 or unk_hidden != 131:
        raise AssertionError(f'附件3数量或共同缺失数异常: rows={len(rows)} UNK={unk_hidden}')
    write_csv(output / '问题2_附件3全量预测.csv', rows)
    dump(output / '问题2_完整数据模型推理核验.json', {
        'count': len(rows), 'selected': selected,
        'checkpoint_sha256': sha(ckpt_path),
        'conclusion_sha256': sha(conclusion_path),
        'safe_manifest_sha256': sha(manifest_path),
        'safety_audit_sha256': sha(safety_path),
        'bert_revision': encoder.revision,
        'masked_joint_unknown_positions': unk_hidden,
        'probabilities_sum_to_one': True, 'repeat_inference_identical': True,
        'full_test_labels_used': False,
    })
    print(f'FINISHED {len(rows)} samples, joint UNK masked={unk_hidden}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default=ROOT / 'outputs/question2/问题2_完整MOSEI训练实验')
    parser.add_argument('--annex3-dir', default=ROOT / 'E题数据/E题数据/附件3-模态缺失特征样本/对齐版本')
    parser.add_argument('--safe-manifest', default=ROOT / 'work/full_mosei/safe_train_valid.manifest.json')
    parser.add_argument('--safety-audit', default=ROOT / 'outputs/question2/问题2_完整版MOSEI安全整合审计/问题2_完整版MOSEI与竞赛数据安全整合审计.json')
    parser.add_argument('--out', default=ROOT / 'outputs/question2/问题2_完整数据附件3推理结果')
    parser.add_argument('--device', default='cuda:1')
    run(parser.parse_args())
