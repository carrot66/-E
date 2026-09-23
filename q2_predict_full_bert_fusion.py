"""附件3的监督 BERT + Fusion 专用无标签推理入口。

Only run after the full-valid comparison has accepted this candidate. This
program opens the safe train/valid pickle only to hash it; it never deserializes
full MOSEI test or reads any labels for the 30 anonymous attachment-3 inputs.
"""
from __future__ import annotations

import argparse
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import torch

from q2_ensemble import load_models
from q2_experiment import (ROOT, Fusion, derive_masks, dump, normalize, predict,
                           prepare_split, runtime, sha, write_csv)
from q2_full_bert_fusion_experiment import (DEFAULT_OUT as FUSION_OUT,
                                             FineTunedTextAdapter)
from q2_full_bert_text_experiment import (FULL_SOURCE, SAFE_MANIFEST,
                                           SAFETY_AUDIT, read_safe_manifest,
                                           read_safety_audit)

CLASS_NAMES = ('Negative', 'Neutral', 'Positive')
ANNEX3 = ROOT / 'E题数据/E题数据/附件3-模态缺失特征样本/对齐版本'


def checked_stats(stats):
    expected = {'t':768, 'a':74, 'v':35}
    if set(stats) != set(expected):
        raise ValueError('Fusion checkpoint normalization fields differ')
    for modality, dimension in expected.items():
        mean = np.asarray(stats[modality]['mean'], dtype=np.float32)
        std = np.asarray(stats[modality]['std'], dtype=np.float32)
        if (mean.shape != (dimension,) or std.shape != (dimension,) or
                not np.isfinite(mean).all() or not np.isfinite(std).all() or
                (std <= 0).any()):
            raise ValueError(f'Invalid {modality} normalization statistics')


def verified_model(args):
    model_dir = Path(args.model_dir)
    conclusion_path = model_dir / '完整数据_监督BERT融合验证结论.json'
    conclusion = json.loads(conclusion_path.read_text(encoding='utf-8'))
    if conclusion.get('accepted_vs_frozen') is not True:
        raise ValueError('监督BERT融合模型未通过预声明验证门槛，禁止生成正式附件3预测')
    selected = conclusion.get('new_selected')
    if selected not in ('teacher','student'):
        raise ValueError(f'Invalid selected Fusion branch: {selected!r}')
    fusion_path = model_dir / conclusion[f'{selected}_checkpoint']
    fusion_hash = sha(fusion_path)
    if fusion_hash != conclusion[f'{selected}_sha256']:
        raise ValueError('Selected Fusion checkpoint SHA-256 differs from conclusion')
    delta_path = Path(args.text_delta)
    delta_hash = sha(delta_path)
    if delta_hash != conclusion['text_delta_sha256']:
        raise ValueError('Supervised BERT delta SHA-256 differs from conclusion')
    safety_hash = read_safety_audit(Path(args.safety_audit))
    safe_source = Path(args.safe_source)
    safe_hash = sha(safe_source)
    manifest_hash, manifest = read_safe_manifest(Path(args.safe_manifest),
        safe_source, safe_hash, safety_hash)
    adapter = FineTunedTextAdapter(delta_path, args.device)
    text_provenance = adapter.checkpoint['provenance']
    if (text_provenance['full_source_sha256'] != safe_hash or
            text_provenance['safe_train_valid_manifest_sha256'] != manifest_hash or
            text_provenance['independent_safety_audit_sha256'] != safety_hash or
            text_provenance['original_full_source_sha256'] != manifest['source_sha256']):
        raise ValueError('BERT delta provenance does not match safe source and audit')
    models, checkpoints = load_models({'selected':fusion_path}, args.device)
    fusion = models['selected']
    fusion_checkpoint = checkpoints['selected']
    provenance = fusion_checkpoint['provenance']
    if ((selected == 'teacher' and fusion_checkpoint['method'] != 'baseline') or
            (selected == 'student' and fusion_checkpoint['method'] != 'distilled')):
        raise ValueError('Selected checkpoint method differs from model conclusion')
    if (provenance['safe_source_sha256'] != safe_hash or
            provenance['safe_manifest_sha256'] != manifest_hash or
            provenance['original_full_source_sha256'] != manifest['source_sha256'] or
            provenance['safety_audit_sha256'] != safety_hash or
            provenance['text_delta_sha256'] != delta_hash or
            provenance['text_delta_revision'] != adapter.revision):
        raise ValueError('Fusion checkpoint provenance does not match BERT delta or safe data')
    checked_stats(fusion_checkpoint['stats'])
    return {'selected': selected, 'model': fusion, 'adapter': adapter,
            'stats': fusion_checkpoint['stats'], 'fusion_sha256': fusion_hash,
            'delta_sha256': delta_hash, 'safe_source_sha256':safe_hash,
            'safe_manifest_sha256':manifest_hash, 'safety_audit_sha256':safety_hash,
            'conclusion_sha256':sha(conclusion_path),
            'original_full_source_sha256':manifest['source_sha256']}


def read_anonymous(path):
    with path.open('rb') as stream:
        container = pickle.load(stream)
    if set(container) != {'test'} or set(container['test']) != {'text_bert','audio','vision'}:
        raise ValueError(f'{path.name}: anonymous file contains unexpected fields')
    source = container['test']
    for key, trailing in (('text_bert',(3,50)),('audio',(50,74)),('vision',(50,35))):
        if np.asarray(source[key]).shape != (1,*trailing):
            raise ValueError(f'{path.name}: {key} shape mismatch')
    if not np.isfinite(np.asarray(source['audio'])).all() or not np.isfinite(np.asarray(source['vision'])).all():
        raise ValueError(f'{path.name}: nonfinite A/V feature')
    source = {key: np.asarray(value) for key,value in source.items()}
    source['id'] = [path.stem]
    return source


def predict_one(path, verified, device, cache):
    source = read_anonymous(path)
    bert, valid, masks = derive_masks(source)
    valid_count = int(valid.sum())
    if valid_count < 1:
        raise ValueError(f'{path.name}: no content positions')
    unknown = valid & (bert[:,0] == 100)
    a_seen = np.any(source['audio'] != 0, axis=-1)
    v_seen = np.any(source['vision'] != 0, axis=-1)
    joint_unknown = unknown & ~a_seen & ~v_seen
    if np.any(masks[:,:,0][joint_unknown]):
        raise AssertionError('Shared-gap [UNK] is wrongly treated as observed text')
    adapter = verified['adapter']
    # Re-encode the original anonymous item with the missing lexical IDs
    # removed before BERT, rather than applying a mask to clean BERT states.
    repeated_a = adapter.encode(bert, masks[:,:,0], batch=1)
    repeated_b = adapter.encode(bert, masks[:,:,0], batch=1)
    if not np.array_equal(repeated_a, repeated_b):
        raise AssertionError(f'{path.name}: BERT token inference is not repeatable')
    prepared = prepare_split(source, adapter, cache, 'fine_annex3', sha(path))
    if not np.array_equal(repeated_a, prepared['t']):
        raise AssertionError(f'{path.name}: cached BERT tokens differ from direct re-encoding')
    normalize(prepared, verified['stats'])
    probability_a, intensity_a = predict(verified['model'], prepared, device)
    probability_b, intensity_b = predict(verified['model'], prepared, device)
    if not np.array_equal(probability_a, probability_b) or not np.array_equal(intensity_a, intensity_b):
        raise AssertionError(f'{path.name}: Fusion inference is not repeatable')
    p = probability_a[0]
    if not np.isfinite(p).all() or not np.isclose(p.sum(), 1., atol=1e-5):
        raise AssertionError(f'{path.name}: invalid class probabilities')
    cls = int(p.argmax())
    intensity = float(intensity_a[0])
    if cls == 1:
        intensity = 0.
    elif cls == 0:
        intensity = min(intensity, -.01)
    else:
        intensity = max(intensity, .01)
    intensity = float(np.clip(intensity, -3., 3.))
    row = {'样本编号':path.stem, '预测极性':CLASS_NAMES[cls], '预测强度':intensity,
           '预测最大概率':float(p.max()), '负向概率':float(p[0]),
           '中性概率':float(p[1]), '正向概率':float(p[2]),
           '文本不可用比例':float(1-masks[:,:,0].sum()/valid_count),
           '语音不可用比例':float(1-masks[:,:,1].sum()/valid_count),
           '视觉不可用比例':float(1-masks[:,:,2].sum()/valid_count),
           '共同缺失UNK位置数':int(joint_unknown.sum())}
    return row, {'file':path.name,'sha256':sha(path),'joint_unknown':int(joint_unknown.sum()),
                 'content_positions':valid_count}


def run(args):
    runtime(args.device)
    verified = verified_model(args)
    directory = Path(args.annex3_dir)
    files = sorted(directory.glob('附件3_*.pkl'))
    if len(files) != 30 or len({path.stem for path in files}) != 30:
        raise ValueError('附件3对齐版须有30个唯一无标签样本')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = ROOT / 'work/问题2_监督BERT融合缓存' / verified['delta_sha256'][:20]
    rows, inputs = [], []
    for path in files:
        row, provenance = predict_one(path, verified, args.device, cache)
        rows.append(row)
        inputs.append(provenance)
    if len(rows) != 30 or len({row['样本编号'] for row in rows}) != 30:
        raise AssertionError('附件3预测记录不完整或编号重复')
    csv_path = out / '问题2_附件3全量预测.csv'
    write_csv(csv_path, rows)
    record = {'model':'监督BERT+Fusion', 'selected_branch':verified['selected'],
              'count':len(rows), 'class_counts':{name:sum(r['预测极性']==name for r in rows)
                                                 for name in CLASS_NAMES},
              'output_csv_sha256':sha(csv_path),
              'fusion_checkpoint_sha256':verified['fusion_sha256'],
              'text_delta_sha256':verified['delta_sha256'],
              'safe_train_valid_sha256':verified['safe_source_sha256'],
              'safe_manifest_sha256':verified['safe_manifest_sha256'],
              'safety_audit_sha256':verified['safety_audit_sha256'],
              'selection_conclusion_sha256':verified['conclusion_sha256'],
              'source_full_sha256':verified['original_full_source_sha256'],
              'code_sha256':sha(__file__),
              'all_BERT_and_Fusion_repeated_predictions_identical':True,
              'total_joint_unknown_positions':sum(x['joint_unknown'] for x in inputs),
              'input_files':inputs,
              'policy':'Only anonymous attachment-3 features were inferred; no full-test labels or IDs were loaded.'}
    dump(out / '问题2_附件3推理哈希与核验.json', record)
    print(json.dumps({'output':str(csv_path),'count':len(rows),
                      'class_counts':record['class_counts'],
                      'csv_sha256':record['output_csv_sha256']},ensure_ascii=False),flush=True)


def smoke():
    """Synthetic CPU probe of natural/artificial UNK and Fusion output."""
    runtime('cpu')
    from q2_full_bert_text_experiment import (CONFIG_PATH, SupervisedTextEncoder,
                                               save_delta)
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    encoder = SupervisedTextEncoder(config).cpu().eval()
    with tempfile.TemporaryDirectory(prefix='q2_ft_annex3_smoke_') as directory:
        directory = Path(directory)
        delta = directory / 'synthetic_delta.pt'
        save_delta(encoder, delta, config, {'synthetic':True}, 0, 0.)
        adapter = FineTunedTextAdapter(delta, 'cpu')
        source = {'text_bert':np.zeros((1,3,50),np.int64),
                  'audio':np.zeros((1,50,74),np.float32),
                  'vision':np.zeros((1,50,35),np.float32)}
        source['text_bert'][0,0,:6] = [101,2023,100,2024,100,102]
        source['text_bert'][0,1,:6] = 1
        source['audio'][0,1:5] = 1
        source['vision'][0,1:5] = 1
        source['audio'][0,4] = 0
        source['vision'][0,4] = 0
        file = directory / '附件3_01.pkl'
        with file.open('wb') as stream:
            pickle.dump({'test':source},stream)
        item = read_anonymous(file)
        _bert, _valid, masks = derive_masks(item)
        if not masks[0,2,0] or masks[0,4,0]:
            raise AssertionError('Natural [UNK] or joint-gap [UNK] mask is wrong')
        fusion = Fusion(gated=True).cpu().eval()
        stats = {key:{'mean':[0.]*dimension,'std':[1.]*dimension}
                 for key,dimension in (('t',768),('a',74),('v',35))}
        verified = {'adapter':adapter,'model':fusion,'stats':stats}
        row, provenance = predict_one(file,verified,'cpu',directory/'cache')
        if row['共同缺失UNK位置数'] != 1 or row['预测极性'] not in CLASS_NAMES:
            raise AssertionError('Synthetic anonymous inference failed')
        print('SMOKE_OK',json.dumps({'natural_unk_preserved':True,
              'joint_gap_redacted':True,'BERT_and_Fusion_replay_exact':True,
              'prediction_fields':list(row)},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('predict','smoke'),default='predict')
    parser.add_argument('--model-dir',type=Path,default=FUSION_OUT)
    parser.add_argument('--text-delta',type=Path,default=ROOT/'outputs/问题2_完整数据BERT文本编码实验/问题2_完整数据BERT文本编码器_最优delta.pt')
    parser.add_argument('--safe-source',type=Path,default=FULL_SOURCE)
    parser.add_argument('--safe-manifest',type=Path,default=SAFE_MANIFEST)
    parser.add_argument('--safety-audit',type=Path,default=SAFETY_AUDIT)
    parser.add_argument('--annex3-dir',type=Path,default=ANNEX3)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/问题2_监督BERT附件3预测')
    parser.add_argument('--device',default='cuda:1')
    args = parser.parse_args()
    if args.mode == 'smoke':
        smoke()
    else:
        run(args)
