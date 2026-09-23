"""Optional full-data Fusion with supervised fine-tuned BERT token states.

This mirrors q2_full_mosei_experiment's A/V masks, train-only normalization,
architecture, distillation, and validation score. Every text-gap bank/scenario
is encoded anew after lexical redaction, never clean BERT states times a mask.
The source pickle must contain ONLY train/valid; no complete test is loaded.
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from q2_ensemble import load_models
from q2_experiment import (ROOT, Fusion, block_mask, dump, fit_normalization,
                           norm_text, normalize, predict, prepare_split, runtime, scenarios,
                           seed_all, sha, write_csv)
from q2_full_bert_text_experiment import (CONFIG_PATH as TEXT_CONFIG,
                                           DEFAULT_OUT as TEXT_OUT,
                                           FULL_SOURCE, SAFE_MANIFEST,
                                           SAFETY_AUDIT,
                                           SupervisedTextEncoder,
                                           load_delta, read_safe_manifest,
                                           read_safety_audit, save_delta)
from q2_full_mosei_experiment import (audit_splits, class_weights,
                                      hide_joint_unknown, target_validation_rows,
                                      train_student, train_teacher)
from q2_scattered_gap_experiment import (get_banks, make_scattered_suite,
                                         profile)

CONFIG_PATH = ROOT / 'q2_full_bert_fusion_config.json'
DEFAULT_OUT = ROOT / 'outputs/问题2_完整数据监督BERT融合实验'
FROZEN_OUT = ROOT / 'outputs/问题2_完整MOSEI训练实验'
LABEL_CSV = (ROOT / 'work/full_mosei/CMU-MOSEI_完整版_对齐特征与标签/'
             'label.csv')


class FineTunedTextAdapter:
    """Exact interface needed by q2_experiment.text_features/scenarios."""
    def __init__(self, delta_path: Path, device: str):
        self.model, self.checkpoint = load_delta(delta_path, device)
        self.device = device
        self.delta_sha256 = sha(delta_path)
        self.base_revision = self.checkpoint['config']['model_revision']
        # text_features uses revision in its cache key. The delta hash and
        # preprocessing version prevent collisions with frozen BERT caches.
        self.revision = f'{self.base_revision}-fine-{self.delta_sha256}-prebertmask-v1'

    @torch.inference_mode()
    def encode(self, bert_input: np.ndarray, observed: np.ndarray, batch=64):
        if bert_input.shape[1:] != (3, 50) or observed.shape != bert_input.shape[:1] + (50,):
            raise ValueError('Expected [N,3,50] token triplets and [N,50] mask')
        content = np.zeros_like(observed, bool)
        for i, row in enumerate(bert_input):
            cls = np.flatnonzero(row[0] == 101)
            sep = np.flatnonzero(row[0] == 102)
            if not len(cls) or not len(sep):
                raise ValueError(f'BERT sample {i} lacks CLS/SEP')
            content[i, cls[0] + 1:sep[-1]] = True
        if np.any(observed & ~content):
            raise ValueError('CLS/SEP/PAD cannot be marked observed text content')
        result = []
        self.model.eval()
        for start in range(0, len(bert_input), batch):
            end = min(start + batch, len(bert_input))
            tokens = torch.as_tensor(bert_input[start:end].copy(),
                                     dtype=torch.long, device=self.device)
            mask = torch.as_tensor(observed[start:end], dtype=torch.bool,
                                   device=self.device)
            valid = torch.as_tensor(content[start:end], dtype=torch.bool,
                                    device=self.device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                enabled=self.device.startswith('cuda')):
                states = self.model.encode_tokens(tokens, mask, valid)
            result.append(states.float().cpu().numpy().astype(np.float16))
        encoded = np.concatenate(result)
        if np.any(encoded[~observed] != 0):
            raise AssertionError('Fine-tuned BERT emitted nonzero missing-text vectors')
        return encoded


def check_prebert_redaction(adapter, split):
    """Mutated missing lexical IDs must not affect retained token states."""
    eligible = np.flatnonzero((split['mask'][:, :, 0].sum(1) >= 3) &
                             (split['mask'][:, :, 0].sum(1) == split['valid'].sum(1)))
    if not len(eligible):
        raise AssertionError('No complete-text sample available for redaction probe')
    sample = int(eligible[0])
    b = split['b'][sample:sample + 1].copy()
    observed = split['mask'][sample:sample + 1, :, 0].copy()
    hidden = block_mask(split['valid'][sample:sample + 1], .3, 52283)
    observed &= ~hidden
    if not hidden.any() or not observed.any():
        raise AssertionError('Selected pre-BERT redaction probe is degenerate')
    changed = b.copy()
    changed[:, 0][hidden] = 2023
    original_state = adapter.encode(b, observed, batch=1)
    changed_state = adapter.encode(changed, observed, batch=1)
    if not np.array_equal(original_state, changed_state):
        raise AssertionError('Hidden lexical ID leaked through BERT context')
    clean = adapter.encode(b, split['mask'][sample:sample + 1, :, 0], batch=1)
    if np.array_equal(original_state[observed], clean[observed]):
        raise AssertionError('Text gap appears to reuse clean contextual states')
    return {'missing_id_mutation_invariant': True,
            'masked_context_reencoded': True,
            'hidden_token_states_zero': True}


def load_safe_data(source, data_root, label_csv, safety_audit, out):
    with source.open('rb') as stream:
        data = pickle.load(stream)
    if set(data) != {'train', 'valid'}:
        raise ValueError('Fusion experiment requires train/valid-only safe pickle')
    audit, target = audit_splits(data, data_root, label_csv, safety_audit)
    train_raw, valid_raw = data['train'], data['valid']
    del data
    gc.collect()
    audit['joint_zero_unk_masked_train'] = hide_joint_unknown(train_raw)
    audit['joint_zero_unk_masked_valid'] = hide_joint_unknown(valid_raw)
    for split in (train_raw, valid_raw):
        split.pop('text', None)
    dump(out / '完整数据_监督BERT融合安全审计.json', audit)
    return train_raw, valid_raw, target, audit


def compare_frozen_reference(path, source_hash, safety_hash, av_stats, label_csv):
    conclusion = json.loads((path / '完整数据_验证选模结论.json').read_text(encoding='utf-8'))
    old_config = json.loads((path / '完整数据_实验预注册.json').read_text(encoding='utf-8'))
    if old_config['source_sha256'] == source_hash:
        # The frozen run may use the same safe source; the full-source version
        # differs by container metadata, while train/valid IDs are audited.
        same_source = True
    else:
        same_source = False
    if old_config.get('label_csv_sha256') != sha(label_csv):
        raise ValueError('Frozen Fusion comparison used a different label CSV')
    teacher_checkpoint = path / conclusion['teacher_checkpoint']
    frozen = torch.load(teacher_checkpoint, map_location='cpu', weights_only=False)
    for modality in ('a', 'v'):
        if frozen['stats'][modality] != av_stats[modality]:
            raise ValueError(f'A/V train normalization differs for {modality}')
    old_selected = conclusion['selected']
    old_profile = conclusion[f'{old_selected}_full_valid']
    return {'frozen_selected': old_selected, 'frozen_profile': old_profile,
            'frozen_source_sha256': old_config['source_sha256'],
            'same_source_sha256': same_source,
            'frozen_teacher_sha256': sha(teacher_checkpoint),
            'safety_audit_sha256': safety_hash}


def checkpoint_replay(paths, split, scenario, device):
    loaded, _ = load_models(paths, device)
    for name, model in loaded.items():
        first = predict(model, split, device, scenario)
        second = predict(model, split, device, scenario)
        if not all(np.array_equal(a, b) for a, b in zip(first, second)):
            raise AssertionError(f'{name} Fusion checkpoint inference is not repeatable')
    return True


def text_checkpoint_replay(delta_path, split, quick, stats, device):
    """Reload BERT delta and reproduce clean and gapped Fusion inputs."""
    adapter = FineTunedTextAdapter(delta_path, device)
    # Match the original cache batch size so mixed-precision GEMM kernels use
    # the same shapes during replay.
    indices = slice(0, min(64, len(split['ids'])))
    clean_mask = split['mask'][indices, :, 0]
    clean = adapter.encode(split['b'][indices], clean_mask, batch=64)
    normalized = norm_text(clean, clean_mask, stats)
    if not np.array_equal(normalized, split['t'][indices]):
        raise AssertionError('Reloaded BERT clean token features differ from training features')
    gap = next(s for s in quick if s['subset'] == 'T')
    gap_mask = gap['mask'][indices, :, 0]
    gap_features = adapter.encode(split['b'][indices], gap_mask, batch=64)
    gap_normalized = norm_text(gap_features, gap_mask, stats)
    if not np.array_equal(gap_normalized, gap['t'][indices]):
        raise AssertionError('Reloaded BERT text-gap features differ from validation bank')
    return True


def run(args):
    runtime(args.device)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    seed_all(config['seed'])
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    source_hash = sha(args.safe_source)
    text_sha = sha(args.text_delta)
    safety_hash = read_safety_audit(args.safety_audit)
    manifest_hash, manifest = read_safe_manifest(args.safe_manifest,
                                                  args.safe_source, source_hash,
                                                  safety_hash)
    adapter = FineTunedTextAdapter(args.text_delta, args.device)
    if adapter.checkpoint['provenance']['full_source_sha256'] != source_hash:
        raise ValueError('Supervised BERT delta and safe train/valid source differ')
    if adapter.checkpoint['provenance']['independent_safety_audit_sha256'] != safety_hash:
        raise ValueError('Supervised BERT delta and safety audit differ')
    if adapter.checkpoint['provenance']['safe_train_valid_manifest_sha256'] != manifest_hash:
        raise ValueError('Supervised BERT delta and safe manifest differ')
    train_raw, valid_raw, target_ids, audit = load_safe_data(
        args.safe_source, args.data_root, args.label_csv, args.safety_audit, args.out)
    cache = ROOT / 'work/问题2_监督BERT融合缓存' / text_sha[:20]
    train = prepare_split(train_raw, adapter, cache, 'fine_full_train', source_hash)
    valid = prepare_split(valid_raw, adapter, cache, 'fine_full_valid', source_hash)
    del train_raw, valid_raw
    gc.collect()
    leakage_check = check_prebert_redaction(adapter, train)
    stats = fit_normalization(train)
    normalize(train, stats)
    normalize(valid, stats)
    reference = compare_frozen_reference(args.frozen_dir, source_hash,
                                         safety_hash, stats, args.label_csv)
    if (reference['frozen_source_sha256'] != source_hash and
            reference['frozen_source_sha256'] != manifest['source_sha256']):
        raise ValueError('Frozen reference uses neither the safe nor original source')
    provenance = {'safe_source_sha256': source_hash,
                  'safe_manifest_sha256': manifest_hash,
                  'original_full_source_sha256': manifest['source_sha256'],
                  'label_csv_sha256': sha(args.label_csv),
                  'safety_audit_sha256': safety_hash,
                  'script_sha256': sha(__file__),
                  'fusion_config_sha256': sha(args.config),
                  'text_delta_sha256': text_sha,
                  'text_delta_revision': adapter.revision,
                  'text_delta_selected_epoch': adapter.checkpoint['epoch'],
                  'train_samples': len(train['ids']),
                  'validation_samples': len(valid['ids']),
                  'mask_before_bert_checks': leakage_check,
                  'frozen_reference': reference,
                  'student_selection': config['student_selection'],
                  'no_full_test_or_annex3_labels': True}
    dump(args.out / '完整数据_监督BERT融合预注册.json', provenance)
    quick = scenarios(valid, adapter, cache, stats, source_hash)
    sparse = make_scattered_suite(valid, adapter, cache, stats, source_hash)
    banks = get_banks(train, adapter, cache, stats, source_hash)
    # All BERT representations (including every missing-text pattern) are
    # cached and materialized. Fusion training no longer needs BERT on GPU.
    del adapter
    gc.collect()
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    teacher, teacher_path = train_teacher(train, valid, quick, args.device,
        args.out, provenance, stats, config['seed'], config['teacher_epochs'],
        config['patience'])
    teacher_profile, teacher_old, teacher_sparse = profile('ft_teacher', teacher,
        valid, quick, sparse, args.device)
    write_csv(args.out / '完整数据_监督BERT教师验证场景.csv',
              [{'model': 'fine_teacher', **r} for r in teacher_old + teacher_sparse])
    student, student_path = train_student(train, valid, quick, sparse, teacher,
        banks, args.device, args.out, provenance, stats, config['seed'],
        config['student_epochs'], config['patience'])
    student_profile, student_old, student_sparse = profile('ft_student', student,
        valid, quick, sparse, args.device)
    write_csv(args.out / '完整数据_监督BERT学生验证场景.csv',
              [{'model': 'fine_student', **r} for r in student_old + student_sparse])
    student_accepted = (student_profile['selection_score'] >=
                        teacher_profile['selection_score'] + .005 and
                        student_profile['clean_F1'] >= teacher_profile['clean_F1'] - .01)
    selected = 'student' if student_accepted else 'teacher'
    new_profile = student_profile if student_accepted else teacher_profile
    old_profile = reference['frozen_profile']
    delta = {key: new_profile[key] - old_profile[key] for key in new_profile}
    accepted = (delta['selection_score'] >= .005 and delta['clean_F1'] >= -.005
                and delta['scattered_TAV_mean_F1'] >= .005)
    for name, model in (('teacher', teacher), ('student', student)):
        rows = target_validation_rows(model, valid, target_ids['valid'],
                                      quick + sparse, args.device, f'fine_{name}')
        write_csv(args.out / f'附件2验证子集_监督BERT_{name}_仅分析.csv', rows)
    repeatable = checkpoint_replay({'teacher': teacher_path, 'student': student_path},
                                   valid, quick[0], args.device)
    text_repeatable = text_checkpoint_replay(args.text_delta, valid, quick,
                                             stats, args.device)
    result = {'new_selected': selected, 'new_profile': new_profile,
              'frozen_selected': reference['frozen_selected'],
              'frozen_profile': old_profile, 'delta': delta,
              'accepted_vs_frozen': bool(accepted),
              'acceptance_rule': config['acceptance_vs_frozen_full_baseline'],
              'teacher_checkpoint': teacher_path.name,
              'teacher_sha256': sha(teacher_path),
              'student_checkpoint': student_path.name,
              'student_sha256': sha(student_path),
              'text_delta_sha256': text_sha,
              'inference_repeatable': repeatable,
              'text_encoder_replay_exact': text_repeatable,
              'mask_before_bert': leakage_check,
              'elapsed_seconds': time.time() - started,
              'test_or_attachment3_labels_used': False}
    dump(args.out / '完整数据_监督BERT融合验证结论.json', result)
    print('FINISHED', json.dumps(result, ensure_ascii=False), flush=True)


def smoke(args):
    """Two synthetic utterances; no MOSEI source or GPU."""
    runtime('cpu')
    config = json.loads(TEXT_CONFIG.read_text(encoding='utf-8'))
    model = SupervisedTextEncoder(config).cpu().eval()
    raw = np.zeros((2, 3, 50), np.int64)
    raw[:, 0, :6] = [101, 2023, 100, 2024, 2025, 102]
    raw[:, 1, :6] = 1
    audio = np.zeros((2, 50, 74), np.float32)
    vision = np.zeros((2, 50, 35), np.float32)
    audio[:, 1:5] = 1
    vision[:, 1:5] = 1
    audio[1, 2] = 0
    vision[1, 2] = 0
    split = {'text_bert': raw, 'audio': audio, 'vision': vision,
             'id': ['smoke_a', 'smoke_b'],
             'raw_text': ['synthetic a', 'synthetic b']}
    with tempfile.TemporaryDirectory(prefix='q2_ft_fusion_smoke_') as directory:
        base = Path(directory)
        delta = base / 'text_delta.pt'
        save_delta(model, delta, config, {'synthetic': True}, 0, 0.)
        adapter = FineTunedTextAdapter(delta, 'cpu')
        prepared = prepare_split(split, adapter, base / 'cache', 'smoke', 'synthetic')
        check = check_prebert_redaction(adapter, prepared)
        if prepared['mask'][0, 2, 0] != 1 or prepared['mask'][1, 2, 0] != 0:
            raise AssertionError('Natural and synthetic unknown IDs were confused')
        hidden = block_mask(prepared['valid'], .3, 2113)
        observed = prepared['mask'][:, :, 0] & ~hidden
        masked = adapter.encode(prepared['b'], observed, batch=2)
        if np.any(masked[~observed] != 0):
            raise AssertionError('Missing text has nonzero token states')
        if np.array_equal(masked[observed], prepared['t'][observed]):
            raise AssertionError('Masked text reused clean BERT representations')
        checkpoint = base / 'fusion.pt'
        fusion = Fusion(gated=True).cpu().eval()
        torch.save({'model': fusion.state_dict(), 'method': 'distilled',
                    'architecture': {'hidden': 96, 'dropout': .2}}, checkpoint)
        loaded, _ = load_models({'smoke': checkpoint}, 'cpu')
        if not isinstance(loaded['smoke'], Fusion):
            raise AssertionError('Fusion checkpoint could not be reloaded')
    print('SMOKE_OK', json.dumps({'mask_before_bert': check,
                                  'natural_unk_kept': True,
                                  'joint_unknown_masked': True,
                                  'fusion_checkpoint_reload': True},
                                 ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('train', 'smoke'), required=True)
    parser.add_argument('--safe-source', type=Path, default=FULL_SOURCE)
    parser.add_argument('--safe-manifest', type=Path, default=SAFE_MANIFEST)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'E题数据/E题数据')
    parser.add_argument('--label-csv', type=Path, default=LABEL_CSV)
    parser.add_argument('--safety-audit', type=Path, default=SAFETY_AUDIT)
    parser.add_argument('--text-delta', type=Path,
                        default=TEXT_OUT / '问题2_完整数据BERT文本编码器_最优delta.pt')
    parser.add_argument('--frozen-dir', type=Path, default=FROZEN_OUT)
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    if args.mode == 'smoke':
        smoke(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
