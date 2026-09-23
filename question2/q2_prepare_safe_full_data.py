"""Create a train/valid-only CMU-MOSEI pickle for leakage-safe Q2 training.

This one-time preparation step opens the source monolithic pickle, which also
contains test, but it never indexes test. Its output has only train and valid
and only the fields used by the Q2 trainers. Subsequent training/encoding must
read the safe output rather than deserialize the monolithic source again.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import pickle
from pathlib import Path

import numpy as np

from q2_experiment import ROOT, dump, sha


EXPECTED_FULL_SHA256 = '45eccfb748a87c80ecab9bfac29582e7b1466bf6605ff29d3b338a75120bf791'
FIELDS = ('id', 'raw_text', 'text_bert', 'audio', 'vision',
          'classification_labels', 'regression_labels')
SPLITS = ('train', 'valid')


def video_id(sample_id: str) -> str:
    video, sep, clip = sample_id.partition('$_$')
    if not sep or not video or not clip:
        raise ValueError(f'Invalid sample ID: {sample_id!r}')
    return video


def check_safety_report(path: Path):
    report = json.loads(path.read_text(encoding='utf-8'))
    if report.get('annex3_unique_matches') != 30:
        raise ValueError('Annex-3 safety audit lacks 30 unique matches')
    if report.get('annex3_match_split_counts') != {'test': 30}:
        raise ValueError('Annex-3 features match train/valid or are ambiguous')
    if report.get('train_videos_excluded_due_to_annex3_overlap'):
        raise ValueError('Full train contains videos overlapping Annex 3')
    if report.get('safe_train_sample_count_after_annex3_exclusion') != 16326:
        raise ValueError('Safe full-train size differs from audited 16,326')
    return report


def load_train_valid_csv(csv_path: Path):
    rows = {'train': {}, 'valid': {}}
    with csv_path.open('r', encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            mode = row['mode']
            if mode == 'test':
                # Explicitly skip full-test IDs and labels, including the
                # anonymous Annex-3 source records.
                continue
            if mode not in rows:
                raise ValueError(f'Unexpected split in label.csv: {mode}')
            sample_id = f"{row['video_id']}$_${row['clip_id']}"
            if sample_id in rows[mode]:
                raise ValueError(f'Duplicate label.csv ID: {sample_id}')
            value = float(row['label'])
            label = int(np.sign(value)) + 1
            if (not np.isfinite(value) or abs(value) > 3 or
                    {'Negative': 0, 'Neutral': 1, 'Positive': 2}.get(row['annotation']) != label):
                raise ValueError(f'Invalid label or annotation: {sample_id}')
            rows[mode][sample_id] = (value, row['text'].strip())
    return rows


def copy_split(source, name: str, csv_rows):
    if not set(FIELDS).issubset(source):
        raise ValueError(f'{name} is missing required fields')
    ids = list(map(str, source['id']))
    count = len(ids)
    if len(set(ids)) != count or set(ids) != set(csv_rows):
        raise ValueError(f'{name} IDs are duplicated or differ from label.csv')
    bert = np.asarray(source['text_bert'])
    audio = np.asarray(source['audio'])
    vision = np.asarray(source['vision'])
    if bert.shape != (count, 3, 50) or audio.shape != (count, 50, 74) or vision.shape != (count, 50, 35):
        raise ValueError(f'{name} feature dimensions differ from Q2 specification')
    if not np.isfinite(audio).all() or not np.isfinite(vision).all():
        raise ValueError(f'{name} A/V contains NaN or Inf')
    if not np.issubdtype(bert.dtype, np.integer) or np.min(bert[:, 0]) < 0 or np.max(bert[:, 0]) >= 30522:
        raise ValueError(f'{name} invalid BERT token IDs')
    y = np.asarray(source['classification_labels'], np.int64).reshape(-1)
    r = np.asarray(source['regression_labels'], np.float32).reshape(-1)
    raw_text = list(map(str, source['raw_text']))
    if len(y) != count or len(r) != count or len(raw_text) != count:
        raise ValueError(f'{name} label or text count mismatch')
    if not np.isfinite(r).all() or (np.abs(r) > 3).any() or not np.array_equal(y, np.sign(r).astype(np.int64) + 1):
        raise ValueError(f'{name} class/intensity label inconsistency')
    for sample_id, value, text in zip(ids, r, raw_text):
        csv_value, csv_text = csv_rows[sample_id]
        if abs(float(value) - csv_value) > 1e-6 or text.strip() != csv_text:
            raise ValueError(f'{name}/{sample_id}: pickle and label.csv mismatch')
    prepared = {
        'id': ids,
        'raw_text': raw_text,
        'text_bert': bert.astype(np.int32, copy=True),
        'audio': audio.astype(np.float32, copy=True),
        'vision': vision.astype(np.float32, copy=True),
        'classification_labels': y.astype(np.int64, copy=True),
        'regression_labels': r.astype(np.float32, copy=True),
    }
    joint_zero = (prepared['audio'] == 0).all(-1) & (prepared['vision'] == 0).all(-1)
    unknown = (prepared['text_bert'][:, 0] == 100) & (prepared['text_bert'][:, 1] > 0)
    report = {'samples': count, 'videos': len({video_id(x) for x in ids}),
              'class_counts': np.bincount(y, minlength=3).tolist(),
              'bert_token_shape': list(prepared['text_bert'].shape),
              'audio_shape': list(prepared['audio'].shape),
              'vision_shape': list(prepared['vision'].shape),
              'natural_unk_positions': int((unknown & ~joint_zero).sum()),
              'joint_zero_unk_positions': int((unknown & joint_zero).sum())}
    return prepared, report


def main(args):
    source = Path(args.source).resolve(strict=True)
    label_csv = source.with_name('label.csv')
    safety_path = Path(args.safety_audit).resolve(strict=True)
    target = Path(args.output).resolve()
    if target == source:
        raise ValueError('Output must differ from source')
    target.parent.mkdir(parents=True, exist_ok=True)
    check_safety_report(safety_path)
    source_sha = sha(source)
    if source_sha != args.expected_source_sha256:
        raise ValueError('Source pickle SHA-256 differs from the audited package')
    csv_rows = load_train_valid_csv(label_csv)
    with source.open('rb') as stream:
        original = pickle.load(stream)
    if not {'train','valid','test'}.issubset(original):
        raise ValueError('Full pickle lacks expected split keys')
    # Do not inspect original['test']; the monolithic pickle is discarded after
    # copying only train and valid into a new object.
    safe = {}
    reports = {}
    for name in SPLITS:
        safe[name], reports[name] = copy_split(original[name], name, csv_rows[name])
    del original
    gc.collect()
    if reports['train']['samples'] != 16326 or reports['valid']['samples'] != 1871:
        raise ValueError('Unexpected train/valid sizes')
    train_ids = set(safe['train']['id'])
    valid_ids = set(safe['valid']['id'])
    train_videos = {video_id(x) for x in train_ids}
    valid_videos = {video_id(x) for x in valid_ids}
    if train_ids & valid_ids or train_videos & valid_videos:
        raise ValueError('Train/valid sample or video overlap')
    temp = target.with_name(target.name + f'.tmp-{os.getpid()}')
    try:
        with temp.open('wb') as stream:
            pickle.dump(safe, stream, protocol=4)
        with temp.open('rb') as stream:
            reloaded = pickle.load(stream)
        if set(reloaded) != {'train','valid'} or any(set(reloaded[name]) != set(FIELDS) for name in SPLITS):
            raise AssertionError('Safe pickle contains unexpected split or field')
        if any(len(reloaded[name]['id']) != reports[name]['samples'] for name in SPLITS):
            raise AssertionError('Safe pickle changed sample counts')
        del reloaded
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()
    manifest = {'source_sha256': source_sha, 'label_csv_sha256': sha(label_csv),
                'independent_safety_audit_sha256': sha(safety_path),
                'safe_pickle_sha256': sha(target), 'safe_pickle_bytes': target.stat().st_size,
                'splits': reports, 'train_valid_video_overlap': 0,
                'excluded_fields': ['text', 'annotations', 'entire_test_split'],
                'dtype_changes': {'text_bert': 'int32', 'audio': 'float32',
                                  'vision': 'float32', 'regression_labels': 'float32'},
                'policy': 'Only the one-time preparation step deserializes the monolithic source; subsequent trainers must load this train/valid-only pickle.'}
    dump(target.with_suffix('.manifest.json'), manifest)
    print(json.dumps({'output': str(target), 'sha256': manifest['safe_pickle_sha256'],
                      'samples': {k: v['samples'] for k,v in reports.items()},
                      'train_valid_video_overlap': 0}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=ROOT/'work/full_mosei/CMU-MOSEI_完整版_对齐特征与标签/aligned_50.pkl')
    parser.add_argument('--safety-audit', default=ROOT/'outputs/question2/问题2_完整版MOSEI安全整合审计/问题2_完整版MOSEI与竞赛数据安全整合审计.json')
    parser.add_argument('--output', default=ROOT/'work/full_mosei/safe_train_valid.pkl')
    parser.add_argument('--expected-source-sha256', default=EXPECTED_FULL_SHA256)
    main(parser.parse_args())
