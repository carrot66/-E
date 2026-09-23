"""Question 1: audit -> frozen extraction -> temporal pooling -> verification."""
import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import sys
import traceback
from pathlib import Path
import numpy as np
from alignment import time_grid, overlap_pool, interval_coverage, overlap_scalar
from media import probe, audio_decode


def write_json(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def discover(root, labels=None):
    import pandas as pd
    matches = list(root.rglob('label-100.xlsx')) if labels is None else [labels]
    if len(matches) != 1:
        raise ValueError(f'Expected one label-100.xlsx, found {len(matches)}; use --labels')
    labels = matches[0].resolve()
    df = pd.read_excel(labels, dtype={'video_id': str})
    required = {'video_id', 'clip_id', 'text', 'label', 'annotation'}
    if not required.issubset(df.columns) or df[list(required)].isna().any().any():
        raise ValueError('Missing columns or null labels/transcripts')
    rows, seen = [], set()
    for row_index, r in enumerate(df.to_dict('records')):
        if float(r['clip_id']) != int(r['clip_id']):
            raise ValueError('Noninteger clip_id')
        clip = str(int(r['clip_id']))
        video_id = r['video_id']
        if not video_id or any(c in video_id for c in '/\\') or video_id in ('.', '..'):
            raise ValueError('Unsafe video_id')
        sid = f'{video_id}$_${clip}'
        if sid in seen:
            raise ValueError(f'Duplicate sample: {sid}')
        seen.add(sid)
        path = labels.parent / video_id / f'{clip}.mp4'
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(dict(row_index=row_index, id=sid, video_id=video_id, clip_id=clip,
                         text=str(r['text']), label=float(r['label']), annotation=str(r['annotation']),
                         source=path.relative_to(labels.parent).as_posix(),
                         source_sha256=sha256(path)))
    actual = {p.relative_to(labels.parent).as_posix() for p in labels.parent.rglob('*.mp4')}
    expected = {r['source'] for r in rows}
    if actual != expected:
        raise ValueError(f'Unlisted/missing videos: {sorted(actual ^ expected)}')
    return labels, rows


def directory_sha256(root):
    """Hash model files and relative names so a local model bundle is auditable."""
    h = hashlib.sha256()
    root = Path(root)
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        rel = path.relative_to(root).as_posix().encode('utf-8')
        h.update(rel + b'\0')
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
    return h.hexdigest()


def model_snapshots(out, cache, models_dir=None):
    from huggingface_hub import HfApi, snapshot_download
    from extractors import MODEL_IDS
    lockfile = out / 'models.lock.json'
    lock = json.loads(lockfile.read_text(encoding='utf-8')) if lockfile.exists() else {}
    paths = {}
    if models_dir is not None:
        models_dir = Path(models_dir).resolve()
        for key, repo in MODEL_IDS.items():
            local = models_dir / key
            if not local.is_dir():
                raise FileNotFoundError(
                    f'Missing local model directory {local}. Expected models-dir/{key}/ '
                    'with config, tokenizer/processor and weight files.')
            files_sha256 = directory_sha256(local)
            if key not in lock:
                lock[key] = dict(repo=repo, revision='local', files_sha256=files_sha256)
                write_json(lockfile, lock)
            item = lock[key]
            if item.get('repo') != repo or item.get('revision') != 'local':
                raise ValueError('Model lock does not match local model bundle; use a new output directory')
            if item.get('files_sha256') != files_sha256:
                raise ValueError(f'Local model files changed for {key}; use a new output directory')
            paths[key] = str(local)
        return paths, lock
    for key, repo in MODEL_IDS.items():
        if key not in lock:
            try:
                cached = snapshot_download(
                    repo_id=repo, cache_dir=str(cache), local_files_only=True,
                    allow_patterns=['*.json', '*.txt', '*.safetensors', 'pytorch_model.bin', '*.model'])
                revision = Path(cached).name
                lock[key] = dict(repo=repo, revision=revision,
                                 files_sha256=directory_sha256(cached), source='local_cache')
                write_json(lockfile, lock)
            except Exception:
                cached = None
            if cached is not None:
                paths[key] = str(cached)
                continue
            try:
                revision = HfApi().model_info(repo).sha
            except Exception as exc:
                endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co')
                raise RuntimeError(
                    f'Cannot reach the Hugging Face endpoint {endpoint}. '
                    'Use HF_ENDPOINT with a reachable mirror, or download model snapshots '
                    'and pass --models-dir models.') from exc
            lock[key] = dict(repo=repo, revision=revision)
            write_json(lockfile, lock)
        item = lock[key]
        if item['repo'] != repo:
            raise ValueError('Model lock does not match code; use a new output directory')
        if item.get('source') == 'local_cache':
            try:
                cached = snapshot_download(
                    repo_id=repo, cache_dir=str(cache), revision=item['revision'], local_files_only=True,
                    allow_patterns=['*.json', '*.txt', '*.safetensors', 'pytorch_model.bin', '*.model'])
            except Exception as exc:
                raise RuntimeError(f'Locked local model cache is incomplete for {key}; restore the cache or use --models-dir.') from exc
            if item.get('files_sha256') != directory_sha256(cached):
                raise ValueError(f'Local cached model files changed for {key}; use a new output directory')
            paths[key] = str(cached)
            continue
        paths[key] = snapshot_download(repo_id=repo, revision=item['revision'], cache_dir=str(cache),
                                      allow_patterns=['*.json', '*.txt', '*.safetensors', 'pytorch_model.bin', '*.model'])
    return paths, lock


def extract_sample(row, path, dest, engines, args, fingerprint):
    dest.mkdir(parents=True, exist_ok=True)
    info = probe(path)
    grid = time_grid(info['duration'], args.step)
    signal, observed = audio_decode(path, info, dest / 'audio.wav')
    words, tv, ti, tm, chunks, text_alignment_error = engines.text(
        signal, observed, row['text'], info['duration'], args.min_confidence)
    av, ai, am = engines.audio(signal, observed)
    ai = np.clip(ai, 0., info['duration'])
    frames, vv, vi, vm = engines.vision(path, info, dest / 'frames', args.fps)
    arrays = dict(time=grid, sequence_mask=np.ones(len(grid), bool),
                  audio_observed=observed, sample_rate=np.array(16000),
                  label=np.array(row['label'], np.float32),
                  text_global=(tv.mean(axis=0) if len(tv) else np.zeros(engines.text_dim, np.float32)),
                  text_global_mask=np.array(bool(len(tv)), bool))
    links = []
    quality_sources = {
        'text': np.array([w.get('confidence', 0.) if w.get('valid', False) else 0.
                          for w in words], np.float32),
        'audio': am.astype(np.float32),
        'vision': vm.astype(np.float32),
    }
    for name, val, intervals, valid, std in [('text', tv, ti, tm, False),
                                          ('audio', av, ai, am, True),
                                          ('vision', vv, vi, vm, False)]:
        pooled, mask, weights = overlap_pool(intervals, val, grid, valid, std=std)
        quality, _, _ = overlap_scalar(intervals, quality_sources[name], grid, valid)
        coverage = interval_coverage(intervals, grid, valid)
        arrays.update({name: pooled, name+'_mask': mask, name+'_native': val,
                       name+'_intervals': intervals, name+'_native_mask': valid,
                       name+'_native_quality': quality_sources[name],
                       name+'_weights': weights.astype(np.float32),
                       name+'_coverage': coverage,
                       name+'_quality': quality.astype(np.float32)})
        links.append((name, weights))
    # No length truncation: aggregate padding is performed only after all samples complete.
    with (dest / 'features.npz.tmp').open('wb') as f:
        np.savez_compressed(f, **arrays)
    (dest / 'features.npz.tmp').replace(dest / 'features.npz')
    timeline = []
    for k, (a, b) in enumerate(grid):
        timeline.append(dict(position=k, start=float(a), end=float(b),
                             **{name+'_source_indices': np.flatnonzero(w[k] > 0).tolist() for name, w in links}))
    media_streams = {s['type']: s['index'] for s in info['streams']}
    metadata = dict(**row, fingerprint=fingerprint, media=info, length=len(grid),
                    modality_sources={
                        'text': {'file': 'label-100.xlsx', 'row_index': row['row_index'], 'field': 'text'},
                        'audio': {'file': row['source'], 'stream_index': media_streams.get('audio')},
                        'vision': {'file': row['source'], 'stream_index': media_streams.get('video')},
                    },
                    words=words, text_chunks=chunks, text_alignment_error=text_alignment_error,
                    frames=frames, timeline=timeline,
                    quality_summary={
                        'word_count': len(words),
                        'valid_word_count': int(sum(w.get('valid', False) for w in words)),
                        'mean_valid_word_confidence': float(np.mean([w['confidence'] for w in words if w.get('valid', False)]))
                        if any(w.get('valid', False) for w in words) else 0.,
                        'frame_count': len(frames),
                        'valid_face_frame_count': int(sum(r['valid'] for r in frames)),
                        'multiple_face_frame_count': int(sum(r['face_count'] > 1 for r in frames)),
                    },
                    valid_counts={m:int(arrays[m+'_mask'].sum()) for m in ('text','audio','vision')},
                    feature_sha256=sha256(dest / 'features.npz'),
                    audio_sha256=sha256(dest / 'audio.wav'))
    write_json(dest / 'metadata.json', metadata)
    return metadata


def aggregate(out, rows):
    records = [json.loads((out/'samples'/r['id']/'metadata.json').read_text(encoding='utf-8')) for r in rows]
    maximum = max(r['length'] for r in records)
    data = dict(ids=np.array([r['id'] for r in rows]),
                lengths=np.array([r['length'] for r in records], np.int32),
                labels=np.array([r['label'] for r in rows], np.float32),
                sequence_mask=np.zeros((len(rows), maximum), bool),
                time=np.full((len(rows), maximum, 2), -1., np.float64))
    data['text_global'] = np.zeros((len(rows), 0), np.float32)
    data['text_global_mask'] = np.zeros(len(rows), bool)
    for i, row in enumerate(rows):
        with np.load(out/'samples'/row['id']/'features.npz', allow_pickle=False) as f:
            n = len(f['time'])
            if data['text_global'].shape[1] == 0:
                data['text_global'] = np.zeros((len(rows), f['text_global'].shape[0]), np.float32)
            data['text_global'][i] = f['text_global']
            data['text_global_mask'][i] = f['text_global_mask']
            data['sequence_mask'][i, :n] = True
            data['time'][i, :n] = f['time']
            for name in ('text','audio','vision'):
                if name not in data:
                    data[name] = np.zeros((len(rows), maximum, f[name].shape[1]), np.float32)
                    data[name+'_mask'] = np.zeros((len(rows), maximum), bool)
                    data[name+'_coverage'] = np.zeros((len(rows), maximum), np.float32)
                    data[name+'_quality'] = np.zeros((len(rows), maximum), np.float32)
                data[name][i, :n] = f[name]
                data[name+'_mask'][i, :n] = f[name+'_mask']
                data[name+'_coverage'][i, :n] = f[name+'_coverage']
                data[name+'_quality'][i, :n] = f[name+'_quality']
    with (out/'aligned_features.npz.tmp').open('wb') as f:
        np.savez_compressed(f, **data)
    (out/'aligned_features.npz.tmp').replace(out/'aligned_features.npz')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument('--labels', type=Path)
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'outputs')
    p.add_argument('--cache', type=Path, default=Path(__file__).resolve().parent/'.cache'/'huggingface')
    p.add_argument('--models-dir', type=Path,
                   help='Offline model bundle containing aligner/, text/ and vision/ snapshots')
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--step', type=float, default=.5)
    p.add_argument('--fps', type=float, default=5.)
    p.add_argument('--min-confidence', type=float, default=.05)
    p.add_argument('--expected-count', type=int, default=100)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--audit-only', action='store_true')
    args = p.parse_args()
    if (not np.isfinite(args.step) or not np.isfinite(args.fps) or args.step <= 0 or args.fps <= 0
            or not 0 <= args.min_confidence <= 1 or args.limit < 0):
        p.error('Invalid numeric parameters')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(out/'processing.log', encoding='utf-8')])
    labels, rows = discover(args.root.resolve(), args.labels)
    if len(rows) != args.expected_count:
        raise ValueError(f'Expected {args.expected_count} samples, found {len(rows)}')
    inventory = dict(label_file=str(labels), label_sha256=sha256(labels), count=len(rows), samples=rows)
    if args.audit_only:
        for row in rows:
            row['media'] = probe(labels.parent/row['source'])
        write_json(out/'input_audit.json', inventory)
        logging.info('Input audit passed: %d/%d', len(rows), args.expected_count)
        return
    import torch
    from extractors import Extractors
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Check PyTorch/driver or pass --device cpu')
    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    snapshots, lock = model_snapshots(out, args.cache, args.models_dir)
    runtime = {name:importlib.metadata.version(name) for name in
               ('numpy','pandas','av','soundfile','opensmile','opencv-python-headless',
                'torch','transformers','tokenizers','num2words','Pillow')}
    settings = dict(step=args.step, fps=args.fps, min_confidence=args.min_confidence,
                    sample_rate=16000, seed=2026, models=lock, label_sha256=inventory['label_sha256'],
                    runtime=runtime, device=args.device,
                    model_source=str(args.models_dir.resolve()) if args.models_dir else 'huggingface_cache',
                    hf_endpoint=os.environ.get('HF_ENDPOINT', 'https://huggingface.co'),
                    sources={r['id']:r['source_sha256'] for r in rows},
                    code={f.name:sha256(f) for f in Path(__file__).parent.glob('*.py')})
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    if (out/'run.json').exists():
        old = json.loads((out/'run.json').read_text(encoding='utf-8'))
        if old['fingerprint'] != fingerprint:
            raise ValueError('Code/data/parameters/runtime changed. Select a new --output to prevent stale results.')
    write_json(out/'input_audit.json', inventory)
    write_json(out/'run.json', dict(settings=settings, fingerprint=fingerprint,
                                  python=sys.version, platform=platform.platform(), device=args.device,
                                  gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None,
                                  command=sys.argv,
                                  packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}))
    engines = Extractors(snapshots, args.device)
    engines.schema['vision_detector']['cascade_sha256'] = sha256(Path(engines.face_cascade_path))
    write_json(out/'feature_schema.json', engines.schema)
    selected = rows[:args.limit] if args.limit else rows
    failures = []
    for i, row in enumerate(selected):
        dest = out/'samples'/row['id']
        try:
            meta = dest/'metadata.json'
            reusable = False
            if meta.exists() and (dest/'features.npz').exists() and (dest/'audio.wav').exists():
                m = json.loads(meta.read_text(encoding='utf-8'))
                reusable = (m['fingerprint']==fingerprint and
                            m['feature_sha256']==sha256(dest/'features.npz') and
                            m['audio_sha256']==sha256(dest/'audio.wav') and
                            all((dest/f['image']).exists() for f in m['frames']))
            if not reusable:
                extract_sample(row, labels.parent/row['source'], dest, engines, args, fingerprint)
            logging.info('[%d/%d] %s %s', i+1, len(selected), row['id'], 'cached' if reusable else 'extracted')
        except Exception:
            failures.append(dict(id=row['id'], traceback=traceback.format_exc()))
            logging.exception('Failed: %s', row['id'])
            if args.device=='cuda':
                torch.cuda.empty_cache()
    write_json(out/'failures.json', failures)
    manifest = []
    for row in rows:
        path = out/'samples'/row['id']/'metadata.json'
        status = 'failed' if any(r['id']==row['id'] for r in failures) else ('complete' if path.exists() else 'pending')
        manifest.append(dict(**row, status=status, metadata=f"samples/{row['id']}/metadata.json",
                             features=f"samples/{row['id']}/features.npz"))
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in manifest), encoding='utf-8')
    if failures:
        raise RuntimeError(f'{len(failures)} samples failed; see failures.json. Rerun the same command to retry.')
    if all(r['status']=='complete' for r in manifest):
        aggregate(out, rows)
        from verify import verify
        verify(out)
        from make_report import make_report
        make_report(out)
        logging.info('All samples extracted and verified: %d', len(rows))
    else:
        logging.info('Partial smoke run complete. Run without --limit for the full dataset.')


if __name__ == '__main__':
    main()
