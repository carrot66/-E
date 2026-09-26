"""Shared I/O, metrics and official aligned-data adapter for Q3."""
from __future__ import annotations
import csv
import hashlib
import json
import pickle
import random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODS = ('text', 'audio', 'vision')
LABELS = ('Negative', 'Neutral', 'Positive')


def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csv_write(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def csv_read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(part)
    return h.hexdigest()


class OfficialUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if np.__version__.startswith('1.'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


def read_pickle(path):
    with Path(path).open('rb') as f: return OfficialUnpickler(f).load()


def seed_all(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def scores(y, r, prob, reg):
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, confusion_matrix
    pred = prob.argmax(1)
    return {'Accuracy': float(accuracy_score(y, pred)),
            'Macro_F1': float(f1_score(y, pred, labels=[0, 1, 2], average='macro', zero_division=0)),
            'Weighted_F1': float(f1_score(y, pred, average='weighted', zero_division=0)),
            'MAE': float(mean_absolute_error(r, reg)),
            'Pearson': float(np.corrcoef(r, reg)[0, 1]) if np.std(reg) > 1e-12 and np.std(r) > 1e-12 else None,
            'sign_conflict_rate': float(np.mean(pred != np.sign(reg) + 1)),
            'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()}


def adapt(source, tokenizer, labelled=False):
    b = np.asarray(source['text_bert'])
    if b.ndim == 2:
        source = {k: np.asarray(v)[None] for k, v in source.items()}
        b = np.asarray(source['text_bert'])
    n = len(b)
    if b.shape != (n, 3, 50) or not np.isfinite(b).all() or not np.equal(b, np.round(b)).all():
        raise ValueError('Expected integer-valued aligned text_bert (N,3,50)')
    b = b.astype(np.int64)
    if not np.isin(b[:, 1:], [0, 1]).all() or b[:, 0].min() < 0 or b[:, 0].max() >= len(tokenizer):
        raise ValueError('Invalid token IDs / masks')
    av = [np.asarray(source[k], np.float32) for k in MODS[1:]]
    for x, d in zip(av, (74, 35)):
        if x.shape != (n, 50, d) or not np.isfinite(x).all(): raise ValueError('Invalid aligned A/V features')
    ids = list(map(str, source['id'])); raw = list(map(str, source['raw_text']))
    if len(ids) != n or len(set(ids)) != n: raise ValueError('Duplicate or malformed sample IDs')
    text_valid = np.zeros((n, 50), bool)
    offsets = []; words = []; mapping = []; audit = []
    for i in range(n):
        cls = np.flatnonzero(b[i, 0] == tokenizer.cls_token_id)
        sep = np.flatnonzero(b[i, 0] == tokenizer.sep_token_id)
        if len(cls) != 1 or len(sep) != 1 or sep[0] <= cls[0] + 1: raise ValueError(f'Bad CLS/SEP: {ids[i]}')
        text_valid[i, cls[0]+1:sep[0]] = True
        enc = tokenizer(raw[i], max_length=50, truncation=True, padding='max_length', return_offsets_mapping=True)
        exact = bool(np.array_equal(enc['input_ids'], b[i, 0]))
        offsets.append(enc['offset_mapping'] if exact else [(0, 0)] * 50)
        words.append(enc.word_ids() if exact else [None] * 50)
        mapping.append(exact)
        audit.append({'sample_id': ids[i], 'text_token_ids_match_raw': exact,
                      'text_content_rows': int(text_valid[i].sum()),
                      'audio_observed_rows': int(np.any(av[0][i] != 0, -1).sum()),
                      'vision_observed_rows': int(np.any(av[1][i] != 0, -1).sum()),
                      'at_50_token_limit': bool(b[i, 1].sum() == 50),
                      'av_alignment_status': 'unverified: separate modality axes'})
    # Do NOT impose text CLS/SEP offsets on official A/V feature rows.
    mask = np.stack([text_valid & (b[:, 1] > 0) & (b[:, 0] != 0),
                     np.any(av[0] != 0, -1), np.any(av[1] != 0, -1)], -1)
    result = {'b': b, 'a': av[0], 'v': av[1], 'mask': mask, 'ids': ids, 'raw': raw,
              'offsets': np.array(offsets), 'word_ids': words, 'text_mapping': mapping, 'audit': audit}
    if labelled:
        r = np.asarray(source['regression_labels'], np.float32).reshape(-1)
        yy = np.asarray(source['classification_labels']).reshape(-1)
        if not np.isfinite(r).all() or np.any(np.abs(r) > 3) or not np.array_equal(yy, np.sign(r) + 1):
            raise ValueError('Classification/regression label inconsistency')
        result.update(y=yy.astype(np.int64), r=r)
    return result


def fit_stats(data):
    stats = {}
    for key, m in [('a', 1), ('v', 2)]:
        x = data[key][data['mask'][:, :, m]]
        if not len(x): raise ValueError('Training modality has no observations')
        stats[key + '_mean'] = x.mean(0, dtype=np.float64).astype(np.float32)
        stats[key + '_std'] = np.maximum(x.std(0, dtype=np.float64), 1e-5).astype(np.float32)
    return stats


def normalize(data, stats):
    out = dict(data)
    for key, m in [('a', 1), ('v', 2)]:
        out[key] = ((data[key] - stats[key + '_mean']) / stats[key + '_std']) * data['mask'][:, :, m, None]
    return out


def special_data(root, tokenizer):
    folders = list(Path(root).glob('附件4*/**/对齐版本'))
    if len(folders) != 1: raise ValueError('Expected exactly one attachment-4 aligned directory')
    files = sorted(folders[0].glob('*.pkl'))
    if len(files) != 20: raise ValueError(f'Expected 20 attachment-4 files, found {len(files)}')
    samples = [read_pickle(f) for f in files]
    for f, sample in zip(files, samples):
        if str(sample['id']) != f.stem: raise ValueError(f'ID mismatch: {f}')
    merged = {k: np.stack([s[k] for s in samples]) for k in ('id', 'raw_text', 'text_bert', 'audio', 'vision')}
    data = adapt(merged, tokenizer)
    data['videos'] = [str(folders[0] / 'videos' / (sid + '.mp4')) for sid in data['ids']]
    return data, files
