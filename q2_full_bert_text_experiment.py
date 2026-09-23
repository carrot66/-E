"""Optional supervised BERT token encoder from the complete CMU-MOSEI train split.

Train uses full-MOSEI train labels; checkpoint selection uses its valid labels.
Neither mode accesses the complete test split. `encode` writes per-token
unnormalized representations for Fusion and does not write labels.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers import AutoModel

from q2_experiment import ROOT, derive_masks, dump, metrics, runtime, seed_all, sha, write_csv

CONFIG_PATH = ROOT / 'q2_full_bert_text_config.json'
FULL_SOURCE = ROOT / 'work/full_mosei/safe_train_valid.pkl'
SAFE_MANIFEST = ROOT / 'work/full_mosei/safe_train_valid.manifest.json'
DEFAULT_OUT = ROOT / 'outputs/问题2_完整数据BERT文本编码实验'
SAFETY_AUDIT = (ROOT / 'outputs/问题2_完整版MOSEI安全整合审计/'
                '问题2_完整版MOSEI与竞赛数据安全整合审计.json')


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding='utf-8'))
    assert config['model_revision'] and 1 <= config['unfreeze_last_layers'] <= 4
    augmentation = config['training_augmentation']
    total = sum(augmentation[k] for k in ('clean_probability', 'contiguous_probability',
                                           'scattered_probability'))
    if not np.isclose(total, 1.):
        raise ValueError('Training augmentation probabilities must sum to 1')
    return config


def video_groups(ids):
    return {x.split('$_$')[0] for x in ids}


def id_digest(ids):
    return hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()


def prepare_text(split, name):
    b, valid, _available = derive_masks(split)
    # The complete train/valid data contain a few natural tokenizer [UNK]
    # tokens. In attachment 3, artificial shared gaps also use ID 100, but
    # those positions have both A and V exactly zero. Do not discard natural
    # [UNK] merely because its token ID is 100.
    observed = valid & (b[:, 1] > 0) & (b[:, 0] != 0)
    unknown = observed & (b[:, 0] == 100)
    audio_nonzero = np.any(np.asarray(split['audio'])[unknown] != 0, axis=-1)
    vision_nonzero = np.any(np.asarray(split['vision'])[unknown] != 0, axis=-1)
    both_av_zero = int(np.sum(~audio_nonzero & ~vision_nonzero))
    if name not in ('train', 'valid'):
        # This branch is for a future feature-only Annex-3 adapter. It is not
        # used by either train or encode mode on the complete dataset.
        artificial = unknown.copy()
        artificial[unknown] = ~audio_nonzero & ~vision_nonzero
        observed &= ~artificial
    ids = list(map(str, split['id']))
    if len(ids) != len(set(ids)):
        raise ValueError(f'{name} has duplicate IDs')
    result = {'b': b, 'valid': valid, 'observed': observed, 'ids': ids,
              'natural_unk_positions': int(unknown.sum()),
              'unk_with_both_av_zero': both_av_zero}
    if name in ('train', 'valid'):
        y = np.asarray(split['classification_labels'], np.int64).reshape(-1)
        r = np.asarray(split['regression_labels'], np.float32).reshape(-1)
        if len(y) != len(b) or len(r) != len(b) or not np.isfinite(r).all():
            raise ValueError(f'{name} label shape or finiteness is invalid')
        if not np.array_equal(y, np.sign(r).astype(np.int64) + 1):
            raise ValueError(f'{name} class/intensity labels are inconsistent')
        result.update({'y': y, 'r': r})
    return result


def load_full_train_valid(source: Path):
    # This dedicated source must physically exclude the complete test split.
    with source.open('rb') as stream:
        data = pickle.load(stream)
    if set(data) != {'train', 'valid'}:
        raise ValueError('Expected a safe pickle with train/valid only; test must be absent')
    train = prepare_text(data['train'], 'train')
    valid = prepare_text(data['valid'], 'valid')
    del data
    gc.collect()
    overlap = video_groups(train['ids']) & video_groups(valid['ids'])
    if overlap:
        raise ValueError(f'Full train/valid video-group overlap: {len(overlap)}')
    return train, valid


def read_safety_audit(path: Path):
    with path.open('r', encoding='utf-8') as stream:
        report = json.load(stream)
    if (report.get('annex3_unique_matches') != 30 or
            report.get('annex3_match_split_counts') != {'test': 30} or
            report.get('train_videos_excluded_due_to_annex3_overlap') or
            report.get('safe_train_sample_count_after_annex3_exclusion') != 16326):
        raise ValueError('Independent Annex-3 leakage audit did not clear full train')
    return sha(path)


def read_safe_manifest(path: Path, source: Path, source_hash: str,
                       safety_hash: str):
    with path.open('r', encoding='utf-8') as stream:
        manifest = json.load(stream)
    if (manifest.get('safe_pickle_sha256') != source_hash or
            manifest.get('safe_pickle_bytes') != source.stat().st_size or
            manifest.get('independent_safety_audit_sha256') != safety_hash or
            manifest.get('train_valid_video_overlap') != 0 or
            'entire_test_split' not in manifest.get('excluded_fields', []) or
            manifest.get('splits', {}).get('train', {}).get('samples') != 16326 or
            manifest.get('splits', {}).get('valid', {}).get('samples') != 1871):
        raise ValueError('Safe train/valid manifest failed provenance checks')
    return sha(path), manifest


def scatter_one(valid: np.ndarray, rate: float, rng: np.random.Generator):
    positions = np.flatnonzero(valid)
    length = len(positions)
    target = min(length, max(1, round(rate * length))) if rate else 0
    hidden = np.zeros(length, bool)
    while int(hidden.sum()) < target:
        remaining = target - int(hidden.sum())
        run = min(remaining, int(rng.choice([1, 2, 3, 4], p=[.37, .31, .22, .10])))
        starts = [start for start in range(length - run + 1)
                  if not hidden[start:start + run].any()
                  and (start == 0 or not hidden[start - 1])
                  and (start + run == length or not hidden[start + run])]
        if starts:
            start = int(rng.choice(starts))
            hidden[start:start + run] = True
        else:
            free = np.flatnonzero(~hidden)
            hidden[int(rng.choice(free))] = True
    result = np.zeros_like(valid, bool)
    result[positions[hidden]] = True
    return result


def block_one(valid: np.ndarray, rate: float, rng: np.random.Generator):
    positions = np.flatnonzero(valid)
    length = len(positions)
    target = min(length, max(1, round(rate * length))) if rate else 0
    result = np.zeros_like(valid, bool)
    if target:
        start = int(rng.integers(0, length - target + 1))
        result[positions[start:start + target]] = True
    return result


def training_observed(batch, config, rng):
    observed = batch['observed'].copy()
    a = config['training_augmentation']
    for row in range(len(observed)):
        draw = rng.random()
        if draw < a['clean_probability']:
            continue
        rate = float(rng.choice(a['rates']))
        if draw < a['clean_probability'] + a['contiguous_probability']:
            hidden = block_one(batch['valid'][row], rate, rng)
        else:
            hidden = scatter_one(batch['valid'][row], rate, rng)
        observed[row] &= ~hidden
    return observed


def validation_masks(valid):
    suites = [('clean', valid['observed'])]
    for name, rate, helper, seed in [('contiguous_T30', .3, block_one, 122201),
                                    ('scattered_T20', .2, scatter_one, 122202)]:
        rng = np.random.default_rng(seed)
        hidden = np.stack([helper(v, rate, rng) for v in valid['valid']])
        suites.append((name, valid['observed'] & ~hidden))
    return suites


class SupervisedTextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config['model_id'],
                                               revision=config['model_revision'],
                                               local_files_only=True)
        self.requested_revision = config['model_revision']
        self.resolved_revision = getattr(self.bert.config, '_commit_hash', None)
        if self.resolved_revision not in (None, self.requested_revision):
            raise ValueError('The cached BERT revision differs from the pinned revision')
        for parameter in self.bert.parameters():
            parameter.requires_grad_(False)
        n = config['unfreeze_last_layers']
        for layer in self.bert.encoder.layer[-n:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
        self.token_attention = nn.Linear(768, 1)
        self.dropout = nn.Dropout(config['dropout'])
        self.head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256),
                                  nn.GELU(), nn.Dropout(config['dropout']))
        self.cls = nn.Linear(256, 3)
        self.reg = nn.Linear(256, 1)

    def encode_tokens(self, bert_input, observed, valid):
        ids = bert_input[:, 0].clone()
        attention = bert_input[:, 1].clone()
        types = bert_input[:, 2]
        hidden = valid & ~observed
        ids[hidden] = 103
        attention[hidden] = 0
        # Missing lexical IDs are removed BEFORE contextual BERT encoding.
        tokens = self.bert(input_ids=ids, attention_mask=attention,
                           token_type_ids=types).last_hidden_state
        return tokens * observed.unsqueeze(-1)

    def forward(self, bert_input, observed, valid):
        tokens = self.encode_tokens(bert_input, observed, valid)
        weights = self.token_attention(self.dropout(tokens)).squeeze(-1)
        weights = torch.softmax(weights.masked_fill(~observed, -1e4), dim=1)
        weights = weights * observed
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        hidden = self.head(self.dropout(pooled))
        logits = self.cls(hidden)
        intensity = 3 * torch.tanh(self.reg(hidden).squeeze(-1) / 3)
        return logits, intensity


def tensors(split, indices, observed, device):
    return (torch.as_tensor(split['b'][indices].copy(), dtype=torch.long, device=device),
            torch.as_tensor(observed, dtype=torch.bool, device=device),
            torch.as_tensor(split['valid'][indices], dtype=torch.bool, device=device))


@torch.inference_mode()
def evaluate(model, split, suites, device, batch_size):
    model.eval()
    rows = []
    for name, observed in suites:
        probabilities, intensities = [], []
        for start in range(0, len(split['ids']), batch_size):
            indices = np.arange(start, min(start + batch_size, len(split['ids'])))
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                enabled=device.startswith('cuda')):
                logits, reg = model(*tensors(split, indices, observed[indices], device))
            probabilities.append(logits.float().softmax(-1).cpu().numpy())
            intensities.append(reg.float().cpu().numpy())
        probability = np.concatenate(probabilities)
        intensity = np.concatenate(intensities)
        cls = probability.argmax(1)
        intensity[cls == 1] = 0.
        intensity[cls == 0] = np.minimum(intensity[cls == 0], -.01)
        intensity[cls == 2] = np.maximum(intensity[cls == 2], .01)
        rows.append({'scenario': name, **metrics(split['y'], split['r'],
                                                 probability, intensity)})
    return rows


def selection_score(rows):
    weight = {'clean': .5, 'contiguous_T30': .25, 'scattered_T20': .25}
    return sum(weight[row['scenario']] * (row['Macro_F1'] - .1 * row['MAE'])
               for row in rows)


def save_delta(model, path, config, provenance, epoch, score):
    trainable = {name for name, parameter in model.named_parameters()
                 if parameter.requires_grad}
    delta = {name: value.detach().cpu().half()
             for name, value in model.state_dict().items() if name in trainable}
    if set(delta) != trainable:
        raise AssertionError('Trainable parameters missing from delta checkpoint')
    torch.save({'delta': delta, 'config': config, 'provenance': provenance,
                'epoch': epoch, 'selection_score': score}, path)


def load_delta(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = SupervisedTextEncoder(checkpoint['config'])
    result = model.load_state_dict(checkpoint['delta'], strict=False)
    if result.unexpected_keys:
        raise ValueError(f'Unexpected delta keys: {result.unexpected_keys}')
    trainable = {name for name, parameter in model.named_parameters()
                 if parameter.requires_grad}
    if set(checkpoint['delta']) != trainable:
        raise ValueError('Delta lacks a trainable parameter or has an extra one')
    return model.to(device).eval(), checkpoint


@torch.inference_mode()
def check_hidden_text_invariance(model, split, device):
    """Changing IDs behind an absent-text mask must not change token states."""
    model.eval()
    index = np.array([0])
    hidden = block_one(split['valid'][0], .3, np.random.default_rng(95531))
    observed = split['observed'][index] & ~hidden[None]
    original = split['b'][index].copy()
    changed = original.copy()
    changed[0, 0, hidden] = 2023
    original_tensor = torch.as_tensor(original, dtype=torch.long, device=device)
    changed_tensor = torch.as_tensor(changed, dtype=torch.long, device=device)
    observed_tensor = torch.as_tensor(observed, dtype=torch.bool, device=device)
    valid_tensor = torch.as_tensor(split['valid'][index], dtype=torch.bool, device=device)
    original_tokens = model.encode_tokens(original_tensor, observed_tensor, valid_tensor)
    changed_tokens = model.encode_tokens(changed_tensor, observed_tensor, valid_tensor)
    if not torch.equal(original_tokens, changed_tokens):
        raise AssertionError('Hidden lexical IDs affected BERT token representations')
    if torch.any(original_tokens[~observed_tensor] != 0):
        raise AssertionError('Hidden text has nonzero token representation')
    return {'hidden_id_mutation_invariant': True, 'missing_token_vectors_zero': True}


def train(args, config):
    runtime(args.device)
    seed_all(config['seed'])
    args.out.mkdir(parents=True, exist_ok=True)
    source_hash = sha(args.source)
    safety_hash = read_safety_audit(args.safety_audit)
    manifest_hash, manifest = read_safe_manifest(args.safe_manifest, args.source,
                                                  source_hash, safety_hash)
    train_split, valid_split = load_full_train_valid(args.source)
    model = SupervisedTextEncoder(config).to(args.device)
    masking_checks = check_hidden_text_invariance(model, train_split, args.device)
    bert_parameters = [p for p in model.bert.parameters() if p.requires_grad]
    head_parameters = [p for name, p in model.named_parameters()
                       if not name.startswith('bert.')]
    optimizer = torch.optim.AdamW([
        {'params': bert_parameters, 'lr': config['bert_learning_rate']},
        {'params': head_parameters, 'lr': config['head_learning_rate']}],
        weight_decay=config['weight_decay'])
    counts = np.bincount(train_split['y'], minlength=3)
    if (counts == 0).any():
        raise ValueError('A train class is missing')
    class_weights = torch.as_tensor(np.sqrt(len(train_split['y']) / (3 * counts)),
                                    dtype=torch.float32, device=args.device)
    provenance = {'full_source_sha256': source_hash,
                  'safe_train_valid_manifest_sha256': manifest_hash,
                  'original_full_source_sha256': manifest['source_sha256'],
                  'independent_safety_audit_sha256': safety_hash,
                  'script_sha256': sha(__file__), 'config_sha256': sha(args.config),
                  'bert_model_id': config['model_id'],
                  'bert_requested_revision': config['model_revision'],
                  'bert_resolved_revision': model.resolved_revision,
                  'python': sys.version.split()[0], 'torch': torch.__version__,
                  'transformers': transformers.__version__, 'numpy': np.__version__,
                  'train_count': len(train_split['ids']), 'valid_count': len(valid_split['ids']),
                  'train_class_counts': counts.tolist(),
                  'train_ids_sha256': id_digest(train_split['ids']),
                  'valid_ids_sha256': id_digest(valid_split['ids']),
                  'train_natural_unk_positions': train_split['natural_unk_positions'],
                  'valid_natural_unk_positions': valid_split['natural_unk_positions'],
                  'train_unk_with_both_av_zero': train_split['unk_with_both_av_zero'],
                  'valid_unk_with_both_av_zero': valid_split['unk_with_both_av_zero'],
                  'train_valid_video_overlap': 0,
                  'masking_checks': masking_checks,
                  'selection_data': 'full MOSEI valid only; full test and attachment 3 unused'}
    dump(args.out / '问题2_完整数据BERT文本实验配置与来源.json',
         {'config': config, 'provenance': provenance})
    suites = validation_masks(valid_split)
    best = -float('inf')
    best_epoch = 0
    bad = 0
    logs = []
    checkpoint_path = args.out / '问题2_完整数据BERT文本编码器_最优delta.pt'
    for epoch in range(1, config['maximum_epochs'] + 1):
        model.train()
        rng = np.random.default_rng(config['seed'] + epoch * 1009)
        order = rng.permutation(len(train_split['ids']))
        losses = []
        for start in range(0, len(order), config['batch_size']):
            indices = order[start:start + config['batch_size']]
            batch = {key: train_split[key][indices]
                     for key in ('b', 'valid', 'observed', 'y', 'r')}
            observed = training_observed(batch, config, rng)
            inputs = tensors(train_split, indices, observed, args.device)
            y = torch.as_tensor(batch['y'], dtype=torch.long, device=args.device)
            r = torch.as_tensor(batch['r'], dtype=torch.float32, device=args.device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                enabled=args.device.startswith('cuda')):
                logits, intensity = model(*inputs)
                loss = (F.cross_entropy(logits.float(), y, weight=class_weights) +
                        config['regression_loss_weight'] *
                        F.smooth_l1_loss(intensity.float(), r))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            optimizer.step()
            losses.append(float(loss.detach()))
        rows = evaluate(model, valid_split, suites, args.device, config['batch_size'])
        value = selection_score(rows)
        log = {'epoch': epoch, 'train_loss': float(np.mean(losses)),
               'selection_score': float(value)}
        for row in rows:
            prefix = row['scenario']
            log[f'{prefix}_Macro_F1'] = row['Macro_F1']
            log[f'{prefix}_MAE'] = row['MAE']
        logs.append(log)
        write_csv(args.out / '问题2_完整数据BERT文本训练记录.csv', logs)
        print(json.dumps(log, ensure_ascii=False), flush=True)
        if value > best + 1e-4:
            best = float(value)
            best_epoch = epoch
            bad = 0
            save_delta(model, checkpoint_path, config, provenance, epoch, best)
        else:
            bad += 1
        if bad >= config['early_stopping_patience']:
            break
    model, checkpoint = load_delta(checkpoint_path, args.device)
    final_rows = evaluate(model, valid_split, suites, args.device,
                          config['batch_size'])
    write_csv(args.out / '问题2_完整数据BERT文本验证集结果.csv', final_rows)
    dump(args.out / '问题2_完整数据BERT文本训练完成状态.json',
         {'complete': True, 'best_epoch': best_epoch, 'validation_score': best,
          'checkpoint_sha256': sha(checkpoint_path),
          'checkpoint_bytes': checkpoint_path.stat().st_size,
          'test_or_attachment3_used_for_selection': False})
    print('FINISHED', json.dumps({'best_epoch': best_epoch, 'score': best,
                                 'validation': final_rows}, ensure_ascii=False), flush=True)


@torch.inference_mode()
def encode_split(model, split, device, batch_size, target):
    """Write [N, 50, 768] float16 BERT token states and matching masks/IDs."""
    target.mkdir(parents=True, exist_ok=True)
    n = len(split['ids'])
    features = np.lib.format.open_memmap(target / 'text_tokens.npy', mode='w+',
                                         dtype=np.float16, shape=(n, 50, 768))
    model.eval()
    for start in range(0, n, batch_size):
        indices = np.arange(start, min(start + batch_size, n))
        inputs = tensors(split, indices, split['observed'][indices], device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=device.startswith('cuda')):
            token = model.encode_tokens(*inputs)
        features[start:start + len(indices)] = token.float().cpu().numpy().astype(np.float16)
    features.flush()
    del features
    np.save(target / 'text_available.npy', split['observed'])
    np.save(target / 'valid_content.npy', split['valid'])
    write_csv(target / 'sample_ids.csv',
              [{'row': row, 'sample_id': sample_id}
               for row, sample_id in enumerate(split['ids'])])
    stored = np.load(target / 'text_tokens.npy', mmap_mode='r')
    if stored.shape != (n, 50, 768) or not np.isfinite(stored).all():
        raise AssertionError('Exported token representation is invalid')
    if np.any(stored[~split['observed']] != 0):
        raise AssertionError('Exported missing text has nonzero vectors')
    return {'rows': n, 'shape': [n, 50, 768], 'dtype': 'float16',
            'ids_sha256': id_digest(split['ids']),
            'text_tokens_sha256': sha(target / 'text_tokens.npy')}


def encode(args, config):
    runtime(args.device)
    model, checkpoint = load_delta(args.checkpoint, args.device)
    if checkpoint['config'] != config:
        raise ValueError('The supplied config differs from the trained checkpoint')
    source_hash = sha(args.source)
    if source_hash != checkpoint['provenance']['full_source_sha256']:
        raise ValueError('Encoding source differs from checkpoint training source')
    safety_hash = read_safety_audit(args.safety_audit)
    if safety_hash != checkpoint['provenance']['independent_safety_audit_sha256']:
        raise ValueError('Safety audit differs from checkpoint provenance')
    manifest_hash, _ = read_safe_manifest(args.safe_manifest, args.source,
                                           source_hash, safety_hash)
    if manifest_hash != checkpoint['provenance']['safe_train_valid_manifest_sha256']:
        raise ValueError('Safe manifest differs from checkpoint provenance')
    with args.source.open('rb') as stream:
        data = pickle.load(stream)
    if set(data) != {'train', 'valid'}:
        raise ValueError('Feature export requires train/valid-only safe source')
    encoded = {}
    for name in args.splits:
        split = prepare_text(data[name], name)
        encoded[name] = encode_split(model, split, args.device,
                                     config['batch_size'], args.out / 'token_features' / name)
        del split
        gc.collect()
    dump(args.out / '问题2_完整数据BERT逐token特征清单.json',
         {'model_checkpoint_sha256': sha(args.checkpoint),
          'source_sha256': source_hash, 'bert_revision': config['model_revision'],
          'features': encoded,
          'normalization': 'raw BERT states; downstream Fusion fits normalization on train only',
          'labels_in_feature_files': False})
    print('ENCODED', json.dumps(encoded, ensure_ascii=False), flush=True)


def smoke(config):
    """Tiny CPU-only path check; never opens a MOSEI pickle."""
    runtime('cpu')
    raw = np.zeros((2, 3, 50), np.int64)
    raw[:, 0, :5] = np.array([101, 2023, 100, 2024, 102])
    raw[:, 1, :5] = 1
    audio = np.zeros((2, 50, 74), np.float32)
    vision = np.zeros((2, 50, 35), np.float32)
    audio[:, 1:4] = 1
    vision[:, 1:4] = 1
    audio[1, 2] = 0
    vision[1, 2] = 0
    split = prepare_text({'text_bert': raw, 'audio': audio, 'vision': vision,
                          'id': ['smoke_natural_unk', 'smoke_shared_gap']}, 'inference')
    if not split['observed'][0, 2] or split['observed'][1, 2]:
        raise AssertionError('Natural [UNK] and artificial shared gap were confused')
    if split['valid'][:, 0].any() or split['valid'][:, 4].any():
        raise AssertionError('CLS/SEP were included as content')
    model = SupervisedTextEncoder(config).cpu().eval()
    with torch.inference_mode():
        inputs = tensors(split, np.arange(2), split['observed'], 'cpu')
        original = model.encode_tokens(*inputs)
        changed = inputs[0].clone()
        changed[1, 0, 2] = 2047
        mutated = model.encode_tokens(changed, inputs[1], inputs[2])
        if not torch.equal(original, mutated):
            raise AssertionError('Masked lexical ID leaked through BERT attention')
        logits, intensity = model(*inputs)
        if not torch.isfinite(logits).all() or not torch.isfinite(intensity).all():
            raise AssertionError('CPU inference returned a nonfinite output')
    with tempfile.TemporaryDirectory(prefix='q2_full_bert_smoke_') as temp:
        export = encode_split(model, split, 'cpu', 2, Path(temp))
        if export['shape'] != [2, 50, 768]:
            raise AssertionError('Fusion feature shape differs from specification')
    print('SMOKE_OK', json.dumps({'natural_unk_preserved': True,
                                  'artificial_shared_gap_masked': True,
                                  'special_tokens_outside_content': True,
                                  'masked_id_invariance': True,
                                  'fusion_token_shape': [2, 50, 768]},
                                 ensure_ascii=False), flush=True)


def check_source(args):
    source_hash = sha(args.source)
    safety_hash = read_safety_audit(args.safety_audit)
    manifest_hash, manifest = read_safe_manifest(args.safe_manifest, args.source,
                                                  source_hash, safety_hash)
    print('SOURCE_OK', json.dumps({'safe_pickle_sha256': source_hash,
                                   'safe_manifest_sha256': manifest_hash,
                                   'safety_audit_sha256': safety_hash,
                                   'splits': {name: manifest['splits'][name]['samples']
                                              for name in ('train', 'valid')},
                                   'test_excluded': True},
                                  ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('train', 'encode', 'smoke',
                                           'check-source'), required=True)
    parser.add_argument('--source', type=Path, default=FULL_SOURCE)
    parser.add_argument('--safe-manifest', type=Path, default=SAFE_MANIFEST)
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--safety-audit', type=Path, default=SAFETY_AUDIT)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--splits', nargs='+', choices=('train', 'valid'),
                        default=('train', 'valid'))
    args = parser.parse_args()
    config = read_config(args.config)
    if args.mode == 'train':
        train(args, config)
    elif args.mode == 'encode':
        if args.checkpoint is None:
            args.checkpoint = args.out / '问题2_完整数据BERT文本编码器_最优delta.pt'
        encode(args, config)
    else:
        if args.mode == 'smoke':
            smoke(config)
        else:
            check_source(args)


if __name__ == '__main__':
    main()
