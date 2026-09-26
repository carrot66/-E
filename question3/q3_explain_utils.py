"""Exact modality Shapley, input-level window interventions and reviewable evidence."""
import itertools
import json
import math
from pathlib import Path
import numpy as np
import torch
from q3_common import MODS, LABELS, csv_write, dump, sha, normalize, special_data
from q3_base_model import FrozenText, Q3Model
from q3_train_utils import subset, cache_text


def exact_shapley(values):
    values = np.asarray(values)
    if values.shape[0] != 8: raise ValueError('Need 8 coalitions ordered by bitmask T=1,A=2,V=4')
    phi = np.zeros((3,) + values.shape[1:])
    for m in range(3):
        for s in range(8):
            if s & (1 << m): continue
            k = s.bit_count()
            phi[m] += math.factorial(k) * math.factorial(2-k) / 6 * (values[s | (1 << m)] - values[s])
    return phi


class Predictor:
    def __init__(self, model, encoder, device):
        self.model = model.eval(); self.encoder = encoder; self.device = device

    @torch.inference_mode()
    def masks(self, data, i, masks, batch=64, zero_feature=False):
        masks = np.asarray(masks, bool)
        logits = []; reg = []; probabilities = []
        for start in range(0, len(masks), batch):
            active = masks[start:start+batch]; n = len(active)
            text = np.repeat(data['t'][i:i+1], n, 0)
            if zero_feature:
                text *= active[:, :, 0, None]
                effective_mask = np.repeat(data['mask'][i:i+1], n, 0)
            else:
                effective_mask = active
                changed = np.flatnonzero(np.any(active[:, :, 0] != data['mask'][i, :, 0], axis=1))
                if len(changed):
                    # Entirely absent text produces exact zero features without a redundant BERT pass.
                    absent = changed[~active[changed, :, 0].any(1)]; text[absent] = 0
                    changed = changed[active[changed, :, 0].any(1)]
                    if len(changed): text[changed] = self.encoder.encode(np.repeat(data['b'][i:i+1], len(changed), 0), active[changed, :, 0])
            a = np.repeat(data['a'][i:i+1], n, 0) * active[:, :, 1, None]
            v = np.repeat(data['v'][i:i+1], n, 0) * active[:, :, 2, None]
            args = [torch.as_tensor(x, device=self.device) for x in (text, a, v, effective_mask)]
            l, r, _ = self.model(*args)
            logits.append(l.cpu().numpy()); reg.append(r.cpu().numpy()); probabilities.append(l.softmax(-1).cpu().numpy())
        return np.concatenate(logits), np.concatenate(reg), np.concatenate(probabilities)


def units(data, i, m):
    observed = np.flatnonzero(data['mask'][i, :, m]).tolist()
    if m != 0 or not data['text_mapping'][i]: return [[t] for t in observed]
    groups = []
    for _, positions in itertools.groupby(observed, key=lambda t: data['word_ids'][i][t]):
        groups.append(list(positions))
    return groups


def text_span(data, i, positions):
    if not data['text_mapping'][i] or not positions: return '', '', '', 'unresolved'
    intervals = data['offsets'][i, positions]
    if (intervals[:, 1] <= intervals[:, 0]).any(): return '', '', '', 'unresolved'
    lo = int(intervals[:, 0].min()); hi = int(intervals[:, 1].max())
    return data['raw'][i][lo:hi], lo, hi, 'verified'


def load_evidence_map(path):
    if not path: return {}
    from q3_common import csv_read
    result = {}
    for r in csv_read(path):
        key = (r['sample_id'], r['modality'], int(r['feature_index']))
        if key in result: raise ValueError('Duplicate evidence mapping')
        if r['status'] not in ('verified', 'approximate', 'unresolved', 'unavailable'): raise ValueError('Invalid mapping status')
        if r['status'] in ('verified', 'approximate'):
            if not r.get('source') or not 0 <= float(r['start_seconds']) < float(r['end_seconds']):
                raise ValueError('Mapping needs source and valid timestamps')
        if r['status'] == 'verified' and (not r.get('mapping_rule') or not r.get('reviewer')):
            raise ValueError('Verified mapping needs rule and reviewer')
        result[key] = r
    return result


def local_evidence(pred, data, i, target, contrast, original_l, original_r, out, evidence_map):
    original_margin = float(original_l[target] - original_l[contrast])
    tasks = []; masks = []
    for m in range(3):
        groups = units(data, i, m)
        for width in (1, 3, 5):
            if width > len(groups): continue
            for start in range(len(groups)-width+1):
                positions = sum(groups[start:start+width], [])
                if m > 0 and len(positions) > 1 and np.any(np.diff(positions) != 1):
                    continue  # An unobserved internal row breaks a continuous A/V window.
                mask = data['mask'][i].copy(); mask[positions, m] = False
                tasks.append((m, start, width, positions)); masks.append(mask)
    if not masks: return [], [], []
    logits, regressions, prob = pred.masks(data, i, masks)
    records = []; singletons = {}
    for j, (m, start, width, positions) in enumerate(tasks):
        effect = original_margin - float(logits[j, target]-logits[j, contrast])
        snippet, char_lo, char_hi, text_status = text_span(data, i, positions) if m == 0 else ('', '', '', 'unresolved')
        mappings = [evidence_map.get((data['ids'][i], MODS[m], int(t))) for t in positions]
        timed = all(r and r['status'] in ('verified', 'approximate') for r in mappings)
        time_status = ('verified' if all(r['status'] == 'verified' for r in mappings) else 'approximate') if timed else 'unresolved'
        records.append({'sample_id': data['ids'][i], 'modality': MODS[m], 'window_units': width,
                        'unit_start': start, 'feature_indices': json.dumps(positions), 'text': snippet,
                        'char_start': char_lo, 'char_end_exclusive': char_hi,
                        'text_mapping_status': text_status if m == 0 else 'not_applicable',
                        'start_seconds': min(float(r['start_seconds']) for r in mappings) if timed else '',
                        'end_seconds': max(float(r['end_seconds']) for r in mappings) if timed else '',
                        'time_mapping_status': time_status, 'target_class': LABELS[target], 'contrast_class': LABELS[contrast],
                        'class_margin_effect': effect, 'intensity_effect': float(original_r - regressions[j]),
                        'after_target_probability': float(prob[j, target]), 'selected': False, 'direction': 'support' if effect >= 0 else 'oppose'})
        if width == 1: singletons[(m, start)] = effect
    selected = []
    for m in range(3):
        for direction in ('support', 'oppose'):
            candidates = [r for r in records if r['modality'] == MODS[m] and r['direction'] == direction]
            candidates.sort(key=lambda r: abs(r['class_margin_effect']), reverse=True)
            taken = []; count = 0
            for record in candidates:
                positions = set(json.loads(record['feature_indices']))
                if any(len(positions & s) / max(1, len(positions | s)) > .3 for s in taken): continue
                if abs(record['class_margin_effect']) < 1e-8: continue
                record['selected'] = True; selected.append(record); taken.append(positions); count += 1
                if count == 3: break
    # Fixed singleton ranking tested using union deletions and a matched random control.
    rng = np.random.default_rng(2026 + i)
    f_tasks = []; f_masks = []
    for m in range(3):
        groups = units(data, i, m)
        if not groups: continue
        order = sorted(range(len(groups)), key=lambda j: -singletons[(m, j)])
        for rate in (.1, .2, .3):
            k = max(1, math.ceil(len(groups) * rate))
            candidates = [('top', order[:k]), ('bottom', order[-k:])]
            candidates += [('random', rng.choice(len(groups), k, replace=False).tolist()) for _ in range(20)]
            candidates += [('retain_top', order[:k])]
            for kind, chosen in candidates:
                mask = data['mask'][i].copy()
                if kind == 'retain_top':
                    # Sufficiency within this modality, others are held fixed.
                    mask[:, m] = False
                    mask[sum([groups[j] for j in chosen], []), m] = True
                else: mask[sum([groups[j] for j in chosen], []), m] = False
                f_tasks.append((m, rate, k, kind)); f_masks.append(mask)
    fl, fr, fp = pred.masks(data, i, f_masks)
    faithful = []
    from scipy.special import softmax
    original_prob = float(softmax(original_l)[target])
    for j, (m, rate, k, kind) in enumerate(f_tasks):
        faithful.append({'sample_id': data['ids'][i], 'modality': MODS[m], 'rate': rate, 'units': k,
                         'intervention': kind, 'probability_drop': original_prob - float(fp[j, target]),
                         'margin_drop': original_margin - float(fl[j, target]-fl[j, contrast]),
                         'intensity_change': float(original_r-fr[j])})
    return records, selected, faithful


def explain_split(args, pred, data, split, local_indices, evidence_map):
    out = args.out / ('附件4预测' if split == '附件4' else '验证集解释'); out.mkdir(parents=True, exist_ok=True)
    rows = []; coalitions = []; local_rows = []; selected_rows = []; faith = []; err = []
    for i, sid in enumerate(data['ids']):
        masks = []
        for s in range(8):
            mask = data['mask'][i].copy()
            for m in range(3):
                if not s & (1 << m): mask[:, m] = False
            masks.append(mask)
        l, r, p = pred.masks(data, i, masks)
        target = int(p[7].argmax()); contrast = int(np.argsort(p[7])[-2])
        values = np.stack([l[:, target] - l[:, contrast], r], -1).astype(np.float64)
        phi = exact_shapley(values)
        residual = float(np.abs(phi.sum(0) - (values[7]-values[0])).max()); err.append(residual)
        if residual > 1e-5: raise AssertionError('Shapley completeness failed')
        magnitude = np.abs(phi[:, 0]).sum(); reg_mass = np.abs(phi[:, 1]).sum()
        shares = np.abs(phi[:, 0]) / magnitude if magnitude > 1e-6 else np.zeros(3)
        row = {'sample_id': sid, 'polarity': LABELS[target], 'intensity': float(r[7]),
               'p_negative': float(p[7, 0]), 'p_neutral': float(p[7, 1]), 'p_positive': float(p[7, 2]),
               'target_class': LABELS[target], 'contrast_class': LABELS[contrast],
               'main_modality': MODS[int(shares.argmax())] if magnitude > 1e-6 else 'none',
               'intensity_main_modality': MODS[int(np.abs(phi[:, 1]).argmax())] if reg_mass > 1e-6 else 'none',
               'class_baseline': float(values[0, 0]), 'class_margin': float(values[7, 0]),
               'intensity_baseline': float(values[0, 1]), 'completeness_error': residual,
               'raw_text': data['raw'][i], 'text_mapping': bool(data['text_mapping'][i]),
               'sign_agreement': bool(target == np.sign(r[7]) + 1)}
        for m, mod in enumerate(MODS):
            row.update({f'{mod}_shapley_class': float(phi[m, 0]), f'{mod}_share': float(shares[m]),
                        f'{mod}_shapley_intensity': float(phi[m, 1]), f'{mod}_observed_rows': int(data['mask'][i, :, m].sum())})
        for s in range(8):
            coalitions.append({'sample_id': sid, 'coalition_bitmask': s, 'class_margin': float(values[s, 0]),
                               'intensity': float(r[s]), 'target_probability': float(p[s, target]),
                               'target_class': LABELS[target], 'contrast_class': LABELS[contrast]})
        if i in local_indices:
            ll, ss, ff = local_evidence(pred, data, i, target, contrast, l[7], r[7], out, evidence_map)
            local_rows += ll; selected_rows += ss; faith += ff
            # Different reference game: zero hidden features with observation masks held fixed.
            zl, zr, _ = pred.masks(data, i, masks, zero_feature=True)
            zphi = exact_shapley(np.stack([zl[:, target]-zl[:, contrast], zr], -1))
            zm = float(np.abs(zphi[:, 0]).sum())
            row['zero_feature_main_modality'] = MODS[int(np.abs(zphi[:, 0]).argmax())] if zm > 1e-6 else 'none'
            row['baseline_sensitive'] = row['zero_feature_main_modality'] != row['main_modality']
            for m, mod in enumerate(MODS): row[f'{mod}_zero_feature_shapley'] = float(zphi[m, 0])
        else:
            row.update(zero_feature_main_modality='', baseline_sensitive='')
            for mod in MODS: row[f'{mod}_zero_feature_shapley'] = ''
        rows.append(row)
        if i % 20 == 0: print(f'Explain {split} {i+1}/{len(data["ids"])}', flush=True)
    csv_write(out / '全量预测与解释.csv', rows); csv_write(out / '8子集输出.csv', coalitions)
    csv_write(out / '局部窗口全量.csv', local_rows); csv_write(out / '关键证据.csv', selected_rows)
    csv_write(out / '忠实度.csv', faith)
    dump(out / '解释审计.json', {'samples': len(rows), 'local_explanation_samples': len(local_indices),
         'local_sample_ids': [data['ids'][j] for j in sorted(local_indices)], 'max_completeness_error': max(err),
         'all_text_spans_available': all(data['text_mapping']),
         'av_seconds': 'only from externally verified evidence_map; otherwise unresolved',
         'baseline_sensitivity': 'missing-state versus zero-feature/mask-fixed; distinct intervention games',
         'passed_numerical': max(err) < 1e-5})
    return rows


def run_explain(args, encoder=None, valid=None):
    cp_path = args.out / '模型参数/best.pt'
    manifest = json.loads((args.out / '模型参数/manifest.json').read_text(encoding='utf-8'))
    if sha(cp_path) != manifest['sha256']: raise ValueError('Checkpoint checksum mismatch')
    encoder = encoder or FrozenText(args.bert, args.device)
    cp = torch.load(cp_path, map_location='cpu', weights_only=False)
    if cp['bert_fingerprint'] != encoder.fingerprint: raise ValueError('Changed text encoder')
    model = Q3Model(**cp['model_kwargs']).to(args.device); model.load_state_dict(cp['state'])
    pred = Predictor(model, encoder, args.device)
    evidence_map = load_evidence_map(args.evidence_map)
    if valid is None:
        from q3_common import adapt, read_pickle
        source = args.data_root / '附件2-数据集特征文件/aligned_50.pkl'
        training_audit = json.loads((args.out / '配置与审计/数据审计.json').read_text(encoding='utf-8'))
        source_hash = sha(source)
        if source_hash != training_audit['source_sha256']: raise ValueError('Official data changed since training')
        src = read_pickle(source)['valid']
        valid = adapt(src, encoder.tokenizer, True)
        if manifest['smoke']:
            valid = subset(valid, np.concatenate([np.flatnonzero(valid['y'] == c)[:4] for c in range(3)]))
        valid = normalize(valid, cp['stats'])
        cache_text(valid, encoder, args.cache, 'valid_smoke' if manifest['smoke'] else 'valid', source_hash)
    # Fixed class-stratified validation subset; does not choose by attractive explanations.
    local_indices = set()
    rng = np.random.default_rng(2026)
    for c in range(3):
        available = np.flatnonzero(valid['y'] == c)
        local_indices.update(map(int, rng.choice(available, min(len(available), math.ceil(args.explain_valid / 3)), replace=False)))
    explain_split(args, pred, valid, '验证集', local_indices, {})
    special, files = special_data(args.data_root, encoder.tokenizer)
    special = normalize(special, cp['stats'])
    special['t'] = encoder.encode(special['b'], special['mask'][:, :, 0])
    csv_write(args.out / '配置与审计/附件4输入审计.csv', special['audit'])
    dump(args.out / '配置与审计/附件4来源.json', [{'file': str(p.relative_to(args.data_root)), 'sha256': sha(p)} for p in files])
    explain_split(args, pred, special, '附件4', set(range(20)), evidence_map)
    csv_write(args.out / '附件4预测/视频映射.csv', [{'sample_id': s, 'video_relative': str(Path(p).relative_to(args.data_root)),
               'exists': Path(p).is_file()} for s, p in zip(special['ids'], special['videos'])])
