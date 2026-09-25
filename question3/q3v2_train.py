"""Official-train-only training, checkpoint recovery and matched ablations."""
import copy
import itertools
import json
import os
import random
import shutil
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from q3v2_common import ROOT, adapt, read_pickle, fit_stats, normalize, dump, csv_write, sha, seed_all, scores, LABELS
from q3v2_model import FrozenText, Q3Model, tensor_inputs


def cache_text(data, encoder, cache, name, source_hash):
    cache.mkdir(parents=True, exist_ok=True)
    key = source_hash[:16] + '_' + encoder.fingerprint[:16]
    path = cache / f'{name}_{len(data["ids"])}_{key}.npy'
    if path.exists():
        text = np.load(path)
        if text.shape != (len(data['ids']), 50, 768) or not np.isfinite(text).all():
            raise ValueError('Corrupt BERT cache')
    else:
        text = encoder.encode(data['b'], data['mask'][:, :, 0])
        temp = path.with_suffix('.tmp.npy'); np.save(temp, text); temp.replace(path)
    data['t'] = text


def subset(data, indices):
    n = len(data['ids'])
    return {k: (v[indices] if isinstance(v, np.ndarray) and len(v) == n else
                [v[i] for i in indices] if isinstance(v, list) and len(v) == n else v)
            for k, v in data.items()}


def load_training(args, encoder):
    path = args.data_root / '附件2-数据集特征文件/aligned_50.pkl'
    data = read_pickle(path); data.pop('test', None)
    for split in ('train', 'valid'): data[split].pop('text', None)
    train = adapt(data['train'], encoder.tokenizer, True); valid = adapt(data['valid'], encoder.tokenizer, True)
    del data
    train_groups = {s.split('$_$')[0] for s in train['ids']}
    valid_groups = {s.split('$_$')[0] for s in valid['ids']}
    if train_groups & valid_groups: raise ValueError('Train/valid video overlap')
    if not args.smoke and (len(train['ids']), len(valid['ids'])) != (3395, 728):
        raise ValueError('Use official attachment-2 train/valid, not expanded MOSEI')
    mapping_fraction = float(np.mean(train['text_mapping']))
    if mapping_fraction < .95: raise ValueError(f'BERT tokenization mismatch on >5% of train: {mapping_fraction}')
    audit = {'train': len(train['ids']), 'valid': len(valid['ids']), 'video_overlap': 0,
             'bert_raw_token_match_fraction_train': mapping_fraction,
             'av_time_axis': 'independent rows; no unverified text offset imposed',
             'test': 'discarded without label access or scoring', 'source_sha256': sha(path),
             'smoke': args.smoke}
    csv_write(args.out / '配置与审计/输入逐样本审计.csv', train['audit'] + valid['audit'])
    if args.smoke:
        # Technical test only; includes all classes, never a competition result.
        train = subset(train, np.concatenate([np.flatnonzero(train['y'] == c)[:8] for c in range(3)]))
        valid = subset(valid, np.concatenate([np.flatnonzero(valid['y'] == c)[:4] for c in range(3)]))
    stats = fit_stats(train)
    train = normalize(train, stats); valid = normalize(valid, stats)
    source_hash = audit['source_sha256']
    cache_text(train, encoder, args.cache, 'train_smoke' if args.smoke else 'train', source_hash)
    cache_text(valid, encoder, args.cache, 'valid_smoke' if args.smoke else 'valid', source_hash)
    dump(args.out / '配置与审计/数据审计.json', audit)
    return train, valid, stats, source_hash


@torch.inference_mode()
def predict(model, data, device, batch=128):
    model.eval(); probs = []; reg = []
    for start in range(0, len(data['ids']), batch):
        idx = np.arange(start, min(start+batch, len(data['ids'])))
        logits, score, _ = model(*tensor_inputs(data, idx, device))
        probs.append(logits.softmax(-1).cpu().numpy()); reg.append(score.cpu().numpy())
    return np.concatenate(probs), np.concatenate(reg)


def augment(data, idx, rng, encoder):
    mask = data['mask'][idx].copy()
    for j in range(len(idx)):
        if rng.random() > .1: continue
        if rng.random() < .5:
            keep = int(rng.integers(1, 7))  # all six nonempty proper coalitions
            for m in range(3):
                if not keep & (1 << m): mask[j, :, m] = False
        else:
            m = int(rng.integers(3)); obs = np.flatnonzero(mask[j, :, m])
            if len(obs):
                start = int(rng.integers(len(obs))); chosen = obs[start:start+3]
                if m == 0 and data['text_mapping'][idx[j]]:
                    selected_words = {data['word_ids'][idx[j]][int(t)] for t in chosen}
                    chosen = [t for t in obs if data['word_ids'][idx[j]][int(t)] in selected_words]
                mask[j, chosen, m] = False
    text = data['t'][idx].copy()
    affected = np.flatnonzero(np.any(mask[:, :, 0] != data['mask'][idx, :, 0], axis=1))
    if len(affected): text[affected] = encoder.encode(data['b'][idx[affected]], mask[affected, :, 0])
    return text, mask


def better(current, best):
    # Exact fixed lexicographic rule; no drift from successively accepting ties.
    return best is None or (current['Macro_F1'], -current['MAE']) > (best['Macro_F1'], -best['MAE'])


def train_one(args, train, valid, stats, encoder, config, name):
    folder = args.out / '训练记录' / name; folder.mkdir(parents=True, exist_ok=True)
    done = folder / 'result.json'
    if done.exists(): return json.loads(done.read_text(encoding='utf-8'))
    seed_all(config['seed']); rng = np.random.default_rng(config['seed'])
    model_kwargs = {k: config[k] for k in ('hidden', 'dropout', 'mode')}
    model_kwargs.update(prior=(np.bincount(train['y'], minlength=3) / len(train['y'])).tolist(), mean_score=float(train['r'].mean()))
    model = Q3Model(**model_kwargs).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    history = []; start_epoch = 0; best = None; stale = 0
    last = folder / 'last.pt'; best_path = folder / 'best.pt'
    if last.exists():
        cp = torch.load(last, map_location='cpu', weights_only=False)
        if cp['config'] != config: raise ValueError('Resume config differs')
        model.load_state_dict(cp['state']); opt.load_state_dict(cp['optimizer'])
        for state in opt.state.values():
            for key, value in state.items():
                if torch.is_tensor(value): state[key] = value.to(args.device)
        history = cp['history']; best = cp['best']; stale = cp['stale']; start_epoch = cp['epoch'] + 1
        rng.bit_generator.state = cp['rng']; torch.set_rng_state(cp['torch_rng'])
        random.setstate(cp['python_rng']); np.random.set_state(cp['numpy_rng'])
        if torch.cuda.is_available() and cp['cuda_rng'] is not None: torch.cuda.set_rng_state_all(cp['cuda_rng'])
    for epoch in range(start_epoch, args.epochs):
        if stale >= args.patience: break
        model.train(); losses = []
        for idx in np.array_split(rng.permutation(len(train['ids'])), int(np.ceil(len(train['ids']) / args.batch_size))):
            y = torch.as_tensor(train['y'][idx], device=args.device)
            r = torch.as_tensor(train['r'][idx], device=args.device)
            logits, score, _ = model(*tensor_inputs(train, idx, args.device))
            loss = F.cross_entropy(logits, y) + .6 * F.smooth_l1_loss(score, r)
            if config['augmentation']:
                text, mask = augment(train, idx, rng, encoder)
                affected = np.any(mask != train['mask'][idx], axis=(1, 2))
                if affected.any():
                    al, ar, _ = model(*tensor_inputs(train, idx[affected], args.device, text[affected], mask[affected]))
                    am = torch.as_tensor(affected, device=args.device)
                    loss = loss + .2 * (F.cross_entropy(al, y[am]) + .6 * F.smooth_l1_loss(ar, r[am]))
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(float(loss.detach()))
        prob, reg = predict(model, valid, args.device)
        result = scores(valid['y'], valid['r'], prob, reg)
        history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)), **{k: v for k, v in result.items() if k != 'confusion_matrix'}})
        if better(result, best):
            best = result; stale = 0
            torch.save({'state': model.state_dict(), 'model_kwargs': model_kwargs, 'stats': stats,
                        'config': config, 'epoch': epoch+1, 'metrics': result,
                        'bert_fingerprint': encoder.fingerprint}, best_path.with_suffix('.tmp'))
            best_path.with_suffix('.tmp').replace(best_path)
        else: stale += 1
        torch.save({'state': model.state_dict(), 'optimizer': opt.state_dict(), 'history': history, 'best': best,
                    'config': config, 'stale': stale, 'epoch': epoch, 'rng': rng.bit_generator.state,
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()}, last.with_suffix('.tmp'))
        last.with_suffix('.tmp').replace(last)
        csv_write(folder / 'history.csv', history)
        print(f'{name} epoch={epoch+1} F1={result["Macro_F1"]:.4f} Acc={result["Accuracy"]:.4f} MAE={result["MAE"]:.4f}', flush=True)
    cp = torch.load(best_path, map_location='cpu', weights_only=False)
    model.load_state_dict(cp['state']); prob, reg = predict(model, valid, args.device)
    result = {'name': name, 'config': config, 'checkpoint': str(best_path.relative_to(args.out)),
              'best_epoch': cp['epoch'], **scores(valid['y'], valid['r'], prob, reg)}
    dump(done, result)
    return result


def run_train(args, encoder=None):
    encoder = encoder or FrozenText(args.bert, args.device)
    # Fail before overwriting audit files for a differently configured run.
    prior_path = args.out / '配置与审计/训练配置.json'
    if prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding='utf-8'))
        requested = {'epochs': args.epochs, 'patience': args.patience, 'batch_size': args.batch_size,
                     'smoke': args.smoke, 'search': args.search, 'seeds': args.seeds,
                     'ablations': not args.no_ablations, 'bert': encoder.fingerprint,
                     'data_sha256': sha(args.data_root / '附件2-数据集特征文件/aligned_50.pkl'),
                     'code': {p.name: sha(p) for p in Path(__file__).parent.glob('q3v2_*.py')}}
        if prior != requested: raise ValueError('Run configuration/source changed; use a new Q3_RUN')
    train, valid, stats, source_hash = load_training(args, encoder)
    cfg = {'hidden': 96, 'dropout': .2, 'lr': 3e-4, 'mode': 'gated', 'augmentation': True, 'seed': 2026}
    provenance = {'data_sha256': source_hash, 'bert': encoder.fingerprint, 'epochs': args.epochs,
                  'patience': args.patience, 'batch_size': args.batch_size, 'smoke': args.smoke, 'search': args.search,
                  'seeds': args.seeds, 'ablations': not args.no_ablations,
                  'code': {p.name: sha(p) for p in Path(__file__).parent.glob('q3v2_*.py')}}
    path = args.out / '配置与审计/训练配置.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != provenance:
        raise ValueError('Existing run provenance differs; choose a new --out / Q3_RUN')
    dump(path, provenance)
    all_results = []
    if args.search:
        for hidden, dropout, lr in itertools.product((64, 96), (.2, .4), (3e-4, 1e-3)):
            cc = dict(cfg, hidden=hidden, dropout=dropout, lr=lr)
            all_results.append(train_one(args, train, valid, stats, encoder, cc, f'search_h{hidden}_d{dropout}_lr{lr}'))
        cfg = max(all_results, key=lambda r: (r['Macro_F1'], -r['MAE']))['config']
    primary = []
    for seed in args.seeds:
        result = train_one(args, train, valid, stats, encoder, dict(cfg, seed=seed), f'main_seed{seed}')
        all_results.append(result); primary.append(result)
    if not args.no_ablations:
        for mode in ('text_only', 'equal', 'no_temporal', 'no_augmentation'):
            cc = dict(cfg, seed=args.seeds[0], mode='gated' if mode == 'no_augmentation' else mode,
                      augmentation=mode != 'no_augmentation')
            all_results.append(train_one(args, train, valid, stats, encoder, cc, mode))
    # First predeclared seed is the final model; do not cherry-pick best random seed.
    selected = primary[0]
    final = args.out / '模型参数/best.pt'; final.parent.mkdir(exist_ok=True)
    shutil.copyfile(args.out / selected['checkpoint'], final)
    cp = torch.load(final, map_location='cpu', weights_only=False)
    model = Q3Model(**cp['model_kwargs']).to(args.device); model.load_state_dict(cp['state'])
    prob, reg = predict(model, valid, args.device)
    rows = []
    for i, sid in enumerate(valid['ids']):
        rows.append({'sample_id': sid, 'true_class': int(valid['y'][i]), 'predicted_class': int(prob[i].argmax()),
                     'true_intensity': float(valid['r'][i]), 'intensity': float(reg[i]),
                     'p_negative': float(prob[i, 0]), 'p_neutral': float(prob[i, 1]), 'p_positive': float(prob[i, 2]),
                     'confidence': float(prob[i].max()), 'absolute_error': float(abs(valid['r'][i]-reg[i])),
                     'raw_text': valid['raw'][i], 'truncated': bool(valid['b'][i, 1].sum() == 50),
                     'audio_rows': int(valid['mask'][i, :, 1].sum()), 'vision_rows': int(valid['mask'][i, :, 2].sum())})
    csv_write(args.out / '验证集评价/全量预测.csv', rows)
    csv_write(args.out / '验证集评价/模型比较.csv', [{k: v for k, v in r.items() if k not in ('config', 'confusion_matrix')} for r in all_results])
    dump(args.out / '验证集评价/评价.json', scores(valid['y'], valid['r'], prob, reg))
    dump(args.out / '验证集评价/重复种子统计.json', {k: {'mean': float(np.mean([r[k] for r in primary])),
          'std': float(np.std([r[k] for r in primary]))} for k in ('Accuracy', 'Macro_F1', 'MAE')})
    dump(args.out / '模型参数/manifest.json', {'sha256': sha(final), 'selected': selected,
         'selection': 'validation Macro-F1, tie by MAE; first declared seed for final inference',
         'smoke': args.smoke})
    return encoder, valid
