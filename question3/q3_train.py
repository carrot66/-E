"""Official Q3 experiments with conservative and token-interaction candidates."""
import copy
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from q3_common import (read_pickle, adapt, fit_stats, normalize, scores, seed_all,
                         csv_write, dump, sha)
from q3_base_model import FrozenText
from q3_train_utils import subset, better
from q3_adaptive_model import model_inputs
from q3_model import fresh_model, candidates as candidate_configs
from q3_decision import adjust, select as select_bias, crossfit


def load_data(args, tokenizer):
    file = args.data_root / '附件2-数据集特征文件/aligned_50.pkl'
    source = read_pickle(file); source.pop('test', None)
    for k in ('train', 'valid'): source[k].pop('text', None)
    train = adapt(source['train'], tokenizer, True); valid = adapt(source['valid'], tokenizer, True)
    del source
    if (len(train['ids']), len(valid['ids'])) != (3395, 728): raise ValueError('Official 3395/728 split required')
    train_groups = {s.split('$_$')[0] for s in train['ids']}
    valid_groups = {s.split('$_$')[0] for s in valid['ids']}
    if train_groups & valid_groups: raise ValueError('Video-group leakage')
    if np.mean(train['text_mapping']) < .95: raise ValueError('Tokenizer does not match official IDs')
    if args.smoke:
        train = subset(train, np.concatenate([np.flatnonzero(train['y'] == c)[:4] for c in range(3)]))
        valid = subset(valid, np.concatenate([np.flatnonzero(valid['y'] == c)[:2] for c in range(3)]))
    stats = fit_stats(train)
    return normalize(train, stats), normalize(valid, stats), stats, sha(file)


@torch.inference_mode()
def predict(model, data, device, batch=32):
    model.eval(); pp = []; rr = []
    for start in range(0, len(data['ids']), batch):
        idx = np.arange(start, min(start+batch, len(data['ids'])))
        logits, reg = model(*model_inputs(data, idx, device))
        pp.append(logits.softmax(-1).cpu().numpy()); rr.append(reg.cpu().numpy())
    return np.concatenate(pp), np.concatenate(rr)


def make_masks(data, idx, rng):
    mask = data['mask'][idx].copy()
    for j, sample in enumerate(idx):
        if rng.random() >= .1: continue
        if rng.random() < .5:
            keep = int(rng.integers(1, 7))
            for m in range(3):
                if not keep & (1 << m): mask[j, :, m] = False
        else:
            m = int(rng.integers(3)); observed = np.flatnonzero(mask[j, :, m])
            if not len(observed): continue
            first = int(rng.choice(observed)); hidden = np.arange(first, min(first+3, 50))
            if m == 0 and data['text_mapping'][sample]:
                words = {data['word_ids'][sample][int(t)] for t in hidden if mask[j, t, m]}
                hidden = [t for t in observed if data['word_ids'][sample][int(t)] in words]
            mask[j, hidden, m] = False
    return mask


def weighted_loss(logits, y, weights, smoothing=.03):
    return F.cross_entropy(logits, y, weight=weights, label_smoothing=smoothing)


def save_atomic(path, obj):
    temp = path.with_suffix('.tmp'); torch.save(obj, temp); temp.replace(path)


def train_one(args, train, valid, stats, cfg, name, fingerprint):
    folder = args.out / '训练记录' / name; folder.mkdir(parents=True, exist_ok=True)
    done = folder / 'result.json'; best_file = folder / 'best.pt'; last_file = folder / 'last.pt'
    if done.exists():
        result = json.loads(done.read_text(encoding='utf-8'))
        if result['config'] != cfg or sha(best_file) != result['sha256']: raise ValueError('Completed run changed')
        saved_predictions = folder / 'validation_predictions.npz'
        if sha(saved_predictions) != result['prediction_sha256']: raise ValueError('Cached predictions changed')
        with np.load(saved_predictions) as pred:
            if list(pred['ids']) != valid['ids']: raise ValueError('Cached validation order changed')
            print(f'{name}: verified completed checkpoint; skipping training', flush=True)
            return result, pred['probabilities'], pred['regression']
    seed_all(cfg['seed']); rng = np.random.default_rng(cfg['seed'])
    counts = np.bincount(train['y'], minlength=3)
    prior = (counts / counts.sum()).tolist(); mean = float(train['r'].mean())
    model, _ = fresh_model(args.bert, cfg, prior, mean, args.device, verify=False)
    adapter = [p for n,p in model.named_parameters() if p.requires_grad and n.startswith('bert.')]
    heads = [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('bert.')]
    groups = [{'params': heads, 'lr': cfg['head_lr']}]
    if adapter: groups.append({'params': adapter, 'lr': cfg['adapter_lr']})
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    steps_per_epoch = math.ceil(len(train['ids']) / args.batch_size)
    total = steps_per_epoch * args.epochs; warmup = max(1, round(total * .1))
    def multiplier(step):
        if step < warmup: return (step + 1) / warmup
        return .1 + .9 * .5 * (1 + math.cos(math.pi * min(1., (step-warmup) / max(1, total-warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    weights = torch.tensor((counts.sum() / (3*counts)) ** cfg["weight_power"], dtype=torch.float32, device=args.device)
    weights /= weights.mean()
    history = []; best = None; stale = 0; start_epoch = 0
    if last_file.exists():
        checkpoint = torch.load(last_file, map_location='cpu', weights_only=False)
        if checkpoint['config'] != cfg or checkpoint['bert_fingerprint'] != fingerprint: raise ValueError('Resume inputs changed')
        model.load_portable(checkpoint['state']); optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v): state[k] = v.to(args.device)
        scheduler.load_state_dict(checkpoint['scheduler'])
        history = checkpoint['history']; best = checkpoint['best']; stale = checkpoint['stale']; start_epoch = checkpoint['epoch']+1
        rng.bit_generator.state = checkpoint['rng']; torch.set_rng_state(checkpoint['torch_rng'])
        random.setstate(checkpoint['python_rng']); np.random.set_state(checkpoint['numpy_rng'])
        if torch.cuda.is_available() and checkpoint['cuda_rng'] is not None: torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    for epoch in range(start_epoch, args.epochs):
        if stale >= args.patience: break
        model.train(); losses = []; ce_values = []; reg_values = []
        batches = np.array_split(rng.permutation(len(train['ids'])), steps_per_epoch)
        for idx in batches:
            y = torch.as_tensor(train['y'][idx], device=args.device)
            r = torch.as_tensor(train['r'][idx], device=args.device)
            logits, regression, tl, tr, al, ar = model(*model_inputs(train, idx, args.device), return_aux=True)
            ce = weighted_loss(logits, y, weights, cfg["smoothing"]); regloss = F.smooth_l1_loss(regression, r)
            loss = ce + cfg["reg_weight"]*regloss + cfg["text_ce"]*weighted_loss(tl, y, weights, cfg["smoothing"]) + cfg["text_reg"]*F.smooth_l1_loss(tr, r)
            has_av = torch.as_tensor(train['mask'][idx, :, 1:].any((1,2)), device=args.device)
            if has_av.any(): loss += cfg["av_aux"]*(weighted_loss(al[has_av], y[has_av], weights, cfg["smoothing"]) + cfg["reg_weight"]*F.smooth_l1_loss(ar[has_av], r[has_av]))
            altered = make_masks(train, idx, rng)
            affected = np.any(altered != train['mask'][idx], axis=(1,2))
            if affected.any():
                il, ir = model(*model_inputs(train, idx[affected], args.device, altered[affected]))
                iy, it = y[torch.as_tensor(affected, device=args.device)], r[torch.as_tensor(affected, device=args.device)]
                loss += cfg["augmentation"]*(weighted_loss(il, iy, weights, cfg["smoothing"]) + cfg["reg_weight"]*F.smooth_l1_loss(ir, it))
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            optimizer.step(); scheduler.step()
            losses.append(float(loss.detach())); ce_values.append(float(ce.detach())); reg_values.append(float(regloss.detach()))
        prob, regression = predict(model, valid, args.device, args.batch_size)
        metrics = scores(valid['y'], valid['r'], prob, regression)
        from sklearn.metrics import f1_score
        per_class = f1_score(valid['y'], prob.argmax(1), labels=[0,1,2], average=None, zero_division=0)
        history.append({'epoch': epoch+1, 'loss': float(np.mean(losses)), 'classification_loss': float(np.mean(ce_values)),
                        'regression_loss': float(np.mean(reg_values)), 'head_lr': optimizer.param_groups[0]['lr'],
                        **{name+'_F1': float(value) for name,value in zip(('Negative','Neutral','Positive'),per_class)},
                        **{k:v for k,v in metrics.items() if k != 'confusion_matrix'}})
        core = {'state': model.portable_state(), 'config': cfg, 'stats': stats, 'prior': prior,
                'mean_score': mean, 'bert_fingerprint': fingerprint, 'epoch': epoch+1, 'metrics': metrics}
        if better(metrics, best):
            best = metrics; stale = 0; save_atomic(best_file, core)
        else: stale += 1
        save_atomic(last_file, {**core, 'epoch': epoch, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'history': history, 'best': best, 'stale': stale, 'rng': rng.bit_generator.state,
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()})
        csv_write(folder / 'history.csv', history)
        print(f'{name} epoch={epoch+1} Macro_F1={metrics["Macro_F1"]:.4f} Neutral_F1={per_class[1]:.4f} Acc={metrics["Accuracy"]:.4f} MAE={metrics["MAE"]:.4f} train_loss={history[-1]["loss"]:.4f}', flush=True)
    checkpoint = torch.load(best_file, map_location='cpu', weights_only=False)
    model.load_portable(checkpoint['state'])
    prob, regression = predict(model, valid, args.device, args.batch_size)
    result = {'name': name, 'config': cfg, 'best_epoch': checkpoint['epoch'], 'checkpoint': best_file.relative_to(args.out).as_posix(),
              'sha256': sha(best_file), **scores(valid['y'], valid['r'], prob, regression)}
    np.savez_compressed(folder / 'validation_predictions.npz', ids=np.array(valid['ids']), probabilities=prob, regression=regression)
    result['prediction_sha256'] = sha(folder / 'validation_predictions.npz')
    dump(done, result)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result, prob, regression


def train(args):
    encoder = FrozenText(args.bert, 'cpu'); fingerprint = encoder.fingerprint; tokenizer = encoder.tokenizer; del encoder
    source = args.data_root / '附件2-数据集特征文件/aligned_50.pkl'
    provenance = {'data_sha256': sha(source), 'bert': fingerprint, 'epochs': args.epochs, 'patience': args.patience,
                  'batch_size': args.batch_size, 'smoke': args.smoke, 'seeds': args.seeds, 'target_macro_f1': .65,
                  'candidates': args.candidates,
                   'code': {p.name: sha(p) for p in Path(__file__).parent.glob('q3_*.py')}}
    config_path = args.out / '配置与审计/训练配置.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != provenance:
        raise ValueError('Changed run configuration/code: use a new Q3_RUN')
    dump(config_path, provenance)
    import platform
    import transformers
    dump(args.out / '配置与审计/运行环境.json', {'python': platform.python_version(), 'torch': torch.__version__,
        'transformers': transformers.__version__, 'device': args.device,
        'gpu': torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None})
    train_data, valid, stats, source_hash = load_data(args, tokenizer)
    dump(args.out / '配置与审计/数据审计.json', {'train': len(train_data['ids']), 'valid': len(valid['ids']),
        'source_sha256': source_hash, 'train_valid_video_overlap': 0, 'special_test_used_for_selection': False,
        'test': 'not scored or used for selection', 'smoke': args.smoke})
    csv_write(args.out / '配置与审计/输入逐样本审计.csv', train_data['audit']+valid['audit'])
    configs = candidate_configs(args.seeds[0])
    candidates = []; prediction_sets = {}
    for name in args.candidates:
        result, p, r = train_one(args, train_data, valid, stats, configs[name], name, fingerprint)
        candidates.append(result); prediction_sets[name] = (p,r)
    winner = max(candidates, key=lambda r: (r['Macro_F1'], -r['MAE']))
    repeated = [winner]; pp = [prediction_sets[winner['name']][0]]; rr = [prediction_sets[winner['name']][1]]
    for seed in args.seeds[1:]:
        cfg = dict(winner['config'], seed=seed)
        result, p, r = train_one(args, train_data, valid, stats, cfg, winner['name']+f'_seed{seed}', fingerprint)
        repeated.append(result); pp.append(p); rr.append(r)
    pool=[(result['name'],[result],prediction_sets[result['name']][0],prediction_sets[result['name']][1]) for result in candidates]
    if len(repeated)>1:
        pool.append(('winner_seed_mean',repeated,np.mean(pp,0),np.mean(rr,0)))
    ordered=sorted(candidates,key=lambda x:(x['Macro_F1'],-x['MAE']),reverse=True)
    if len(ordered)>1:
        models=ordered[:2]
        pool.append(('top_two_structure_mean',models,
            np.mean([prediction_sets[m['name']][0] for m in models],0),
            np.mean([prediction_sets[m['name']][1] for m in models],0)))
    deployment=[]
    for name,models,p,r in pool:
        bias=select_bias(p,valid['y'])
        raw=scores(valid['y'],valid['r'],p,r)
        check=crossfit(p,valid['y'],valid['ids']) if not args.smoke else None
        # Nonzero correction deploys only if group cross-fit improves over zero bias.
        accepted=check is not None and check['crossfit_macro_f1']>=raw['Macro_F1']+.002
        used=bias if accepted else np.zeros(3)
        selected_prob=adjust(p,used); m=scores(valid['y'],valid['r'],selected_prob,r)
        deployment.append({'name':name,'models':[x['name'] for x in models],
            'bias':used.tolist(),'fitted_bias':bias.tolist(),'bias_accepted':accepted,'raw_metrics':raw,
            'metrics':m,'crossfit':check})
    best_idx=max(range(len(pool)),key=lambda i:(deployment[i]['metrics']['Macro_F1'],-deployment[i]['metrics']['MAE']))
    selected_rule=deployment[best_idx]
    _,selected_models,raw_prob,final_reg=pool[best_idx]
    final_prob=adjust(raw_prob,selected_rule['bias'])
    use_ensemble=len(selected_models)>1
    components = []
    final_folder = args.out / '模型参数'; final_folder.mkdir(exist_ok=True)
    import shutil
    for j, result in enumerate(selected_models):
        destination = final_folder / f'component_{j}.pt'
        shutil.copyfile(args.out / result['checkpoint'], destination)
        components.append({'path': destination.relative_to(args.out).as_posix(), 'sha256': sha(destination), 'weight': 1/len(selected_models)})
    final_metrics = scores(valid['y'], valid['r'], final_prob, final_reg)
    manifest = {'version': 5, 'components': components, 'selected': selected_models[0], 'ensemble': use_ensemble, 'class_bias': selected_rule['bias'], 'deployment_rule': selected_rule['name'],
                'smoke': args.smoke, 'bert_fingerprint': fingerprint, 'metrics': final_metrics,
                'selection': 'official-valid: declared structures, winner seed mean, top-two structure mean; group-crossfit-guarded 81-point class-bias grid; not independent test performance'}
    dump(final_folder / 'manifest.json', manifest)
    save_predictions(args.out, valid, final_prob, final_reg)
    all_rows = candidates + repeated[1:]
    csv_write(args.out / '验证集评价/模型比较.csv', [{k:v for k,v in r.items() if k not in ('config','confusion_matrix')} for r in all_rows])
    dump(args.out / '验证集评价/部署规则比较.json', {'rules':deployment,'selected':selected_rule['name']})
    dump(args.out / '验证集评价/重复种子统计.json', {k:{'mean': float(np.mean([r[k] for r in repeated])),
         'std': float(np.std([r[k] for r in repeated]))} for k in ('Accuracy','Macro_F1','MAE')})
    dump(args.out / '验证集评价/评价.json', final_metrics)
    dump(args.out / '目标检查.json', {'metric': 'Macro_F1', 'threshold': .65, 'value': final_metrics['Macro_F1'],
        'passed': not args.smoke and final_metrics['Macro_F1'] >= .65, 'smoke': args.smoke,
        'scope': 'official validation used for model selection; no guaranteed future/test score'})
    print(f'Final validation Macro-F1={final_metrics["Macro_F1"]:.6f}; target 0.65 reached={not args.smoke and final_metrics["Macro_F1"]>=.65}', flush=True)
    return valid


def save_predictions(out, data, probabilities, regressions):
    rows=[]
    for i,sid in enumerate(data['ids']):
        p=probabilities[i]; r=float(regressions[i])
        rows.append({'sample_id': sid, 'true_class': int(data['y'][i]), 'predicted_class': int(p.argmax()),
            'true_intensity': float(data['r'][i]), 'intensity': r, 'p_negative': float(p[0]), 'p_neutral': float(p[1]),
            'p_positive': float(p[2]), 'confidence': float(p.max()), 'absolute_error': abs(r-float(data['r'][i])),
            'raw_text': data['raw'][i], 'truncated': bool(data['b'][i,1].sum()==50),
            'audio_rows': int(data['mask'][i,:,1].sum()), 'vision_rows': int(data['mask'][i,:,2].sum())})
    csv_write(out / '验证集评价/全量预测.csv', rows)
