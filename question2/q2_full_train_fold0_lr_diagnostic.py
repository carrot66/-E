"""Train-only fold-0 diagnosis of learning rate and dropout for Q2 Fusion.

Read only the train entry of safe_train_valid.pkl. Fold 0 is an internal
validation group; the other four video-disjoint folds fit normalization,
class weights, teacher and student. The full valid and all test labels are
excluded from fitting, selection and reporting in this experiment.
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from q2_experiment import (
    ROOT, Fusion, TextEncoder, dump, fit_normalization, metrics, normalize,
    predict, prepare_split, runtime, scenarios, seed_all, sha, tensor_batch,
    write_csv,
)
from q2_group_cv_audit import sha256_id_order
from q2_scattered_gap_experiment import (
    augment_batch, get_banks, make_scattered_suite, profile,
)


SETTINGS = (
    {'name': 'current', 'learning_rate': 3e-4, 'dropout': .2},
    {'name': 'conservative', 'learning_rate': 1e-4, 'dropout': .3},
)


def choose_rows(split, indices):
    indices = np.asarray(indices, dtype=np.int64)
    result = {}
    for key, value in split.items():
        if key in ('ids', 'raw'):
            result[key] = [value[int(i)] for i in indices]
        else:
            result[key] = value[indices].copy()
    return result


def class_weights(split, device):
    count = np.bincount(split['y'], minlength=3)
    if (count == 0).any():
        raise ValueError('A class is absent from four-fold fit split')
    return torch.as_tensor(np.sqrt(len(split['y'])/(3*count)),
                           dtype=torch.float32, device=device)


def checkpoint(path, model, phase, setting, epoch, score, stats, provenance):
    torch.save({'model': model.state_dict(), 'phase': phase, 'setting': setting,
                'epoch': epoch, 'selection_score': float(score), 'stats': stats,
                'provenance': provenance}, path)


def train_teacher(setting, train, valid, quick, device, out, stats,
                  provenance, seed, max_epochs, patience):
    seed_all(seed)
    model = Fusion(gated=False, dropout=setting['dropout']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=setting['learning_rate'],
                                  weight_decay=1e-3)
    weights = class_weights(train, device)
    path = out / f"fold0_{setting['name']}_teacher.pt"
    history = []
    best, stale = -float('inf'), 0
    for epoch in range(1, max_epochs+1):
        model.train()
        order = np.random.permutation(len(train['ids']))
        losses = []
        for start in range(0, len(order), 64):
            idx = order[start:start+64]
            logits, reg = model(*tensor_batch(train, idx, device))
            y = torch.as_tensor(train['y'][idx], device=device)
            r = torch.as_tensor(train['r'][idx], device=device)
            loss = F.cross_entropy(logits, y, weight=weights) + .6*F.smooth_l1_loss(reg, r)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        prob, intensity = predict(model, valid, device, quick[0])
        clean = metrics(valid['y'], valid['r'], prob, intensity)
        value = clean['Macro_F1'] - .1*clean['MAE']
        row = {'setting':setting['name'], 'phase':'teacher', 'epoch':epoch,
               'train_loss':float(np.mean(losses)), 'clean_F1':clean['Macro_F1'],
               'clean_MAE':clean['MAE'], 'selection_score':value}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        write_csv(out / f"fold0_{setting['name']}_teacher_history.csv", history)
        if value > best + 1e-4:
            best, stale = value, 0
            checkpoint(path, model, 'teacher', setting, epoch, value, stats, provenance)
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)['model'])
    model.eval()
    return model, path, history


def train_student(setting, train, valid, quick, scattered, banks, teacher,
                  device, out, stats, provenance, seed, max_epochs, patience):
    # The same setting-specific seed gives a paired data order/augmentation
    # stream across the two hyperparameter candidates.
    seed_all(seed + 1)
    teacher_prob, teacher_reg = predict(teacher, train, device)
    teacher_eligible = (teacher_prob.argmax(1) == train['y']) & (
        np.abs(teacher_reg-train['r']) < 1.)
    model = Fusion(gated=True, dropout=setting['dropout']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=setting['learning_rate'],
                                  weight_decay=1e-3)
    weights = class_weights(train, device)
    path = out / f"fold0_{setting['name']}_student.pt"
    history = []
    best, stale = -float('inf'), 0
    blocks, sparse_banks = banks
    for epoch in range(1, max_epochs+1):
        model.train()
        order = np.random.permutation(len(train['ids']))
        losses = []
        for start in range(0, len(order), 64):
            idx = order[start:start+64]
            mask, text, changed = augment_batch(train, idx, blocks, sparse_banks)
            logits, reg = model(*tensor_batch(train, idx, device, text, mask))
            y = torch.as_tensor(train['y'][idx], device=device)
            r = torch.as_tensor(train['r'][idx], device=device)
            loss = F.cross_entropy(logits, y, weight=weights) + .6*F.smooth_l1_loss(reg, r)
            eligible = changed & teacher_eligible[idx]
            if eligible.any():
                selected = torch.as_tensor(eligible, device=device)
                target = torch.as_tensor(teacher_prob[idx][eligible], device=device)
                softened = F.softmax(torch.log(target.clamp_min(1e-8))/2, -1)
                loss += .2*4*F.kl_div(F.log_softmax(logits[selected]/2,-1),
                                       softened, reduction='batchmean')
                loss += .1*F.smooth_l1_loss(reg[selected],
                          torch.as_tensor(teacher_reg[idx][eligible],device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        result, _, _ = profile(setting['name'], model, valid, quick, scattered, device)
        row = {'setting':setting['name'], 'phase':'student', 'epoch':epoch,
               'train_loss':float(np.mean(losses)), **result}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        write_csv(out / f"fold0_{setting['name']}_student_history.csv", history)
        if result['selection_score'] > best + 1e-4:
            best, stale = result['selection_score'], 0
            checkpoint(path, model, 'student', setting, epoch, best, stats, provenance)
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)['model'])
    model.eval()
    return model, path, history


def main(args):
    runtime(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(args.safe_data)
    manifest_path = source.with_suffix('.manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    source_hash = sha(source)
    if manifest['safe_pickle_sha256'] != source_hash:
        raise ValueError('Safe train/valid pickle differs from preparation manifest')
    fold_path = Path(args.fold_manifest)
    folds = json.loads(fold_path.read_text(encoding='utf-8'))
    with source.open('rb') as stream:
        container = pickle.load(stream)
    if set(container) != {'train','valid'}:
        raise ValueError('The safe input contains unexpected split(s)')
    raw = container['train']
    del container
    gc.collect()
    ids = list(map(str, raw['id']))
    if folds['safe_train_sample_id_order_sha256'] != sha256_id_order(ids):
        raise ValueError('Five-fold manifest does not match safe train order')
    fold0 = np.asarray(folds['validation_indices_within_safe_train_by_fold']['0'],dtype=np.int64)
    if len(np.unique(fold0)) != len(fold0) or fold0.min() < 0 or fold0.max() >= len(ids):
        raise ValueError('Invalid fold-0 indices')
    fit = np.setdiff1d(np.arange(len(ids)), fold0)
    groups_fit = {ids[int(i)].split('$_$')[0] for i in fit}
    groups_val = {ids[int(i)].split('$_$')[0] for i in fold0}
    if groups_fit & groups_val or len(fold0) != 3265 or len(fit) != 13061:
        raise ValueError('Fold-0 video leakage or sample count mismatch')
    encoder = TextEncoder(args.device)
    cache = ROOT / 'work/问题2_特征缓存'
    prepared = prepare_split(raw, encoder, cache, 'safe_full_train', source_hash)
    del raw
    gc.collect()
    train = choose_rows(prepared, fit)
    valid = choose_rows(prepared, fold0)
    del prepared
    gc.collect()
    stats = fit_normalization(train)
    normalize(train, stats)
    normalize(valid, stats)
    quick = scenarios(valid, encoder, cache, stats, source_hash)
    sparse = make_scattered_suite(valid, encoder, cache, stats, source_hash)
    banks = get_banks(train, encoder, cache, stats, source_hash)
    provenance = {'safe_data_sha256':source_hash, 'safe_manifest_sha256':sha(manifest_path),
                  'fold_manifest_sha256':sha(fold_path), 'script_sha256':sha(__file__),
                  'fold':0, 'seed':args.seed, 'fit_samples':len(fit),
                  'internal_validation_samples':len(fold0),
                  'train_valid_video_overlap':0,
                  'normalization':'fit on four train folds only',
                  'full_valid_or_test_used':False,
                  'model':'Fusion teacher ungated + student gated; hidden96 one temporal Transformer',
                  'loss':'weighted CE +0.6 SmoothL1; student correct-teacher KD 0.2*4 KL(T=2)+0.1 SmoothL1',
                  'augmentation':'35% clean/30% contiguous 10,30,50%/35% scattered TAV 10,20,30%',
                  'teacher_selection':'fold0 clean MacroF1 -0.1 MAE',
                  'student_selection':'fold0 0.4 clean F1+0.3 contiguous F1+0.3 scattered F1-0.1 corresponding MAE',
                  'settings':SETTINGS,
                  'acceptance':'conservative student score >= current+0.005 and clean F1 >=current-0.01 and scattered F1 >=current-0.01',
                  'max_teacher_epochs':args.teacher_epochs,
                  'max_student_epochs':args.student_epochs,
                  'patience':args.patience, 'batch_size':64,
                  'python':sys.version.split()[0], 'torch':torch.__version__}
    dump(out / 'fold0_预注册.json', provenance)
    final = {}
    for setting in SETTINGS:
        teacher, teacher_path, teacher_history = train_teacher(setting,train,valid,
            quick,args.device,out,stats,provenance,args.seed,args.teacher_epochs,args.patience)
        student, student_path, student_history = train_student(setting,train,valid,
            quick,sparse,banks,teacher,args.device,out,stats,provenance,args.seed,
            args.student_epochs,args.patience)
        teacher_profile, _, _ = profile(setting['name']+'_teacher',teacher,valid,quick,sparse,args.device)
        student_profile, _, _ = profile(setting['name']+'_student',student,valid,quick,sparse,args.device)
        final[setting['name']] = {'learning_rate':setting['learning_rate'],
                                 'dropout':setting['dropout'],
                                 'teacher':teacher_profile, 'student':student_profile,
                                 'teacher_best_epoch':int(max(teacher_history,key=lambda x:x['selection_score'])['epoch']),
                                 'student_best_epoch':int(max(student_history,key=lambda x:x['selection_score'])['epoch']),
                                 'teacher_checkpoint_sha256':sha(teacher_path),
                                 'student_checkpoint_sha256':sha(student_path)}
        dump(out / 'fold0_当前比较.json',final)
    base = final['current']['student']
    other = final['conservative']['student']
    accepted = (other['selection_score'] >= base['selection_score'] + .005 and
                other['clean_F1'] >= base['clean_F1'] - .01 and
                other['scattered_TAV_mean_F1'] >= base['scattered_TAV_mean_F1'] - .01)
    dump(out / 'fold0_学习率与dropout结论.json',
         {'settings':final,'conservative_passed_predeclared_rule':bool(accepted),
          'interpretation_limit':'Single train-only video fold diagnoses optimization; full-valid remains untouched here.'})
    print(json.dumps({'current':base,'conservative':other,'conservative_passed':bool(accepted)},
                     ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--safe-data',default=ROOT/'work/full_mosei/safe_train_valid.pkl')
    parser.add_argument('--fold-manifest',default=ROOT/'outputs/question2/问题2_完整版MOSEI安全整合审计/问题2_完整版训练集视频分组五折清单.json')
    parser.add_argument('--out',default=ROOT/'outputs/question2/问题2_完整训练集fold0学习率诊断')
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--teacher-epochs',type=int,default=12)
    parser.add_argument('--student-epochs',type=int,default=16)
    parser.add_argument('--patience',type=int,default=5)
    main(parser.parse_args())
