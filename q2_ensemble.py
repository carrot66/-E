"""问题2优化版：三随机种子集成，验证集选型，留出测试集只做一次复核。"""
from __future__ import annotations
import argparse
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from q2_experiment import (ROOT, METHODS, Fusion, HierarchicalFusion, TextEncoder, dump,
                           metrics, normalize, prepare_split, predict, runtime,
                           save_predictions, scenarios, seed_all, sha, write_csv)


def load_models(paths, device):
    models = {}
    ckpts = {}
    for name, path in paths.items():
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = (HierarchicalFusion(**ckpt['architecture']) if ckpt['method']=='hierarchical'
                 else Fusion(gated=ckpt['method'] in ('gated', 'distilled'), **ckpt['architecture'])).to(device)
        model.load_state_dict(ckpt['model']); model.eval()
        models[name] = model; ckpts[name] = ckpt
    return models, ckpts


def predict_ensemble(models, member_names, split, device, scenario):
    predictions = [predict(models[name], split, device, scenario) for name in member_names]
    prob = np.mean(np.stack([p[0] for p in predictions]), axis=0)
    reg = np.mean(np.stack([p[1] for p in predictions]), axis=0)
    # 题目规定：Neutral <=> 强度0；Negative <0；Positive >0。
    # This deterministic output rule uses no labels or tuned threshold.
    cls=prob.argmax(1)
    reg[cls==1]=0.
    reg[cls==0]=np.minimum(reg[cls==0],-.01)
    reg[cls==2]=np.maximum(reg[cls==2],.01)
    return prob, reg


def summarize(model_names, models, split, suite, device):
    rows = []; predictions = []
    for sc in suite:
        p, pr = predict_ensemble(models, model_names, split, device, sc)
        row = {k:sc[k] for k in ('subset','rate','position','replicate')}
        row.update(metrics(split['y'], split['r'], p, pr))
        for j, m in enumerate('TAV'):
            row[f'{m}_实际不可用比例'] = float((split['valid'].sum()-sc['mask'][:,:,j].sum())/split['valid'].sum())
        rows.append(row); predictions.append((p,pr))
    return rows,predictions


def selection_score(rows):
    clean=rows[0]; missing=rows[1:]
    f1=float(np.mean([x['Macro_F1'] for x in missing]))
    mae=float(np.mean([x['MAE'] for x in missing]))
    return .5*(clean['Macro_F1']+f1)-.05*(clean['MAE']+mae)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root', default=str(ROOT/'E题数据/E题数据'))
    parser.add_argument('--out', default=str(ROOT/'outputs/问题2_优化实验结果'))
    parser.add_argument('--device', default='cuda:1')
    args=parser.parse_args(); runtime(args.device); seed_all(2026)
    root=Path(args.data_root); out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    source=root/'附件2-数据集特征文件/aligned_50.pkl'; source_hash=sha(source)
    originals={2026:ROOT/'outputs/问题2_首轮实验结果',
               2027:ROOT/'outputs/问题2_重复实验_种子2027',
               2028:ROOT/'outputs/问题2_重复实验_种子2028'}
    paths={}
    for seed, folder in originals.items():
        for method in ('baseline','augmentation','gated','distilled'):
            path=folder/f'问题2_{METHODS[method]}_种子{seed}.pt'
            if not path.is_file(): raise FileNotFoundError(path)
            paths[f'{method}_{seed}']=path
    models,ckpts=load_models(paths,args.device)
    first=next(iter(ckpts.values()))
    for ck in ckpts.values():
        if ck['provenance']['data_sha256']!=source_hash or ck['stats']!=first['stats']:
            raise ValueError('训练数据或归一化参数不一致，不能集成')
        if ck['provenance']['bert_revision']!=first['provenance']['bert_revision']:
            raise ValueError('BERT版本不一致，不能集成')
    stats=first['stats']; encoder=TextEncoder(args.device, first['provenance']['bert_revision'])
    with source.open('rb') as f: d=pickle.load(f)
    cache=ROOT/'work/问题2_特征缓存'
    valid=normalize(prepare_split(d['valid'],encoder,cache,'valid',source_hash),stats)
    quick=scenarios(valid,encoder,cache,stats,source_hash)
    candidates={method:[f'{method}_{seed}' for seed in originals] for method in ('baseline','augmentation','gated','distilled')}
    candidates['gated_distilled'] = candidates['gated']+candidates['distilled']
    score_rows=[]; score_lookup={}
    for name,members in candidates.items():
        r,_=summarize(members,models,valid,quick,args.device)
        score=selection_score(r); score_lookup[name]=score
        score_rows.append({'候选':name,'成员数':len(members),'验证选模分数':score,
                           '完整Macro_F1':r[0]['Macro_F1'],
                           '30%缺失平均Macro_F1':float(np.mean([x['Macro_F1'] for x in r[1:]])),
                           '完整MAE':r[0]['MAE'],
                           '30%缺失平均MAE':float(np.mean([x['MAE'] for x in r[1:]]))})
        print('CANDIDATE',score_rows[-1],flush=True)
    write_csv(out/'问题2_集成候选验证.csv',score_rows)
    # Pick only from robust candidates; baseline is a comparison, not a Q2 proposal.
    selected=max(('augmentation','gated','distilled','gated_distilled'), key=lambda name:score_lookup[name])
    members=candidates[selected]
    manifest={'selected':selected,'members':members,'validation_scores':score_lookup,
              'data_sha256':source_hash,'bert_revision':encoder.revision,
              'selection_suite':'完整输入 + T/A/V/AV/TAV 30%连续缺失；仅验证集',
              'checkpoint_sha256':{name:sha(paths[name]) for name in members},
              'test_used_for_selection':False,
              'output_consistency_rule':'Neutral=0; Negative<=-0.01; Positive>=0.01; applied to all reported scores and predictions'}
    dump(out/'问题2_集成选模结论.json',manifest)
    full=scenarios(valid,encoder,cache,stats,source_hash,full=True,repeats=2)
    report=[]
    for name in ('baseline',selected):
        rows,preds=summarize(candidates[name],models,valid,full,args.device)
        report.extend([{'模型':name,**r} for r in rows])
        save_predictions(out,f'问题2_验证集_{name}_全量预测.csv',valid,*preds[0])
    write_csv(out/'问题2_验证集缺失类型率位置.csv',report)
    # Test split is loaded only after all candidates, checkpoints and selection are frozen.
    test=normalize(prepare_split(d['test'],encoder,cache,'test',source_hash),stats)
    test_suite=scenarios(test,encoder,cache,stats,source_hash)
    test_report=[]
    for name in ('baseline',selected):
        rows,preds=summarize(candidates[name],models,test,test_suite,args.device)
        test_report.extend([{'模型':name,**r} for r in rows])
        save_predictions(out,f'问题2_留出测试集_{name}_全量预测.csv',test,*preds[0])
    write_csv(out/'问题2_留出测试集一次性评价.csv',test_report)
    # Attachment 3 has no labels. The same BERT and normalization interface is used.
    annex=[]
    files=sorted((root/'附件3-模态缺失特征样本/对齐版本').glob('*.pkl'))
    if len(files)!=30: raise ValueError(f'附件3对齐版本文件数={len(files)}，预期30')
    for file in files:
        with file.open('rb') as f: s=pickle.load(f)['test']
        s['id']=[file.stem] if len(s['audio'])==1 else [f'{file.stem}_{i}' for i in range(len(s['audio']))]
        item=normalize(prepare_split(s,encoder,cache,'annex3',sha(file)),stats)
        probability,intensity=predict_ensemble(models,members,item,args.device,None)
        for row in save_predictions(out/'逐样本预测',f'{file.stem}.csv',item,probability,intensity):
            row['模型']=selected; annex.append(row)
    if len(annex)!=30 or len({x['样本编号'] for x in annex})!=30:
        raise ValueError('附件3全量结果未通过数量检查')
    write_csv(out/'问题2_附件3全量预测.csv',annex)
    target=out/'模型参数'; target.mkdir(exist_ok=True)
    for name in members: shutil.copy2(paths[name],target/f'{name}.pt')
    dump(out/'问题2_完成状态.json',{'complete':True,'attachment3_count':len(annex),
        'selected':selected,'seeds':[2026,2027,2028],
        'model_files':[f'{name}.pt' for name in members],'github_synced':False})
    print('FINISHED',selected,'attachment3',len(annex),flush=True)


if __name__=='__main__': main()
