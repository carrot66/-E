"""针对中性类别误判的分层门控消融，仅用附件2训练/验证划分。"""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from q2_experiment import (ROOT, TextEncoder, block_mask, evaluate, fit_normalization,
                           normalize, norm_text, prepare_split, runtime, scenarios,
                           seed_all, sha, text_features, train_one, write_csv, save_predictions,
                           dump)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',default=str(ROOT/'E题数据/E题数据'))
    p.add_argument('--out',default=str(ROOT/'outputs/question2/问题2_分层模型实验'))
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--epochs',type=int,default=30)
    args=p.parse_args(); runtime(args.device)
    root=Path(args.data_root); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    source=root/'附件2-数据集特征文件/aligned_50.pkl'; source_hash=sha(source)
    with source.open('rb') as f:data=pickle.load(f)
    config=json.loads((ROOT/'outputs/question2/问题2_首轮实验结果/问题2_实验配置.json').read_text(encoding='utf-8'))
    ckpt=torch.load(ROOT/'outputs/question2/问题2_首轮实验结果/问题2_普通融合_种子2026.pt',map_location='cpu',weights_only=False)
    stats=ckpt['stats']; encoder=TextEncoder(args.device,config['bert_revision'])
    cache=ROOT/'work/问题2_特征缓存'
    tr=normalize(prepare_split(data['train'],encoder,cache,'train',source_hash),stats)
    va=normalize(prepare_split(data['valid'],encoder,cache,'valid',source_hash),stats)
    banks=[]
    for rate in (.1,.3,.5):
        hide=block_mask(tr['valid'],rate,51000+int(rate*100))
        observed=tr['mask'][:,:,0]&~hide
        text=text_features(encoder,tr['b'],observed,cache,'train_aug',source_hash)
        banks.append({'hide':hide,'t':norm_text(text,observed,stats)})
    quick=scenarios(va,encoder,cache,stats,source_hash)
    full=scenarios(va,encoder,cache,stats,source_hash,full=True,repeats=2)
    provenance=dict(config); provenance['code_sha256']=sha(ROOT/'question2/q2_experiment.py')
    provenance['variant']='hierarchical neutral gate and polarity gate'
    selected=[]
    for seed in (2026,2027,2028):
        seed_all(seed); sub=out/f'种子{seed}'; sub.mkdir(parents=True,exist_ok=True)
        model,path,logs=train_one('hierarchical',seed,tr,va,quick,banks,args.device,sub,args.epochs,None,stats,provenance)
        rows,preds=evaluate(model,va,full,args.device)
        write_csv(sub/'问题2_验证集缺失实验.csv',[{'模型':'hierarchical','种子':seed,**r} for r in rows])
        save_predictions(sub,'问题2_验证集全量预测.csv',va,*preds[0])
        clean=rows[0]; random=[r for r in rows[1:] if r['position']=='random']
        x={'seed':seed,'model_file':str(path),'best_epoch':int(torch.load(path,map_location='cpu',weights_only=False)['epoch']),
           'clean_F1':clean['Macro_F1'],'clean_MAE':clean['MAE'],
           'missing_F1':float(np.mean([r['Macro_F1'] for r in random])),
           'missing_MAE':float(np.mean([r['MAE'] for r in random]))}
        selected.append(x); print('SEED_FINISHED',json.dumps(x,ensure_ascii=False),flush=True)
    dump(out/'问题2_分层模型验证汇总.json',{'source_sha256':source_hash,'seeds':selected,
         'test_used_for_selection':False,'attachment3_used_for_training':False})
    print('FINISHED',flush=True)

if __name__=='__main__':main()
