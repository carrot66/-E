"""问题2进一步实验：仅微调BERT最后一层，测试中性与文本缺失瓶颈。

This script is an experimental candidate. It never reads attachment-2 test or
attachment-3 labels for training/selection. The saved delta contains only the
trainable BERT layer and small fusion head, not the full pretrained model.
"""
from __future__ import annotations
import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from q2_experiment import (ROOT, Fusion, block_mask, derive_masks, dump, metrics,
                           runtime, seed_all, sha, write_csv)


class TrainableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert=AutoModel.from_pretrained('google-bert/bert-base-uncased',local_files_only=True)
        self.revision=getattr(self.bert.config,'_commit_hash',None)
        for p in self.bert.parameters():p.requires_grad_(False)
        for p in self.bert.encoder.layer[-1].parameters():p.requires_grad_(True)
        self.fusion=Fusion(gated=True)

    def forward(self,b,a,v,mask,valid):
        ids=b[:,0].clone(); att=b[:,1].clone(); types=b[:,2]
        redact=valid & ~mask[:,:,0]
        ids[redact]=103; att[redact]=0
        text=self.bert(input_ids=ids,attention_mask=att,token_type_ids=types).last_hidden_state
        text=text*mask[:,:,0,None]
        return self.fusion(text,a,v,mask,valid)


def prepare(split,stats=None):
    b,valid,mask=derive_masks(split)
    a=np.asarray(split['audio'],np.float32); v=np.asarray(split['vision'],np.float32)
    if stats is None:
        stats={}
        for j,k,x in [(1,'a',a),(2,'v',v)]:
            observed=x[mask[:,:,j]]
            stats[k]={'mean':observed.mean(0).tolist(),'std':np.maximum(observed.std(0),1e-4).tolist()}
    for j,k,x in [(1,'a',a),(2,'v',v)]:
        mean=np.asarray(stats[k]['mean'],np.float32); std=np.asarray(stats[k]['std'],np.float32)
        normalized=np.clip((x-mean)/std,-8,8)*mask[:,:,j,None]
        if k=='a':a=normalized
        else:v=normalized
    return {'b':b,'a':a,'v':v,'valid':valid,'mask':mask,
            'y':np.asarray(split['classification_labels'],np.int64),
            'r':np.asarray(split['regression_labels'],np.float32),
            'id':list(map(str,split['id']))},stats


def batch_data(split,index,mask,device):
    a=torch.as_tensor(split['a'][index]*mask[:,:,1,None],device=device)
    v=torch.as_tensor(split['v'][index]*mask[:,:,2,None],device=device)
    return (torch.as_tensor(split['b'][index].copy(),dtype=torch.long,device=device),
            a,v,torch.as_tensor(mask,device=device),torch.as_tensor(split['valid'][index],device=device))


def scenario_masks(split):
    return [('none',0,split['mask'])]+[(subset,.3, mask_for(split,subset,.3)) for subset in ('T','A','V','AV','TAV')]


def mask_for(split,subset,rate):
    mask=split['mask'].copy()
    hide=block_mask(split['valid'],rate,82000+int(rate*100))
    for ch in subset:mask[:,:,'TAV'.index(ch)] &= ~hide
    return mask


@torch.inference_mode()
def infer(model,split,mask,device,batch=64):
    model.eval(); pp=[]; rr=[]
    for start in range(0,len(split['id']),batch):
        idx=np.arange(start,min(start+batch,len(split['id'])))
        args=batch_data(split,idx,mask[idx],device)
        logits,reg=model(*args)
        pp.append(logits.softmax(-1).cpu().numpy());rr.append(reg.cpu().numpy())
    prob=np.concatenate(pp);reg=np.concatenate(rr)
    cls=prob.argmax(1);reg[cls==1]=0.;reg[cls==0]=np.minimum(reg[cls==0],-.01);reg[cls==2]=np.maximum(reg[cls==2],.01)
    return prob,reg


def evaluate(model,split,scenarios,device):
    rows=[]
    for name,rate,mask in scenarios:
        p,r=infer(model,split,mask,device)
        rows.append({'subset':name,'rate':rate,**metrics(split['y'],split['r'],p,r)})
    return rows


def score(rows):
    clean=rows[0]; other=rows[1:]
    return .5*(clean['Macro_F1']+np.mean([x['Macro_F1'] for x in other]))-.05*(clean['MAE']+np.mean([x['MAE'] for x in other]))


def run(args):
    runtime(args.device);seed_all(args.seed)
    root=Path(args.data_root);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    source=root/'附件2-数据集特征文件/aligned_50.pkl'; source_hash=sha(source)
    with source.open('rb') as f:d=pickle.load(f)
    train,stats=prepare(d['train']);valid,_=prepare(d['valid'],stats)
    model=TrainableModel().to(args.device)
    params=[{'params':model.bert.encoder.layer[-1].parameters(),'lr':2e-5},
            {'params':model.fusion.parameters(),'lr':3e-4}]
    optim=torch.optim.AdamW(params,weight_decay=1e-3)
    class_weights=torch.as_tensor(np.sqrt(len(train['y'])/(3*np.bincount(train['y'],minlength=3))),dtype=torch.float32,device=args.device)
    scenarios=scenario_masks(valid)
    best=-float('inf');bad=0;logs=[]
    for ep in range(1,args.epochs+1):
        model.train();rng=np.random.default_rng(args.seed+ep*1009)
        order=rng.permutation(len(train['id']));losses=[]
        for st in range(0,len(order),args.batch):
            idx=order[st:st+args.batch];mask=train['mask'][idx].copy()
            for k,i in enumerate(idx):
                if rng.random()<.65:
                    subset=random.choice(('T','A','V','TA','TV','AV','TAV'))
                    rate=float(rng.choice([.1,.3,.5]))
                    hide=block_mask(train['valid'][i:i+1],rate,int(rng.integers(0,2**31)))[0]
                    for ch in subset:mask[k,:,'TAV'.index(ch)] &= ~hide
            args_batch=batch_data(train,idx,mask,args.device)
            y=torch.as_tensor(train['y'][idx],device=args.device)
            r=torch.as_tensor(train['r'][idx],device=args.device)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=args.device.startswith('cuda')):
                logits,reg=model(*args_batch)
                loss=F.cross_entropy(logits.float(),y,weight=class_weights)+.6*F.smooth_l1_loss(reg.float(),r)
            optim.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step()
            losses.append(float(loss.detach()))
        rows=evaluate(model,valid,scenarios,args.device)
        value=score(rows)
        log={'epoch':ep,'train_loss':float(np.mean(losses)),'val_clean_F1':rows[0]['Macro_F1'],
             'val_missing_F1':float(np.mean([x['Macro_F1'] for x in rows[1:]])),
             'val_clean_MAE':rows[0]['MAE'],'val_missing_MAE':float(np.mean([x['MAE'] for x in rows[1:]])),
             'selection_score':float(value)}
        logs.append(log);write_csv(out/'问题2_BERT末层微调训练记录.csv',logs)
        print(json.dumps(log,ensure_ascii=False),flush=True)
        if value>best+1e-4:
            best=float(value);bad=0
            trainable={k:p.detach().cpu().half() for k,p in model.state_dict().items()
                       if k.startswith('fusion.') or k.startswith('bert.encoder.layer.11.')}
            torch.save({'delta':trainable,'stats':stats,'epoch':ep,'seed':args.seed,
                'bert_revision':model.revision,'source_sha256':source_hash,'selection_score':best,
                'architecture':'BERT final layer fine-tune + gated Fusion(hidden=96)'},out/'问题2_BERT末层微调最优参数.pt')
        else:bad+=1
        if bad>=args.patience:break
    checkpoint=torch.load(out/'问题2_BERT末层微调最优参数.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['delta'],strict=False)
    rows=evaluate(model,valid,scenarios,args.device)
    write_csv(out/'问题2_验证集基础结果.csv',rows)
    dump(out/'问题2_完成状态.json',{'training_finished':True,'best_epoch':checkpoint['epoch'],
        'selection_score':checkpoint['selection_score'],'model_bytes':(out/'问题2_BERT末层微调最优参数.pt').stat().st_size,
        'test_used_for_selection':False,'attachment3_used_for_training':False})
    print('FINISHED',json.dumps(rows,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',default=str(ROOT/'E题数据/E题数据'))
    p.add_argument('--out',default=str(ROOT/'outputs/question2/问题2_BERT微调实验'))
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--patience',type=int,default=3)
    run(p.parse_args())
