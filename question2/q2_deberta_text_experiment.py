"""Leakage-safe DeBERTa-v3 text experiment for Problem 2.

The official full MOSEI train split is fitted and its video-disjoint valid
split is used for early stopping.  Token masking is applied before DeBERTa so
the experiment can later be combined with the missing-modality model without
letting hidden words leak through contextual states.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch import nn
from transformers import AutoModel, AutoTokenizer


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows: return
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


class DebertaSentiment(nn.Module):
    def __init__(self, model_path, dropout=.15, train_layers=3):
        super().__init__()
        self.base=AutoModel.from_pretrained(model_path,local_files_only=True)
        layers=self.base.encoder.layer
        freeze=max(0,len(layers)-train_layers)
        for p in self.base.embeddings.parameters(): p.requires_grad_(False)
        for layer in layers[:freeze]:
            for p in layer.parameters(): p.requires_grad_(False)
        hidden=self.base.config.hidden_size
        self.head=nn.Sequential(nn.LayerNorm(hidden),nn.Dropout(dropout),
                                nn.Linear(hidden,256),nn.GELU(),nn.Dropout(dropout))
        self.cls=nn.Linear(256,3); self.reg=nn.Linear(256,1)

    def forward(self,input_ids,attention_mask):
        state=self.base(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        weight=attention_mask.float(); weight[:,0]=0
        pooled=(state*weight[:,:,None]).sum(1)/weight.sum(1,keepdim=True).clamp_min(1)
        z=self.head(pooled)
        return self.cls(z),3*torch.tanh(self.reg(z).squeeze(-1)/3)


def tokenize(tokenizer,texts,max_length):
    encoded=tokenizer(list(map(str,texts)),padding='max_length',truncation=True,
                      max_length=max_length,return_tensors='np')
    return np.asarray(encoded['input_ids'],np.int64),np.asarray(encoded['attention_mask'],np.int64)


def mask_tokens(ids,attention,mask_id,rng,prob=.35):
    ids=ids.copy(); attention=attention.copy()
    for i in range(len(ids)):
        if rng.random()>=prob: continue
        valid=np.flatnonzero(attention[i])[1:-1]
        if len(valid)<2: continue
        rate=float(rng.choice((.1,.2,.3,.4)))
        count=max(1,int(round(rate*len(valid))))
        if rng.random()<.65:
            start=int(rng.integers(0,len(valid)-count+1)); hidden=valid[start:start+count]
        else:
            hidden=rng.choice(valid,size=count,replace=False)
        ids[i,hidden]=mask_id
    return ids,attention


@torch.inference_mode()
def evaluate(model,ids,attention,y,r,device,batch_size):
    model.eval(); probs=[]; regs=[]
    for start in range(0,len(y),batch_size):
        idx=slice(start,start+batch_size)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.startswith('cuda')):
            logits,reg=model(torch.as_tensor(ids[idx],device=device),
                             torch.as_tensor(attention[idx],device=device))
        probs.append(logits.softmax(-1).float().cpu().numpy()); regs.append(reg.float().cpu().numpy())
    p=np.concatenate(probs); pr=np.concatenate(regs); pred=p.argmax(1)
    return {'Accuracy':float(accuracy_score(y,pred)),
            'Macro_F1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'Weighted_F1':float(f1_score(y,pred,average='weighted',zero_division=0)),
            'MAE':float(mean_absolute_error(r,pr))},p,pr


def run(args):
    seed_all(args.seed); started=time.time(); device=args.device
    source=Path(args.safe_data).resolve(strict=True); out=Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    with source.open('rb') as f: data=pickle.load(f)
    if set(data)!={'train','valid'}: raise ValueError('Only safe train/valid data is allowed')
    train,valid=data['train'],data['valid']
    train_ids=list(map(str,train['id'])); valid_ids=list(map(str,valid['id']))
    train_groups={x.split('$_$',1)[0] for x in train_ids}; valid_groups={x.split('$_$',1)[0] for x in valid_ids}
    if train_groups & valid_groups: raise ValueError('video group leakage')
    tokenizer=AutoTokenizer.from_pretrained(args.model_path,local_files_only=True)
    xtr,mtr=tokenize(tokenizer,train['raw_text'],args.max_length)
    xva,mva=tokenize(tokenizer,valid['raw_text'],args.max_length)
    ytr=np.asarray(train['classification_labels'],np.int64); rtr=np.asarray(train['regression_labels'],np.float32)
    yva=np.asarray(valid['classification_labels'],np.int64); rva=np.asarray(valid['regression_labels'],np.float32)
    model=DebertaSentiment(args.model_path,args.dropout,args.train_layers).to(device)
    base_params=[p for n,p in model.named_parameters() if n.startswith('base.') and p.requires_grad]
    head_params=[p for n,p in model.named_parameters() if not n.startswith('base.')]
    optimizer=torch.optim.AdamW([{'params':base_params,'lr':args.encoder_lr},
                                 {'params':head_params,'lr':args.head_lr}],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,mode='max',factor=.5,patience=args.lr_patience,min_lr=1e-7
    )
    counts=np.bincount(ytr,minlength=3).astype(np.float64)
    weights=torch.as_tensor((len(ytr)/(3*counts))**args.weight_power,dtype=torch.float32,device=device)
    config={'model_path':str(Path(args.model_path).resolve()),'source_sha256':sha(source),
            'train_samples':len(ytr),'valid_samples':len(yva),'video_group_overlap':0,
            'train_layers':args.train_layers,'max_length':args.max_length,
            'encoder_lr':args.encoder_lr,'head_lr':args.head_lr,'mask_probability':args.mask_prob,
            'selection':'clean Macro-F1 - 0.05*MAE','test_or_attachment3_labels_used':False}
    (out/'问题2_DeBERTa实验配置.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    best=-1e9; stale=0; logs=[]; best_path=out/'问题2_DeBERTa文本最佳.pt'
    for epoch in range(1,args.epochs+1):
        model.train(); rng=np.random.default_rng(args.seed+epoch); order=rng.permutation(len(ytr)); losses=[]
        for start in range(0,len(order),args.batch_size):
            idx=order[start:start+args.batch_size]
            ids,att=mask_tokens(xtr[idx],mtr[idx],tokenizer.mask_token_id,rng,args.mask_prob)
            ids=torch.as_tensor(ids,device=device); att=torch.as_tensor(att,device=device)
            y=torch.as_tensor(ytr[idx],device=device); r=torch.as_tensor(rtr[idx],device=device)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.startswith('cuda')):
                logits,reg=model(ids,att)
                loss=F.cross_entropy(logits,y,weight=weights,label_smoothing=args.label_smoothing)
                loss=loss+args.reg_weight*F.smooth_l1_loss(reg,r)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); losses.append(float(loss.detach()))
        metric,prob,reg=evaluate(model,xva,mva,yva,rva,device,args.eval_batch_size)
        score=metric['Macro_F1']-.05*metric['MAE']
        row={'epoch':epoch,'train_loss':float(np.mean(losses)),**metric,'selection_score':float(score),
             'encoder_lr':float(optimizer.param_groups[0]['lr']),
             'head_lr':float(optimizer.param_groups[1]['lr'])}
        logs.append(row); write_csv(out/'问题2_DeBERTa逐轮训练记录.csv',logs)
        print(json.dumps(row,ensure_ascii=False),flush=True)
        if score>best+1e-4:
            best=score; stale=0
            temp=best_path.with_suffix('.tmp')
            torch.save({'model':model.state_dict(),'epoch':epoch,'metrics':metric,'score':score,'config':config},temp); os.replace(temp,best_path)
            np.savez_compressed(out/'问题2_DeBERTa验证预测.npz',probability=prob,regression=reg,y=yva,r=rva,ids=np.asarray(valid_ids))
        else: stale+=1
        scheduler.step(score)
        if stale>=args.patience: break
    checkpoint=torch.load(best_path,map_location=device,weights_only=False); model.load_state_dict(checkpoint['model'])
    metric,_,_=evaluate(model,xva,mva,yva,rva,device,args.eval_batch_size)
    final={'best_epoch':checkpoint['epoch'],'clean_metrics':metric,'selection_score':checkpoint['score'],
           'best_checkpoint':str(best_path),'best_checkpoint_sha256':sha(best_path),
           'elapsed_seconds':time.time()-started,'model_saved':True,'test_or_attachment3_labels_used':False}
    (out/'问题2_DeBERTa实验结论.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(final,ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--safe-data',required=True); ap.add_argument('--model-path',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--device',default='cuda:1'); ap.add_argument('--seed',type=int,default=2040)
    ap.add_argument('--epochs',type=int,default=8); ap.add_argument('--patience',type=int,default=3)
    ap.add_argument('--lr-patience',type=int,default=2)
    ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--eval-batch-size',type=int,default=96)
    ap.add_argument('--max-length',type=int,default=96); ap.add_argument('--train-layers',type=int,default=3)
    ap.add_argument('--encoder-lr',type=float,default=1.5e-5); ap.add_argument('--head-lr',type=float,default=2e-4)
    ap.add_argument('--dropout',type=float,default=.15); ap.add_argument('--weight-power',type=float,default=.5)
    ap.add_argument('--mask-prob',type=float,default=.30); ap.add_argument('--reg-weight',type=float,default=.15)
    ap.add_argument('--label-smoothing',type=float,default=.03)
    run(ap.parse_args())
