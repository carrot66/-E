"""Explain the actual v3 predictor, including adapters and all deployed ensemble members."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from q3v2_common import read_pickle, adapt, normalize, special_data, csv_write, dump, sha
from q3v2_train import subset
from q3v2_explain import explain_split, load_evidence_map
from q3v2_model import FrozenText
from q3v3_model import fresh_model


class AdaptivePredictor:
    def __init__(self, components, device):
        self.components = components; self.device = device
        self.cache_key = None; self.text_cache = []

    @torch.inference_mode()
    def masks(self, data, i, masks, batch=16, zero_feature=False):
        masks = np.asarray(masks, bool)
        key = (id(data), i)
        if key != self.cache_key:
            self.cache_key = key; self.text_cache = [{} for _ in self.components]
        probs = np.zeros((len(masks),3), np.float64); regression = np.zeros(len(masks), np.float64)
        for j,(model,weight) in enumerate(self.components):
            model.eval(); cache = self.text_cache[j]
            # Cache repeated text masks: many A/V windows share the original text.
            text_masks = np.repeat(data['mask'][i:i+1,:,0], len(masks), 0) if zero_feature else masks[:,:,0]
            keys = [m.tobytes() for m in text_masks]
            missing = {}
            for k,m in zip(keys,text_masks):
                if k not in cache: missing[k] = m
            entries = list(missing.items())
            for start in range(0,len(entries),batch):
                part = entries[start:start+batch]; n=len(part)
                observed = np.repeat(data['mask'][i:i+1],n,0)
                observed[:,:,0] = np.stack([m for _,m in part])
                b = torch.as_tensor(np.repeat(data['b'][i:i+1],n,0),device=self.device)
                text = model.encode(b,torch.as_tensor(observed,device=self.device)).cpu().numpy()
                for (k,_),t in zip(part,text): cache[k]=t
            for start in range(0,len(masks),batch):
                active=masks[start:start+batch]; n=len(active)
                text=np.stack([cache[k] for k in keys[start:start+batch]])
                effective=active
                if zero_feature:
                    # Alternate game: keep observed masks, zero the removed modality representation.
                    effective=np.repeat(data['mask'][i:i+1],n,0)
                    text=text * active[:,:,0].any(1)[:,None]
                a=np.repeat(data['a'][i:i+1],n,0)*active[:,:,1,None]
                v=np.repeat(data['v'][i:i+1],n,0)*active[:,:,2,None]
                inputs=[torch.as_tensor(x,device=self.device) for x in (text,a,v,effective)]
                logits,r=model.heads(*inputs)
                probs[start:start+n] += weight*logits.softmax(-1).cpu().numpy()
                regression[start:start+n] += weight*r.cpu().numpy()
        probs /= probs.sum(-1,keepdims=True)
        # log(mean probabilities) defines precisely the deployed ensemble's class margins.
        return np.log(np.clip(probs,1e-12,1)), regression, probs


def explain(args):
    manifest=json.loads((args.out/'模型参数/manifest.json').read_text(encoding='utf-8'))
    frozen=FrozenText(args.bert,'cpu'); tokenizer=frozen.tokenizer; fingerprint=frozen.fingerprint; del frozen
    if fingerprint!=manifest['bert_fingerprint']: raise ValueError('Changed BERT weights')
    components=[]; stats=None
    for record in manifest['components']:
        path=args.out/record['path']
        if sha(path)!=record['sha256']: raise ValueError('Checkpoint checksum failed')
        cp=torch.load(path,map_location='cpu',weights_only=False)
        model,_=fresh_model(args.bert,cp['config'],cp['prior'],cp['mean_score'],args.device,verify=False)
        model.load_portable(cp['state']); components.append((model.eval(),record['weight']))
        if stats is not None:
            for k in stats:
                if not np.array_equal(stats[k],cp['stats'][k]): raise ValueError('Ensemble normalizers differ')
        stats=cp['stats']
    pred=AdaptivePredictor(components,args.device)
    file=args.data_root/'附件2-数据集特征文件/aligned_50.pkl'
    source_hash=sha(file)
    audit=json.loads((args.out/'配置与审计/数据审计.json').read_text(encoding='utf-8'))
    if source_hash!=audit['source_sha256']: raise ValueError('Data changed after training')
    valid=adapt(read_pickle(file)['valid'],tokenizer,True)
    if manifest['smoke']:
        valid=subset(valid,np.concatenate([np.flatnonzero(valid['y']==c)[:2] for c in range(3)]))
    valid=normalize(valid,stats)
    selected=set(); rng=np.random.default_rng(2026)
    for c in range(3):
        pool=np.flatnonzero(valid['y']==c)
        selected.update(map(int,rng.choice(pool,min(len(pool),math.ceil(args.explain_valid/3)),replace=False)))
    explain_split(args,pred,valid,'验证集',selected,{})
    special,files=special_data(args.data_root,tokenizer); special=normalize(special,stats)
    evidence_map=load_evidence_map(args.evidence_map)
    explain_split(args,pred,special,'附件4',set(range(20)),evidence_map)
    csv_write(args.out/'配置与审计/附件4输入审计.csv',special['audit'])
    dump(args.out/'配置与审计/附件4来源.json',[{'file':str(f.relative_to(args.data_root)), 'sha256':sha(f)} for f in files])
    csv_write(args.out/'附件4预测/视频映射.csv',[{'sample_id':s,'video_relative':str(Path(p).relative_to(args.data_root)),
              'exists':Path(p).is_file()} for s,p in zip(special['ids'],special['videos'])])
    dump(args.out/'配置与审计/解释配置.json',{'model_version':3, 'components':manifest['components'],
        'classification_target':'fixed full-input class vs runner-up; log probability difference of deployed predictor',
        'text_intervention':'token identity and attention removed BEFORE each adapted BERT',
        'zero_reference':'separate representation-zero game with original observed masks'})
