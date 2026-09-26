"""Explain the deployed predictor, including token interaction and deployment rules."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from q3_common import read_pickle, adapt, normalize, special_data, csv_write, dump, sha
from q3_train_utils import subset
from q3_explain_utils import explain_split, load_evidence_map
from q3_base_model import FrozenText
from q3_model import fresh_model
from q3_decision import adjust
from q3_adaptive_explain import AdaptivePredictor as BasePredictor


class AdaptivePredictor(BasePredictor):
    def __init__(self,components,device,bias=None):
        super().__init__(components,device)
        self.bias=np.asarray(bias if bias is not None else [0.,0.,0.])

    @torch.inference_mode()
    def masks(self,data,i,masks,batch=16,zero_feature=False):
        # Token-interaction models must re-encode all three streams for
        # each coalition. Calling the old pooled `heads()` path would bypass
        # the cross-modal Transformer and explain a different predictor.
        masks=np.asarray(masks,bool)
        probs=np.zeros((len(masks),3),np.float64)
        reg=np.zeros(len(masks),np.float64)
        for model,weight in self.components:
            model.eval()
            for start in range(0,len(masks),batch):
                active=masks[start:start+batch]
                n=len(active)
                b=np.repeat(data['b'][i:i+1],n,0)
                a=np.repeat(data['a'][i:i+1],n,0)*active[:,:,1,None]
                v=np.repeat(data['v'][i:i+1],n,0)*active[:,:,2,None]
                effective=np.repeat(data['mask'][i:i+1],n,0) if zero_feature else active
                if zero_feature:
                    # ``b`` stores the three BERT channels as (batch, 3, length),
                    # while coalition masks are (batch, length, modality).
                    # Expand the text coalition along the channel axis.
                    bmask=active[:,:,0][:,None,:]
                    b=np.where(bmask,data['b'][i:i+1],b)
                    # Keep the original observation mask for this comparison,
                    # but replace removed text tokens with the padding id.
                    # Preserve BERT special tokens so the sequence remains valid.
                    token_ids=b[:,0]
                    special=(token_ids==101)|(token_ids==102)
                    b[:,0]=np.where((~active[:,:,0]) & ~special,0,token_ids)
                inputs=[torch.as_tensor(x,device=self.device) for x in (b,a,v,effective)]
                logits,r=model(*inputs)
                probs[start:start+n]+=weight*logits.softmax(-1).detach().cpu().numpy()
                reg[start:start+n]+=weight*r.detach().cpu().numpy()
        probs/=probs.sum(-1,keepdims=True)
        prob=probs
        prob=adjust(prob,self.bias)
        return np.log(np.clip(prob,1e-12,1)),reg,prob


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
    pred=AdaptivePredictor(components,args.device,manifest['class_bias'])
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
    dump(args.out/'配置与审计/解释配置.json',{'model_version':5, 'class_bias':manifest['class_bias'], 'components':manifest['components'],
        'classification_target':'fixed full-input class vs runner-up; log probability difference of deployed predictor',
        'text_intervention':'token identity and attention removed BEFORE each adapted BERT',
        'zero_reference':'separate representation-zero game with original observed masks'})
