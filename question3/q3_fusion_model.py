"""BERT adaptation and optional text-conditioned multimodal fusion head."""
import torch
from torch import nn
from q3_adaptive_model import AdaptiveModel, Head


class TunedModel(AdaptiveModel):
    def __init__(self,base,config,prior,mean_score):
        super().__init__(base,config,prior,mean_score)
        layers=int(config.get('full_layers',0))
        if layers:
            if config['lora_layers']: raise ValueError('Do not combine full-layer tuning and LoRA in this experiment')
            for layer in self.bert.encoder.layer[-layers:]: layer.requires_grad_(True)
        if config.get('fusion')=='interaction':
            self.text_project=nn.Sequential(nn.LayerNorm(base.config.hidden_size*2),nn.Linear(base.config.hidden_size*2,64),nn.GELU())
            self.interaction_head=Head(64+128+3,config['dropout'],config['hierarchical'])

    def heads(self,text,a,v,mask,return_aux=False):
        if self.config.get('fusion')!='interaction': return super().heads(text,a,v,mask,return_aux)
        available=mask.any(1)
        tl,tr=self.text_head(text)
        av=torch.cat([enc(x,mask[:,:,m+1]) for m,(enc,x) in enumerate(zip(self.av_enc,(a,v)))],-1)
        al,ar=self.av_head(av)
        t=self.text_project(text)*available[:,0,None]
        joint=torch.cat([t,av,available.float()],-1)
        delta,delta_r=self.interaction_head(joint)
        # Stable text anchor; joint residual can condition its correction on the text itself.
        prior_l=self.prior.clamp_min(1e-8).log()[None]
        anchor=torch.where(available[:,0,None],tl,prior_l)
        anchor_r=torch.where(available[:,0],tr,self.mean_score)
        has_av=available[:,1:].any(1)
        logits=anchor+.5*delta*has_av[:,None]
        reg=3*torch.tanh((anchor_r+.5*delta_r*has_av)/3)
        empty=~available.any(1)
        logits=torch.where(empty[:,None],prior_l,logits)
        reg=torch.where(empty,self.mean_score,reg)
        if return_aux: return logits,reg,tl,3*torch.tanh(tr/3),al,3*torch.tanh(ar/3)
        return logits,reg


def fresh_model(bert_path,config,prior,mean_score,device,verify=False):
    from transformers import AutoModel
    from q3_base_model import FrozenText
    fingerprint=None
    if verify:
        original=FrozenText(bert_path,'cpu'); base=original.model; fingerprint=original.fingerprint
    else: base=AutoModel.from_pretrained(bert_path,local_files_only=True,attn_implementation='eager')
    return TunedModel(base,config,prior,mean_score).to(device),fingerprint


def candidates(seed):
    base=dict(rank=16,dropout=.25,head_lr=1e-4,adapter_lr=2e-4,seed=seed,lora_layers=12,
              hierarchical=False,full_layers=0,fusion='residual',weight_power=.25,smoothing=0.,
              reg_weight=.15,text_ce=.15,text_reg=.05,av_aux=.025,augmentation=.05)
    return {
        'lora12_softmax':dict(base),
        'lora12_hierarchical':dict(base,hierarchical=True),
        'lora12_unweighted':dict(base,weight_power=0.),
        'lora12_interaction':dict(base,fusion='interaction'),
        'full2_softmax':dict(base,lora_layers=0,full_layers=2,adapter_lr=1e-5,head_lr=5e-5),
        'full4_softmax':dict(base,lora_layers=0,full_layers=4,adapter_lr=1e-5,head_lr=5e-5),
    }
