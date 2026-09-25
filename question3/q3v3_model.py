"""Trainable BERT LoRA and text-anchored residual A/V fusion for Q3 v3."""
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from q3v2_model import Temporal, FrozenText


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16, dropout=.1):
        super().__init__(); self.base = base; base.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = alpha / rank; self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.drop(x), self.lora_a), self.lora_b) * self.scale


class Head(nn.Module):
    def __init__(self, dim, dropout, hierarchical=False):
        super().__init__(); self.hierarchical = hierarchical
        self.body = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 128), nn.GELU(), nn.Dropout(dropout))
        self.cls = nn.Linear(128, 2 if hierarchical else 3)
        # Separate regression tower reduces competition with categorical neutral boundaries.
        self.reg = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, h):
        z = self.cls(self.body(h))
        if self.hierarchical:
            n, s = z.unbind(-1)
            z = torch.stack([F.logsigmoid(-n)+F.logsigmoid(-s), F.logsigmoid(n),
                             F.logsigmoid(-n)+F.logsigmoid(s)], -1)
        return z, self.reg(h).squeeze(-1)


class AdaptiveModel(nn.Module):
    def __init__(self, base_bert, config, prior, mean_score):
        super().__init__(); self.config = config; self.bert = base_bert
        self.bert.requires_grad_(False)
        n_layers = int(config['lora_layers'])
        if n_layers:
            for layer in self.bert.encoder.layer[-n_layers:]:
                for name in ('query', 'value'):
                    old = getattr(layer.attention.self, name)
                    if isinstance(old, LoRALinear): raise ValueError('Already injected LoRA')
                    setattr(layer.attention.self, name, LoRALinear(old, config['rank'], config['rank']*2))
        drop = config['dropout']
        self.text_head = Head(base_bert.config.hidden_size * 2, drop, config['hierarchical'])
        self.av_enc = nn.ModuleList([Temporal(74, 64, drop), Temporal(35, 64, drop)])
        self.av_head = Head(128, drop, config['hierarchical'])
        self.gate = nn.Sequential(nn.Linear(128+2, 32), nn.GELU(), nn.Linear(32, 2))
        nn.init.constant_(self.gate[-1].bias, -1.5)
        self.register_buffer('prior', torch.tensor(prior, dtype=torch.float32))
        self.register_buffer('mean_score', torch.tensor(float(mean_score)))

    def train(self, mode=True):
        super().train(mode)
        # Frozen pretrained dropout is disabled; only adapters and task heads regularize.
        self.bert.eval()
        for module in self.bert.modules():
            if isinstance(module, LoRALinear): module.drop.train(mode)
        return self

    def encode(self, b, mask):
        observed = mask[:, :, 0]
        special = (b[:, 0] == 101) | (b[:, 0] == 102)
        attention = (observed | special) & b[:, 1].bool()
        ids = b[:, 0].masked_fill(~attention, 0)
        h = self.bert(input_ids=ids, attention_mask=attention.long(), token_type_ids=b[:, 2]).last_hidden_state
        mean = (h * observed[..., None]).sum(1) / observed.sum(1, keepdim=True).clamp_min(1)
        return torch.cat([h[:, 0], mean], -1) * observed.any(1)[:, None]

    def heads(self, text, a, v, mask, return_aux=False):
        available = mask.any(1)
        tl, tr = self.text_head(text)
        pooled = [enc(x, mask[:, :, m+1]) for m, (enc, x) in enumerate(zip(self.av_enc, (a, v)))]
        av = torch.cat(pooled, -1)
        al, ar = self.av_head(av)
        # Evidence residuals are relative to learned train priors; zero AV contributes zero.
        prior_l = self.prior.clamp_min(1e-8).log()[None]
        tc = tl - tl.mean(-1, keepdim=True)
        ac = al - al.mean(-1, keepdim=True)
        pl = prior_l - prior_l.mean(-1, keepdim=True)
        fractions = mask[:, :, 1:].float().mean(1)
        gates = .5 * torch.sigmoid(self.gate(torch.cat([av, fractions], -1)))
        has_t = available[:, 0]; has_av = available[:, 1:].any(1)
        logits = torch.where(has_t[:, None], tc, pl)
        intensity = torch.where(has_t, tr, self.mean_score)
        weight_cls = torch.where(has_t, gates[:, 0], torch.ones_like(gates[:, 0])) * has_av
        weight_reg = torch.where(has_t, gates[:, 1], torch.ones_like(gates[:, 1])) * has_av
        logits = logits + weight_cls[:, None] * (ac - pl)
        intensity = intensity + weight_reg * (ar - self.mean_score)
        intensity = 3 * torch.tanh(intensity / 3)
        empty = ~available.any(1)
        logits = torch.where(empty[:, None], prior_l, logits)
        intensity = torch.where(empty, self.mean_score, intensity)
        if return_aux: return logits, intensity, tl, 3*torch.tanh(tr/3), al, 3*torch.tanh(ar/3)
        return logits, intensity

    def forward(self, b, a, v, mask, return_aux=False):
        return self.heads(self.encode(b, mask), a, v, mask, return_aux)

    def portable_state(self):
        keep = {name for name, p in self.named_parameters() if p.requires_grad}
        keep |= {name for name, _ in self.named_buffers() if not name.startswith('bert.')}
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k in keep}

    def load_portable(self, state):
        expected = set(self.portable_state())
        if set(state) != expected: raise ValueError(f'Portable state mismatch: {expected ^ set(state)}')
        self.load_state_dict(state, strict=False)


def fresh_model(bert_path, config, prior, mean_score, device, verify=True):
    from transformers import AutoModel
    if verify:
        encoder = FrozenText(bert_path, 'cpu')
        base = encoder.model; fingerprint = encoder.fingerprint; del encoder
    else:
        base = AutoModel.from_pretrained(bert_path, local_files_only=True, attn_implementation='eager')
        fingerprint = None
    return AdaptiveModel(base, config, prior, mean_score).to(device), fingerprint


def model_inputs(data, idx, device, mask=None):
    return tuple(torch.as_tensor(x, device=device) for x in
                 (data['b'][idx], data['a'][idx], data['v'][idx], data['mask'][idx] if mask is None else mask))
