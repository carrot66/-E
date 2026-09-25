"""Frozen offline BERT plus independent temporal axes and modality-level fusion."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from q3v2_common import sha


class FrozenText:
    def __init__(self, path, device):
        from transformers import AutoModel, AutoTokenizer
        path = Path(path)
        manifest = json.loads((path / 'transfer_manifest.json').read_text(encoding='utf-8'))
        if manifest['repo'] != 'google-bert/bert-base-uncased' or manifest['revision'] != '86b5e0934494bd15c9632b12f734a8a67f723594':
            raise ValueError('Wrong BERT repository/revision')
        for filename, expected in manifest['files'].items():
            f = (path / filename).resolve()
            if not f.is_relative_to(path.resolve()) or sha(f) != expected: raise ValueError(f'BERT checksum mismatch: {filename}')
        self.fingerprint = sha(path / 'transfer_manifest.json')
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).to(device).eval()
        self.model.requires_grad_(False)
        self.device = device

    @torch.inference_mode()
    def encode(self, b, observed, batch=64):
        result = []
        for start in range(0, len(b), batch):
            inp = torch.as_tensor(b[start:start+batch], dtype=torch.long, device=self.device).clone()
            obs = torch.as_tensor(observed[start:start+batch], dtype=torch.bool, device=self.device)
            special = (inp[:, 0] == self.tokenizer.cls_token_id) | (inp[:, 0] == self.tokenizer.sep_token_id)
            attention = (obs | special) & inp[:, 1].bool()
            inp[:, 0].masked_fill_(~attention, self.tokenizer.pad_token_id)
            h = self.model(input_ids=inp[:, 0], attention_mask=attention.long(), token_type_ids=inp[:, 2]).last_hidden_state
            result.append((h * obs.unsqueeze(-1)).cpu().numpy())
        return np.concatenate(result).astype(np.float32)


class Temporal(nn.Module):
    def __init__(self, dim, hidden, dropout, temporal=True):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.convs = nn.ModuleList([nn.Conv1d(hidden, hidden, 3, padding=d, dilation=d, groups=hidden) for d in (1, 2)]) if temporal else nn.ModuleList()
        self.mix = nn.ModuleList([nn.Linear(hidden, hidden) for _ in self.convs])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in self.convs])
        self.drop = nn.Dropout(dropout)
        self.pool = nn.Linear(hidden, 1)

    def forward(self, x, mask):
        x = self.project(x) * mask[..., None]
        for conv, mix, norm in zip(self.convs, self.mix, self.norm):
            z = conv(x.transpose(1, 2)).transpose(1, 2)
            x = norm(x + self.drop(mix(torch.nn.functional.gelu(z)))) * mask[..., None]
        w = torch.softmax(self.pool(x).squeeze(-1).masked_fill(~mask, -1e4), -1) * mask
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-8)
        return (w[..., None] * x).sum(1)


class Q3Model(nn.Module):
    def __init__(self, hidden=96, dropout=.2, mode='gated', prior=None, mean_score=0.):
        super().__init__()
        self.mode = mode
        self.enc = nn.ModuleList([Temporal(d, hidden, dropout, mode != 'no_temporal') for d in (768, 74, 35)])
        self.cross = nn.MultiheadAttention(hidden, 4, dropout=dropout, batch_first=True)
        self.null = nn.Parameter(torch.zeros(1, 1, hidden))
        self.gate = nn.Sequential(nn.Linear(hidden + 1, 32), nn.GELU(), nn.Linear(32, 1))
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(dropout))
        self.cls = nn.Linear(64, 3); self.reg = nn.Linear(64, 1)
        self.register_buffer('prior', torch.tensor(prior if prior is not None else [1/3]*3, dtype=torch.float32))
        self.register_buffer('mean_score', torch.tensor(float(mean_score)))

    def forward(self, t, a, v, mask):
        mask = mask.clone()
        if self.mode == 'text_only': mask[:, :, 1:] = False
        available = mask.any(1)
        encoded = torch.stack([e(x, mask[:, :, m]) for m, (e, x) in enumerate(zip(self.enc, (t, a, v)))], 1)
        keys = torch.cat([encoded, self.null.expand(len(t), -1, -1)], 1)
        key_pad = torch.cat([~available, torch.zeros(len(t), 1, dtype=torch.bool, device=t.device)], 1)
        cross, _ = self.cross(encoded, keys, keys, key_padding_mask=key_pad, need_weights=False)
        z = (encoded + cross) * available[..., None]
        if self.mode == 'equal':
            weights = available.float() / available.sum(1, keepdim=True).clamp_min(1)
        else:
            # ratio is observable-row occupancy, NOT an asserted physical missing-rate.
            fraction = mask.float().mean(1).unsqueeze(-1)
            g = self.gate(torch.cat([z, fraction], -1)).squeeze(-1).masked_fill(~available, -1e4)
            weights = torch.softmax(g, -1) * available
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        h = self.head((z * weights[..., None]).sum(1))
        logits = self.cls(h); score = 3 * torch.tanh(self.reg(h).squeeze(-1) / 3)
        empty = ~available.any(1)
        logits = torch.where(empty[:, None], self.prior.clamp_min(1e-8).log()[None], logits)
        score = torch.where(empty, self.mean_score, score)
        return logits, score, weights


def tensor_inputs(data, idx, device, text=None, mask=None):
    m = data['mask'][idx] if mask is None else mask
    return tuple(torch.as_tensor(x, device=device) for x in
                 (data['t'][idx] if text is None else text, data['a'][idx], data['v'][idx], m))
