"""Q3 models: conservative control plus per-position cross-modal fusion."""
import torch
from torch import nn

from q3_adaptive_model import FrozenText, Head, LoRALinear
from q3_fusion_model import TunedModel as ResidualFusionModel


class TokenInteractionModel(ResidualFusionModel):
    """Keep the three aligned modalities at each position before pooling.

    This follows the useful part of Q2's architecture while retaining Q3's
    input masks, heads, normalization, portable checkpoints and explanations.
    """

    def __init__(self, base, config, prior, mean_score):
        super().__init__(base, config, prior, mean_score)
        hidden = 128
        drop = float(config['dropout'])
        self.token_projections = nn.ModuleList([
            nn.Sequential(nn.Linear(base.config.hidden_size, hidden), nn.LayerNorm(hidden), nn.GELU()),
            nn.Sequential(nn.Linear(74, hidden), nn.LayerNorm(hidden), nn.GELU()),
            nn.Sequential(nn.Linear(35, hidden), nn.LayerNorm(hidden), nn.GELU()),
        ])
        self.modality_embedding = nn.Parameter(torch.randn(3, hidden) * .02)
        self.position_embedding = nn.Parameter(torch.randn(1, 50, hidden) * .02)
        self.quality_projection = nn.Sequential(nn.Linear(6, hidden), nn.LayerNorm(hidden), nn.GELU())
        layer = nn.TransformerEncoderLayer(
            hidden, 8, hidden * 4, drop, batch_first=True,
            norm_first=True, activation='gelu')
        self.cross_modal = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        temporal_layer = nn.TransformerEncoderLayer(
            hidden, 8, hidden * 4, drop, batch_first=True,
            norm_first=True, activation='gelu')
        self.token_temporal = nn.TransformerEncoder(
            temporal_layer, 2, enable_nested_tensor=False)
        self.token_attention = nn.Linear(hidden, 1)
        self.token_head = Head(hidden, drop, config['hierarchical'])

    def train(self, mode=True):
        # The shared train mode freezes base BERT; restore dropout for any
        # layers intentionally unfrozen by a full-tuning configuration.
        # train mode only for those full-tuned layers.
        super().train(mode)
        if int(self.config.get('full_layers', 0)):
            for layer in self.bert.encoder.layer[-int(self.config['full_layers']):]:
                layer.train(mode)
        for module in self.bert.modules():
            if isinstance(module, LoRALinear):
                module.drop.p = float(self.config['dropout'])
                module.drop.train(mode)
        return self

    def _encode_tokens(self, b, mask):
        observed = mask[:, :, 0]
        special = (b[:, 0] == 101) | (b[:, 0] == 102)
        attention = (observed | special) & b[:, 1].bool()
        ids = b[:, 0].masked_fill(~attention, 0)
        hidden = self.bert(input_ids=ids, attention_mask=attention.long(),
                           token_type_ids=b[:, 2]).last_hidden_state
        return hidden * observed[..., None], observed

    def _token_fusion(self, text_tokens, a, v, mask):
        xs = (text_tokens, a, v)
        h = torch.stack([
            proj(x) * mask[:, :, j, None]
            for j, (proj, x) in enumerate(zip(self.token_projections, xs))
        ], dim=2)
        length = mask.shape[1]
        fractions = mask.float().mean(1)
        quality = torch.cat([mask.float(), fractions[:, None].expand(-1, length, -1)], -1)
        quality_word = self.quality_projection(quality)
        tokens = h.permute(0, 2, 1, 3).reshape(h.shape[0], 3 * length, -1)
        tokens = tokens + self.modality_embedding.repeat_interleave(length, 0)[None]
        position = self.position_embedding[:, :length]
        tokens = tokens + position.repeat(1, 3, 1)
        tokens = tokens + quality_word[:, None].expand(-1, 3, -1, -1).reshape(h.shape[0], 3 * length, -1)
        padding = ~mask.permute(0, 2, 1).reshape(h.shape[0], 3 * length)
        all_hidden = padding.all(1)
        if all_hidden.any():
            padding = padding.clone(); padding[all_hidden, 0] = False
        tokens = self.cross_modal(tokens, src_key_padding_mask=padding)
        words = tokens.reshape(h.shape[0], 3, length, -1).permute(0, 2, 1, 3)
        weights = mask.float() / mask.float().sum(-1, keepdim=True).clamp_min(1)
        words = (words * weights[..., None]).sum(2) + quality_word
        valid = mask.any(-1)
        words = self.token_temporal(words, src_key_padding_mask=~valid)
        pool = valid & mask.any(-1)
        attention = self.token_attention(words).squeeze(-1).masked_fill(~pool, -1e4)
        attention = torch.softmax(attention, 1) * pool
        attention = attention / attention.sum(1, keepdim=True).clamp_min(1e-8)
        return (words * attention[..., None]).sum(1), words

    def forward(self, b, a, v, mask, return_aux=False):
        if self.config.get('fusion') != 'token_interaction':
            return super().forward(b, a, v, mask, return_aux)
        text_tokens, observed_text = self._encode_tokens(b, mask)
        fused, _ = self._token_fusion(text_tokens, a, v, mask)
        logits, regression = self.token_head(fused)
        # Keep the original Q3 empty-input semantics for masking explanations:
        # a coalition with no observed modality uses the learned class prior
        # and training-set regression mean rather than an uncalibrated head.
        empty = ~mask.any((1, 2))
        prior_l = self.prior.clamp_min(1e-8).log()[None]
        logits = torch.where(empty[:, None], prior_l, logits)
        regression = torch.where(empty, self.mean_score, regression)
        # Keep the auxiliary losses used by the training pipeline, with text and A/V
        # summaries computed from the same masked inputs.
        text_mean = text_tokens.sum(1) / observed_text.sum(1, keepdim=True).clamp_min(1)
        text_summary = torch.cat([text_tokens[:, 0], text_mean], -1)
        tl, tr = self.text_head(text_summary)
        av = torch.cat([
            enc(x, mask[:, :, j + 1])
            for j, (enc, x) in enumerate(zip(self.av_enc, (a, v)))
        ], -1)
        al, ar = self.av_head(av)
        if return_aux:
            return logits, regression, tl, 3 * torch.tanh(tr / 3), al, 3 * torch.tanh(ar / 3)
        return logits, regression


def fresh_model(bert_path, config, prior, mean_score, device, verify=False):
    from transformers import AutoModel
    if verify:
        encoder = FrozenText(bert_path, 'cpu')
        base = encoder.model
        fingerprint = encoder.fingerprint
        del encoder
    else:
        base = AutoModel.from_pretrained(bert_path, local_files_only=True, attn_implementation='eager')
        fingerprint = None
    model_cls = TokenInteractionModel if config.get('fusion') == 'token_interaction' else ResidualFusionModel
    model = model_cls(base, config, prior, mean_score).to(device)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.drop.p = float(config['dropout'])
    return model, fingerprint


def candidates(seed):
    conservative = dict(
        rank=8, dropout=.30, head_lr=2e-4, adapter_lr=1e-4, seed=seed,
        lora_layers=4, hierarchical=True, full_layers=0, fusion='residual',
        weight_power=.5, smoothing=.03, reg_weight=.30, text_ce=.20,
        text_reg=.10, av_aux=.05, augmentation=.15)
    token = dict(
        conservative, hierarchical=False, fusion='token_interaction',
        token_layers=2)
    return {
        'v3_control': dict(conservative),
        'token_interaction': dict(token),
        'token_interaction_hierarchical': dict(token, hierarchical=True),
        'token_interaction_light': dict(token, adapter_lr=7.5e-5, augmentation=.20),
    }


# Compatibility name used by the model smoke tests.
TunedModel = TokenInteractionModel
