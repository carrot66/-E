"""Probe validation-only logit bias calibration for the selected Q2 model.

This is an analysis tool: it never changes the saved model and is not used to
claim a generalization gain.  It reports whether the Macro-F1 ceiling is a
decision-threshold issue or a representation issue.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from q2_crossmodal_full_experiment import CrossModalFusion
from q2_experiment import TextEncoder, fit_normalization, metrics, normalize, predict, prepare_split, sha, tensor_batch
from q2_scattered_gap_experiment import make_scattered_suite


def main(args):
    device = args.device
    source = Path(args.safe_data).resolve(strict=True)
    checkpoint_path = Path(args.checkpoint).resolve(strict=True)
    with source.open('rb') as f:
        safe = pickle.load(f)
    enc = TextEncoder(device)
    cache = Path(args.cache)
    source_hash = sha(source)
    tr = prepare_split(safe['train'], enc, cache, 'threshold_train', source_hash)
    va = prepare_split(safe['valid'], enc, cache, 'threshold_valid', source_hash)
    stats = fit_normalization(tr)
    normalize(tr, stats); normalize(va, stats)
    model = CrossModalFusion(hidden=128, dropout=.2).to(device)
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'], strict=False)
    model.eval()
    suite = make_scattered_suite(va, enc, cache, stats, source_hash)
    # Clean plus a representative 30% all-modality gap; this probe is meant
    # to diagnose threshold sensitivity, not to replace the pre-registered
    # multi-scenario selection rule.
    scenarios = [
        {'name':'clean', 'mask':va['mask'], 't':None},
        {'name':'TAV30', 'mask':suite[2]['mask'], 't':suite[2].get('t')},
    ]
    probs = []; regs=[]
    for sc in scenarios:
        p,r = predict(model, va, device, sc)
        probs.append(p); regs.append(r)
    # Grid a shared neutral-vs-polar bias and a small polarity bias.
    best = None
    for neutral_bias in np.linspace(-.35, .35, 29):
        for polarity_bias in np.linspace(-.20, .20, 17):
            values=[]
            for p,r in zip(probs, regs):
                logits=np.log(np.clip(p,1e-8,1.0))
                logits=logits.copy(); logits[:,1]+=neutral_bias
                logits[:,2]+=polarity_bias
                q=np.exp(logits-logits.max(1,keepdims=True)); q/=q.sum(1,keepdims=True)
                values.append(metrics(va['y'], va['r'], q, r))
            score=.5*values[0]['Macro_F1']+.5*values[1]['Macro_F1']-.1*(values[0]['MAE']+values[1]['MAE'])/2
            if best is None or score>best['score']:
                best={'score':float(score),'neutral_bias':float(neutral_bias),'polarity_bias':float(polarity_bias),'clean':values[0],'TAV30':values[1]}
    base=[]
    for p,r in zip(probs,regs): base.append(metrics(va['y'],va['r'],p,r))
    print(json.dumps({'checkpoint':str(checkpoint_path),'base':base,'best_bias_probe':best},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--safe-data', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--cache', default='work/问题2_特征缓存')
    ap.add_argument('--device', default='cuda:1')
    main(ap.parse_args())
