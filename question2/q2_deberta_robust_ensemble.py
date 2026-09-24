"""Evaluate the calibrated DeBERTa/cross-modal ensemble under missingness."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from q2_crossmodal_full_experiment import CrossModalFusion
from q2_deberta_text_experiment import DebertaSentiment, evaluate, tokenize
from q2_experiment import TextEncoder, metrics, normalize, predict, prepare_split, scenarios, sha
from q2_scattered_gap_experiment import make_scattered_suite, score


def reconstruct_texts(bert_triplets, observed, tokenizer):
    result=[]
    for triplet, keep in zip(bert_triplets,observed):
        ids=np.asarray(triplet[0])[keep]
        result.append(tokenizer.decode(ids.tolist(),skip_special_tokens=True,
                                       clean_up_tokenization_spaces=True))
    return result


def dynamic_blend(cross_p,text_p,base_weight,availability,nb,pb):
    w=(base_weight*np.clip(availability,0,1))[:,None]
    z=(1-w)*np.log(np.clip(cross_p,1e-7,1))+w*np.log(np.clip(text_p,1e-7,1))
    z[:,1]+=nb; z[:,2]+=pb; z-=z.max(1,keepdims=True)
    p=np.exp(z); return p/p.sum(1,keepdims=True),w[:,0]


def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main(args):
    source=Path(args.safe_data).resolve(strict=True)
    with source.open('rb') as f: safe=pickle.load(f)
    enc=TextEncoder(args.device); source_hash=sha(source); cache=Path(args.cache)
    va=prepare_split(safe['valid'],enc,cache,'robust_ensemble_valid',source_hash)
    cross_ck=torch.load(args.cross_checkpoint,map_location=args.device,weights_only=False)
    normalize(va,cross_ck['stats'])
    cross=CrossModalFusion(hidden=128,dropout=.2).to(args.device)
    cross.load_state_dict(cross_ck['model'],strict=False); cross.eval()
    text_tokenizer=AutoTokenizer.from_pretrained(args.deberta_model_path,local_files_only=True)
    bert_tokenizer=AutoTokenizer.from_pretrained(args.bert_model_path,local_files_only=True)
    text=DebertaSentiment(args.deberta_model_path,dropout=.15,train_layers=3).to(args.device)
    text_ck=torch.load(args.deberta_checkpoint,map_location=args.device,weights_only=False)
    text.load_state_dict(text_ck['model']); text.eval()
    saved=np.load(args.deberta_predictions,allow_pickle=False)
    clean_text_p=np.asarray(saved['probability']); clean_text_r=np.asarray(saved['regression'])
    calibration=json.loads(Path(args.calibration).read_text(encoding='utf-8'))['parameters']
    suites=[('contiguous',scenarios(va,enc,cache,cross_ck['stats'],source_hash)),
            ('scattered',make_scattered_suite(va,enc,cache,cross_ck['stats'],source_hash))]
    rows=[]
    valid_count=va['valid'].sum(1).clip(min=1)
    for family,suite in suites:
        for sc in suite:
            cross_p,cross_r=predict(cross,va,args.device,sc)
            availability=sc['mask'][:,:,0].sum(1)/valid_count
            if np.array_equal(sc['mask'][:,:,0],va['mask'][:,:,0]):
                text_p,text_r=clean_text_p,clean_text_r
            else:
                redacted=reconstruct_texts(np.asarray(safe['valid']['text_bert']),sc['mask'][:,:,0],bert_tokenizer)
                ids,att=tokenize(text_tokenizer,redacted,96)
                _,text_p,text_r=evaluate(text,ids,att,va['y'],va['r'],args.device,96)
            p,effective=dynamic_blend(cross_p,text_p,calibration['weight_deberta'],availability,
                                      calibration['neutral_bias'],calibration['positive_bias'])
            pr=(1-effective)*cross_r+effective*text_r
            row={'family':family,'subset':sc['subset'],'rate':float(sc['rate']),
                 'position':sc['position'],'replicate':int(sc['replicate']),
                 'mean_text_availability':float(availability.mean()),
                 'mean_deberta_weight':float(effective.mean()),**metrics(va['y'],va['r'],p,pr)}
            rows.append(row); print(json.dumps(row,ensure_ascii=False),flush=True)
    old=[r for r in rows if r['family']=='contiguous']; sparse=[r for r in rows if r['family']=='scattered']
    result={'selection_score':score(old,sparse),'clean_F1':old[0]['Macro_F1'],'clean_MAE':old[0]['MAE'],
            'contiguous_30pct_mean_F1':float(np.mean([r['Macro_F1'] for r in old[1:]])),
            'contiguous_30pct_mean_MAE':float(np.mean([r['MAE'] for r in old[1:]])),
            'scattered_TAV_mean_F1':float(np.mean([r['Macro_F1'] for r in sparse])),
            'scattered_TAV_mean_MAE':float(np.mean([r['MAE'] for r in sparse])),
            'dynamic_rule':'DeBERTa ensemble weight = calibrated weight * observed text fraction',
            'test_or_attachment3_labels_used':False}
    write_csv(args.out_csv,rows); Path(args.out_json).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('safe_data','cross_checkpoint','deberta_checkpoint','deberta_predictions','calibration','deberta_model_path','bert_model_path','out_csv','out_json'):
        ap.add_argument('--'+name.replace('_','-'),required=True)
    ap.add_argument('--cache',default='work/问题2_特征缓存'); ap.add_argument('--device',default='cuda:1')
    main(ap.parse_args())
