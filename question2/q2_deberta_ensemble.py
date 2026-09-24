"""Video-group calibration of DeBERTa and the selected cross-modal model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, recall_score

from q2_crossmodal_full_experiment import CrossModalFusion
from q2_experiment import TextEncoder, normalize, predict, prepare_split, sha


def score(y,r,p,pr):
    pred=p.argmax(1)
    return {'Accuracy':float(accuracy_score(y,pred)),
            'Macro_F1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'Weighted_F1':float(f1_score(y,pred,average='weighted',zero_division=0)),
            'MAE':float(mean_absolute_error(r,pr))}


def blend(a,b,w,neutral_bias=0.0,positive_bias=0.0):
    # Geometric/logit blend is less sensitive to one overconfident member.
    z=(1-w)*np.log(np.clip(a,1e-7,1))+w*np.log(np.clip(b,1e-7,1))
    z[:,1]+=neutral_bias; z[:,2]+=positive_bias
    z-=z.max(1,keepdims=True); p=np.exp(z); return p/p.sum(1,keepdims=True)


def write_predictions(path, ids, y, r, p, pr):
    names=['Negative','Neutral','Positive']; rows=[]
    pred=p.argmax(1)
    for i,sample_id in enumerate(ids):
        rows.append({'sample_id':sample_id,'video_id':sample_id.split('$_$',1)[0],
                     'true_class':names[int(y[i])],'predicted_class':names[int(pred[i])],
                     'correct':bool(pred[i]==y[i]),'negative_probability':float(p[i,0]),
                     'neutral_probability':float(p[i,1]),'positive_probability':float(p[i,2]),
                     'true_intensity':float(r[i]),'predicted_intensity':float(pr[i]),
                     'absolute_error':float(abs(r[i]-pr[i]))})
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main(args):
    source=Path(args.safe_data).resolve(strict=True)
    with source.open('rb') as f: safe=pickle.load(f)
    enc=TextEncoder(args.device); source_hash=sha(source)
    va=prepare_split(safe['valid'],enc,Path(args.cache),'ensemble_valid',source_hash)
    ck=torch.load(args.cross_checkpoint,map_location=args.device,weights_only=False)
    normalize(va,ck['stats'])
    cross=CrossModalFusion(hidden=128,dropout=.2).to(args.device)
    cross.load_state_dict(ck['model'],strict=False); cross.eval()
    cross_p,cross_r=predict(cross,va,args.device)
    db=np.load(args.deberta_predictions,allow_pickle=False)
    db_ids=list(map(str,db['ids'])); valid_ids=list(map(str,va['ids']))
    if db_ids!=valid_ids or not np.array_equal(db['y'],va['y']): raise ValueError('prediction ID/label mismatch')
    db_p=np.asarray(db['probability']); db_r=np.asarray(db['regression'])
    groups=np.asarray([x.split('$_$',1)[0] for x in valid_ids])
    calibration=np.asarray([int(hashlib.sha256(g.encode()).hexdigest()[:8],16)%2==0 for g in groups])
    holdout=~calibration
    if set(groups[calibration]) & set(groups[holdout]): raise AssertionError('video split leakage')
    best=None
    # All parameters are selected only on calibration video groups.
    for w in np.linspace(0,1,41):
        for nb in np.linspace(-.12,.12,13):
            for pb in np.linspace(-.12,.12,13):
                p=blend(cross_p,db_p,float(w),float(nb),float(pb))
                f=float(f1_score(va['y'][calibration],p[calibration].argmax(1),average='macro',zero_division=0))
                if best is None or f>best['calibration_F1']:
                    best={'weight_deberta':float(w),'neutral_bias':float(nb),'positive_bias':float(pb),'calibration_F1':f}
    p=blend(cross_p,db_p,best['weight_deberta'],best['neutral_bias'],best['positive_bias'])
    pr=(1-best['weight_deberta'])*cross_r+best['weight_deberta']*db_r
    pred=p.argmax(1); matrix=confusion_matrix(va['y'],pred,labels=[0,1,2])
    result={'selection_protocol':'weight and class biases selected on hash-split calibration video groups only',
            'calibration_samples':int(calibration.sum()),'holdout_samples':int(holdout.sum()),
            'calibration_videos':len(set(groups[calibration])),'holdout_videos':len(set(groups[holdout])),
            'video_overlap':0,'parameters':best,
            'cross_modal_full':score(va['y'],va['r'],cross_p,cross_r),
            'deberta_full':score(va['y'],va['r'],db_p,db_r),
            'ensemble_calibration':score(va['y'][calibration],va['r'][calibration],p[calibration],pr[calibration]),
            'ensemble_holdout':score(va['y'][holdout],va['r'][holdout],p[holdout],pr[holdout]),
            'ensemble_full_fixed_parameters':score(va['y'],va['r'],p,pr),
            'confusion_matrix_rows_true_columns_predicted':matrix.tolist(),
            'per_class_recall':dict(zip(['Negative','Neutral','Positive'],
                map(float,recall_score(va['y'],pred,labels=[0,1,2],average=None,zero_division=0)))),
            'test_or_attachment3_labels_used':False}
    Path(args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.predictions_csv:
        write_predictions(args.predictions_csv,valid_ids,va['y'],va['r'],p,pr)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--safe-data',required=True); ap.add_argument('--cross-checkpoint',required=True)
    ap.add_argument('--deberta-predictions',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--predictions-csv')
    ap.add_argument('--cache',default='work/问题2_特征缓存'); ap.add_argument('--device',default='cuda:1')
    main(ap.parse_args())
