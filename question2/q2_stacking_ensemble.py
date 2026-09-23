"""Train-split stacking of independently trained Q2 models."""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from q2_experiment import Fusion, TextEncoder, normalize, predict, prepare_split, scenarios, sha
from q2_crossmodal_full_experiment import CrossModalFusion
from q2_q1_alignment_quality_experiment import AlignmentAwareFusion
from q2_scattered_gap_experiment import make_scattered_suite

def load(path, kind, device):
    ck=torch.load(path,map_location=device,weights_only=False); cls={"student":lambda: Fusion(gated=True),"cross":CrossModalFusion,"q1":AlignmentAwareFusion}[kind]; model=cls().to(device); model.load_state_dict(ck["model"], strict=False); model.eval(); return model,ck
def feat(ps,rs):
    return np.concatenate([*[np.log(np.clip(p,1e-8,1)) for p in ps],*[r[:,None] for r in rs]],axis=1)
def predict_all(models,split,device,sc=None):
    ps=[]; rs=[]
    for m in models:
        p,r=predict(m,split,device,sc); ps.append(p); rs.append(r)
    return feat(ps,rs),ps,rs
def metric(y,r,pred): return {"Accuracy":float(accuracy_score(y,pred)),"Macro_F1":float(f1_score(y,pred,average="macro",zero_division=0)),"MAE":float(mean_absolute_error(r,r))}
def main(a):
    device=a.device; src=Path(a.safe_data).resolve(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    with src.open("rb") as f: safe=pickle.load(f)
    paths=[(Path(a.student).resolve(),"student"),(Path(a.cross).resolve(),"cross"),(Path(a.q1).resolve(),"q1"),(Path(a.reweight).resolve(),"cross")]
    models=[]; checks=[]
    for p,k in paths:
        m,c=load(p,k,device); models.append(m); checks.append(c)
    enc=TextEncoder(device); cache=Path(a.cache); train=prepare_split(safe["train"],enc,cache,"stack_train",sha(src)); valid=prepare_split(safe["valid"],enc,cache,"stack_valid",sha(src)); normalize(train,checks[0]["stats"]); normalize(valid,checks[0]["stats"])
    Xtr,_,rtr=predict_all(models,train,device); Xva,_,rva=predict_all(models,valid,device)
    rng=np.random.default_rng(2035); cal=rng.choice(len(Xtr),size=max(3000,int(.25*len(Xtr))),replace=False)
    stack=LogisticRegression(C=a.C,max_iter=600,class_weight="balanced"); stack.fit(Xtr[cal],train["y"][cal]); pv=stack.predict_proba(Xva); pred=pv.argmax(1); reg=np.mean(np.stack(rva),axis=0)
    clean={"Accuracy":float(accuracy_score(valid["y"],pred)),"Macro_F1":float(f1_score(valid["y"],pred,average="macro",zero_division=0)),"MAE":float(mean_absolute_error(valid["r"],reg))}
    quick=scenarios(valid,enc,cache,checks[0]["stats"],sha(src)); old=[]
    for sc in quick:
        X,_,rr=predict_all(models,valid,device,sc); pp=stack.predict_proba(X); pr=pp.argmax(1); rg=np.mean(np.stack(rr),axis=0); old.append({"subset":sc["subset"],"rate":sc["rate"],"position":sc["position"],"Accuracy":float(accuracy_score(valid["y"],pr)),"Macro_F1":float(f1_score(valid["y"],pr,average="macro",zero_division=0)),"MAE":float(mean_absolute_error(valid["r"],rg))})
    sparse=make_scattered_suite(valid,enc,cache,checks[0]["stats"],sha(src)); new=[]
    for sc in sparse:
        X,_,rr=predict_all(models,valid,device,sc); pp=stack.predict_proba(X); pr=pp.argmax(1); rg=np.mean(np.stack(rr),axis=0); new.append({"subset":sc["subset"],"rate":sc["rate"],"position":sc["position"],"Accuracy":float(accuracy_score(valid["y"],pr)),"Macro_F1":float(f1_score(valid["y"],pr,average="macro",zero_division=0)),"MAE":float(mean_absolute_error(valid["r"],rg))})
    old_f1=np.mean([x["Macro_F1"] for x in old[1:]]); old_mae=np.mean([x["MAE"] for x in old[1:]]); new_f1=np.mean([x["Macro_F1"] for x in new]); new_mae=np.mean([x["MAE"] for x in new]); score=.4*old[0]["Macro_F1"]+.3*old_f1+.3*new_f1-.1*(.4*old[0]["MAE"]+.3*old_mae+.3*new_mae)
    result={"selection_score":float(score),"clean":clean,"contiguous_30pct_mean_F1":float(old_f1),"contiguous_30pct_mean_MAE":float(old_mae),"scattered_TAV_mean_F1":float(new_f1),"scattered_TAV_mean_MAE":float(new_mae),"calibration_n":int(len(cal)),"C":a.C,"test_or_attachment3_labels_used":False}
    (out/"问题2_堆叠集成实验结论.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); (out/"问题2_堆叠集成固定场景.csv").write_text("\n".join([",".join(map(str,r.values())) for r in old+new]),encoding="utf-8"); print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--safe-data",required=True); p.add_argument("--student",required=True); p.add_argument("--cross",required=True); p.add_argument("--q1",required=True); p.add_argument("--reweight",required=True); p.add_argument("--cache",required=True); p.add_argument("--out",required=True); p.add_argument("--device",default="cuda:1"); p.add_argument("--C",type=float,default=.2); main(p.parse_args())
