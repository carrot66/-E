"""Train-split calibration and probability blending for Problem 2.

The calibration subset is drawn from the official training split only. The
official validation split remains untouched for model selection reporting.
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error

from q2_experiment import Fusion, TextEncoder, fit_normalization, normalize, predict, prepare_split, scenarios, sha
from q2_crossmodal_full_experiment import CrossModalFusion
from q2_scattered_gap_experiment import make_scattered_suite


def load_model(path, cls, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = cls().to(device); model.load_state_dict(ck["model"], strict=False); model.eval(); return model, ck


def calibrate(p_cross, p_student, y, seed=2033):
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(y), size=max(1000, int(.2 * len(y))), replace=False)
    yy = y[ids]; best = None
    # Blend and class-logit bias are fit only on this held-out slice of train.
    for alpha in np.linspace(0, 1, 21):
        p = alpha * p_cross[ids] + (1 - alpha) * p_student[ids]
        logp = np.log(np.clip(p, 1e-8, 1))
        for b0 in np.linspace(-.5, .5, 21):
            for b1 in np.linspace(-.5, .5, 21):
                bias = np.array([b0, b1, 0.0])
                pred = np.argmax(logp + bias, axis=1)
                val = f1_score(yy, pred, average="macro", zero_division=0)
                if best is None or val > best["calibration_macro_f1"]:
                    best = {"alpha_cross_modal": float(alpha), "class_bias": bias.tolist(), "calibration_macro_f1": float(val), "calibration_n": int(len(ids))}
    return best


def apply(p_cross, r_cross, p_student, r_student, params):
    a = params["alpha_cross_modal"]; bias = np.asarray(params["class_bias"])
    p = a * p_cross + (1-a) * p_student
    pred = np.argmax(np.log(np.clip(p, 1e-8, 1)) + bias, axis=1)
    r = a * r_cross + (1-a) * r_student
    return p, pred, r


def summarize(y, r, p, pred, reg):
    return {"Accuracy": float(accuracy_score(y, pred)), "Macro_F1": float(f1_score(y, pred, average="macro", zero_division=0)), "MAE": float(mean_absolute_error(r, reg))}


def main(args):
    device=args.device; out=Path(args.out); out.mkdir(parents=True, exist_ok=True); source=Path(args.safe_data).resolve(); source_hash=sha(source)
    with source.open("rb") as f: safe=pickle.load(f)
    student_path=Path(args.student).resolve(); cross_path=Path(args.cross).resolve()
    student, student_ck=load_model(student_path, lambda: Fusion(gated=True), device)
    cross, cross_ck=load_model(cross_path, CrossModalFusion, device)
    enc=TextEncoder(device); cache=Path(args.cache)
    train=prepare_split(safe["train"], enc, cache, "cal_train", source_hash); valid=prepare_split(safe["valid"], enc, cache, "cal_valid", source_hash)
    stats=student_ck["stats"]; normalize(train,stats); normalize(valid,stats)
    psc,rsc=predict(student,train,device); pcc,rcc=predict(cross,train,device)
    params=calibrate(pcc,psc,train["y"]); params.update({"source_sha256":source_hash,"cross_checkpoint_sha256":sha(cross_path),"student_checkpoint_sha256":sha(student_path),"fit_policy":"20% random calibration slice from official train only"})
    (out/"问题2_校准融合参数.json").write_text(json.dumps(params,ensure_ascii=False,indent=2),encoding="utf-8")
    pscv,rscv=predict(student,valid,device); pccv,rccv=predict(cross,valid,device); p, pred, reg=apply(pccv,rccv,pscv,rscv,params)
    clean=summarize(valid["y"],valid["r"],p,pred,reg)
    quick=scenarios(valid,enc,cache,stats,source_hash)
    rows=[]; preds=[]
    for sc in quick:
        a,b=predict(student,valid,device,sc); c,d=predict(cross,valid,device,sc); pp,pr,rr=apply(c,d,a,b,params); row={"subset":sc["subset"],"rate":sc["rate"],"position":sc["position"],**summarize(valid["y"],valid["r"],pp,pr,rr)}; rows.append(row)
    sparse=make_scattered_suite(valid,enc,cache,stats,source_hash); sparse_rows=[]
    for sc in sparse:
        a,b=predict(student,valid,device,sc); c,d=predict(cross,valid,device,sc); pp,pr,rr=apply(c,d,a,b,params); sparse_rows.append({"subset":sc["subset"],"rate":sc["rate"],"position":sc["position"],**summarize(valid["y"],valid["r"],pp,pr,rr)})
    clean_row=rows[0]; old=rows[1:]; new=sparse_rows; old_f1=np.mean([x["Macro_F1"] for x in old]); old_mae=np.mean([x["MAE"] for x in old]); new_f1=np.mean([x["Macro_F1"] for x in new]); new_mae=np.mean([x["MAE"] for x in new]); selection=.4*clean_row["Macro_F1"]+.3*old_f1+.3*new_f1-.1*(.4*clean_row["MAE"]+.3*old_mae+.3*new_mae)
    conclusion={"calibrated_profile":{"selection_score":float(selection),"clean":clean_row,"contiguous_30pct_mean_F1":float(old_f1),"contiguous_30pct_mean_MAE":float(old_mae),"scattered_TAV_mean_F1":float(new_f1),"scattered_TAV_mean_MAE":float(new_mae)},"target_65":bool(clean_row["Macro_F1"]>=.65),"target_70":bool(clean_row["Macro_F1"]>=.70),"test_or_attachment3_labels_used":False,"model_saved":True,"postprocess":"probability blend + train-split class-logit calibration"}
    (out/"问题2_校准融合实验结论.json").write_text(json.dumps(conclusion,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(conclusion,ensure_ascii=False,indent=2))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--safe-data",required=True); ap.add_argument("--student",required=True); ap.add_argument("--cross",required=True); ap.add_argument("--out",required=True); ap.add_argument("--cache",required=True); ap.add_argument("--device",default="cuda:1"); main(ap.parse_args())
