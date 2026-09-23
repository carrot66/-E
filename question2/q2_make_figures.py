"""Generate graphical Problem-2 diagnostics from saved CSV/JSON records."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f: return list(csv.DictReader(f))
def f(row, key): return float(row[key])
def style():
    plt.rcParams.update({"font.family":"DejaVu Sans","figure.dpi":140,"savefig.dpi":220,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.25})
def save(fig, path): fig.tight_layout(); fig.savefig(path,bbox_inches="tight"); plt.close(fig)

def main(root):
    root=Path(root).resolve(); best=root/"问题2_跨模态交互完整优化"; q1=root/"问题2_Q1对齐质量融合完整优化"; fine=root/"问题2_跨模态交互精调96"; large=root/"问题2_跨模态交互大模型192"; legacy=root.parent/"问题2_完整MOSEI训练实验"; out=best/"plots"; out.mkdir(parents=True,exist_ok=True); style()
    training_dir=best
    candidate=root/"问题2_跨模态交互类别重权重"
    if (candidate/"问题2_跨模态交互训练记录.csv").exists(): training_dir=candidate
    train=read_csv(training_dir/"问题2_跨模态交互训练记录.csv"); ep=np.array([f(r,"epoch") for r in train]); loss=np.array([f(r,"train_loss") for r in train]); score=np.array([f(r,"selection_score") for r in train]); clean=np.array([f(r,"clean_F1") for r in train]); contig=np.array([f(r,"contiguous_30pct_mean_F1") for r in train]); sparse=np.array([f(r,"scattered_TAV_mean_F1") for r in train])
    fig,ax=plt.subplots(2,2,figsize=(13,9)); ax[0,0].plot(ep,loss,"o-",lw=2.5,color="#264653"); ax[0,0].set(title="Training loss",xlabel="Epoch",ylabel="Loss"); ax[0,1].plot(ep,score,"o-",lw=2.5,label="Selection score"); ax[0,1].plot(ep,clean,"s-",label="Clean F1"); ax[0,1].plot(ep,contig,"^-",label="Contiguous F1"); ax[0,1].plot(ep,sparse,"d-",label="Scattered TAV F1"); ax[0,1].set(title="F1 and selection score",xlabel="Epoch",ylabel="Score"); ax[0,1].legend(fontsize=8)
    acc_keys=[k for k in ("clean_Accuracy","contiguous_30pct_mean_Accuracy","scattered_TAV_mean_Accuracy") if k in train[0]]
    if acc_keys:
        names={"clean_Accuracy":"Clean Acc","contiguous_30pct_mean_Accuracy":"Contiguous Acc","scattered_TAV_mean_Accuracy":"Scattered TAV Acc"}
        for k in acc_keys: ax[1,0].plot(ep,[f(r,k) for r in train],"o-",label=names[k])
        ax[1,0].legend(fontsize=8)
    else: ax[1,0].text(.5,.5,"Accuracy will be recorded by the updated training script",ha="center",va="center")
    ax[1,0].set(title="Accuracy by epoch",xlabel="Epoch",ylabel="Accuracy"); ax[1,1].plot(ep,[f(r,"clean_MAE") for r in train],"o-",label="Clean MAE"); ax[1,1].plot(ep,[f(r,"contiguous_30pct_mean_MAE") for r in train],"s-",label="Contiguous MAE"); ax[1,1].plot(ep,[f(r,"scattered_TAV_mean_MAE") for r in train],"d-",label="Scattered TAV MAE"); ax[1,1].set(title="Regression MAE by epoch",xlabel="Epoch",ylabel="MAE"); ax[1,1].legend(fontsize=8); save(fig,out/"问题2_逐轮训练_loss_Accuracy_F1_MAE.png")
    entries=[("原完整MOSEI学生",legacy/"完整数据_验证选模结论.json","student_full_valid"),("跨模态交互",best/"问题2_跨模态交互实验结论.json","cross_modal_profile"),("类别重权重",root/"问题2_跨模态交互类别重权重"/"问题2_跨模态交互实验结论.json","cross_modal_profile"),("问题一对齐质量",q1/"问题2_Q1对齐质量融合实验结论.json","cross_modal_profile"),("hidden96精调",fine/"问题2_跨模态交互实验结论.json","cross_modal_profile")]
    if (large/"问题2_跨模态交互实验结论.json").exists(): entries.append(("hidden192",large/"问题2_跨模态交互实验结论.json","cross_modal_profile"))
    names=[]; matrix=[]
    for name,path,key in entries:
        if Path(path).exists(): p=read_json(path)[key]; names.append(name); matrix.append([p["clean_F1"],p["contiguous_30pct_mean_F1"],p["scattered_TAV_mean_F1"]])
    matrix=np.array(matrix); x=np.arange(len(names)); fig,ax=plt.subplots(figsize=(12,6)); w=.24
    for j,label in enumerate(("Clean","Contiguous","Scattered TAV")): ax.bar(x+(j-1)*w,matrix[:,j],w,label=label)
    ax.set_xticks(x,names,rotation=15); ax.set_ylim(.5,.73); ax.set_ylabel("Macro-F1"); ax.set_title("Model robustness comparison"); ax.legend(fontsize=9,ncol=3); save(fig,out/"问题2_模型鲁棒性对比.png")
    rows=read_csv(best/"问题2_跨模态交互最佳验证场景.csv"); order=["none","T","A","V","AV","TAV"]; labels=["Clean","Text","Audio","Vision","Audio+Vision","All"]; quick=[]
    for sub in order:
        z=[r for r in rows if (r["subset"]==sub and (sub=="none" or (r["position"]=="random" and abs(f(r,"rate")-.3)<1e-6)))]; quick.append(z[0] if z else {"Macro_F1":0,"MAE":0})
    fig,ax=plt.subplots(1,2,figsize=(14,5)); colors=["#264653","#2a9d8f","#e9c46a","#f4a261","#e76f51","#d62828"]; ax[0].bar(labels,[f(r,"Macro_F1") for r in quick],color=colors); ax[0].set_ylim(.5,.73); ax[0].set_ylabel("Macro-F1"); ax[0].set_title("30% contiguous missing ablation"); ax[0].tick_params(axis="x",rotation=20); ax[1].bar(labels,[f(r,"MAE") for r in quick],color=colors); ax[1].set_ylabel("MAE (lower is better)"); ax[1].set_title("Regression error in the same ablation"); ax[1].tick_params(axis="x",rotation=20); save(fig,out/"问题2_缺失模态消融.png")
    sparse_rows=[r for r in rows if r["subset"]=="TAV" and r["position"]=="scattered"]; rates=sorted(set(f(r,"rate") for r in sparse_rows)); means=[]; stds=[]; maes=[]; mae_stds=[]
    for rate in rates:
        z=np.array([f(r,"Macro_F1") for r in sparse_rows if abs(f(r,"rate")-rate)<1e-6]); m=np.array([f(r,"MAE") for r in sparse_rows if abs(f(r,"rate")-rate)<1e-6]); means.append(z.mean()); stds.append(z.std()); maes.append(m.mean()); mae_stds.append(m.std())
    x=np.array(rates)*100; fig,ax=plt.subplots(1,2,figsize=(13,5)); ax[0].plot(x,means,"o-",lw=2.5,color="#2a9d8f"); ax[0].fill_between(x,np.array(means)-stds,np.array(means)+stds,alpha=.2,color="#2a9d8f"); ax[0].set(xlabel="Shared TAV gaps (%)",ylabel="Macro-F1",title="Scattered missingness robustness"); ax[1].plot(x,maes,"o-",lw=2.5,color="#e76f51"); ax[1].fill_between(x,np.array(maes)-mae_stds,np.array(maes)+mae_stds,alpha=.2,color="#e76f51"); ax[1].set(xlabel="Shared TAV gaps (%)",ylabel="MAE",title="Error growth with scattered gaps"); save(fig,out/"问题2_散点缺失率曲线.png")
    fig,ax=plt.subplots(figsize=(8,6)); ax.scatter([f(r,"MAE") for r in rows],[f(r,"Macro_F1") for r in rows],c=["#e76f51" if r["position"]=="scattered" else "#457b9d" for r in rows],s=50,alpha=.8,edgecolor="white"); ax.set(xlabel="MAE (lower is better)",ylabel="Macro-F1",title="F1–MAE trade-off across scenarios"); save(fig,out/"问题2_F1与MAE权衡.png")
    manifest={"figure_count":5,"files":sorted(p.name for p in out.glob("*.png")),"test_labels_used":False}; (out/"问题2_绘图说明.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); main(ap.parse_args().root)
