"""Create normalized confusion matrices for the selected Q2 models."""
from __future__ import annotations
import argparse, pickle
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from q2_experiment import Fusion, TextEncoder, normalize, predict, prepare_split, sha
from q2_crossmodal_full_experiment import CrossModalFusion
from q2_figure_style import use_chinese_font

def main(args):
    use_chinese_font()
    device=args.device; source=Path(args.safe_data).resolve()
    with source.open("rb") as f: safe=pickle.load(f)
    student_ck=torch.load(args.student,map_location=device,weights_only=False); student=Fusion(gated=True).to(device); student.load_state_dict(student_ck["model"]); student.eval()
    cross_ck=torch.load(args.cross,map_location=device,weights_only=False); cross=CrossModalFusion().to(device); cross.load_state_dict(cross_ck["model"], strict=False); cross.eval()
    enc=TextEncoder(device); split=prepare_split(safe["valid"],enc,Path(args.cache),"cm_valid",sha(source)); normalize(split,student_ck["stats"])
    y=split["y"]; ps,_=predict(student,split,device); pc,_=predict(cross,split,device)
    mats=[confusion_matrix(y,ps.argmax(1),labels=[0,1,2],normalize="true"),confusion_matrix(y,pc.argmax(1),labels=[0,1,2],normalize="true")]
    fig,ax=plt.subplots(1,2,figsize=(11,4.7),layout="constrained"); names=["原完整数据学生模型","跨模态交互模型"]
    fig.suptitle("两种模型在验证集上的类别识别对比",fontsize=12,fontweight="bold")
    for a,m,n in zip(ax,mats,names):
        im=a.imshow(m,vmin=0,vmax=1,cmap="Blues"); a.set_xticks([0,1,2],["负向","中性","正向"]); a.set_yticks([0,1,2],["负向","中性","正向"]); a.set_xlabel("预测类别"); a.set_ylabel("真实类别"); a.set_title(n)
        for i in range(3):
            for j in range(3): a.text(j,i,f"{m[i,j]:.2f}",ha="center",va="center",color="white" if m[i,j]>.55 else "black")
    fig.colorbar(im,ax=ax.ravel().tolist(),location="right",fraction=.036,pad=.05,shrink=.85,label="行归一化比例")
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=300,bbox_inches="tight"); plt.close(fig); print(out)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--safe-data",required=True); ap.add_argument("--student",required=True); ap.add_argument("--cross",required=True); ap.add_argument("--cache",required=True); ap.add_argument("--out",required=True); ap.add_argument("--device",default="cuda:1"); main(ap.parse_args())
