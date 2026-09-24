"""Publication-style figures for the selected Q2 DeBERTa ensemble.

Conclusion: a video-group-calibrated DeBERTa/cross-modal ensemble improves
clean sentiment classification while retaining gains under missing modalities.
Source data are the complete per-epoch and per-scenario CSV files; no rows are
discarded.  All figures use the validation split (n=1,871 segments).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from q2_figure_style import use_chinese_font

use_chinese_font()
mpl.rcParams.update({
    'font.size':7,
    'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.8,
    'legend.frameon':False,'xtick.major.width':.7,'ytick.major.width':.7,
})

BLUE='#4472A8'; ORANGE='#E69F5B'; GREEN='#5A9A78'; GRAY='#A7ABB2'; RED='#C65D58'


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))


def export(fig,out,stem):
    out.mkdir(parents=True,exist_ok=True)
    svg=out/f'{stem}.svg'
    fig.savefig(svg,bbox_inches='tight')
    svg.write_text(re.sub(r'[ \t]+(?=\r?$)', '', svg.read_text(encoding='utf-8'),
                          flags=re.MULTILINE), encoding='utf-8')
    fig.savefig(out/f'{stem}.pdf',bbox_inches='tight')
    fig.savefig(out/f'{stem}.png',dpi=600,bbox_inches='tight')
    fig.savefig(out/f'{stem}.tiff',dpi=600,bbox_inches='tight')
    plt.close(fig)


def panel(ax,label):
    ax.text(-.13,1.06,label,transform=ax.transAxes,fontweight='bold',fontsize=8,va='top')


def training_figure(rows,out):
    epoch=np.array([int(x['epoch']) for x in rows]); loss=np.array([float(x['train_loss']) for x in rows])
    acc=np.array([float(x['Accuracy']) for x in rows]); f1=np.array([float(x['Macro_F1']) for x in rows])
    mae=np.array([float(x['MAE']) for x in rows]); sel=np.array([float(x['selection_score']) for x in rows])
    best=int(np.argmax(sel))
    fig,axs=plt.subplots(2,2,figsize=(7.2,4.8),constrained_layout=True)
    fig.suptitle('训练过程中分类性能与回归误差的变化',fontsize=9,fontweight='bold')
    axs[0,0].plot(epoch,loss,'o-',color=GRAY,lw=1.5,ms=3); axs[0,0].set(ylabel='训练损失',xlabel='训练轮次')
    axs[0,1].plot(epoch,acc,label='准确率',color=BLUE,lw=1.5); axs[0,1].plot(epoch,f1,label='宏平均F1',color=ORANGE,lw=1.5)
    axs[0,1].scatter(epoch[best],f1[best],s=28,color=RED,zorder=3); axs[0,1].legend(); axs[0,1].set(ylabel='评价指标',xlabel='训练轮次')
    axs[1,0].plot(epoch,mae,'o-',color=GREEN,lw=1.5,ms=3); axs[1,0].set(ylabel='平均绝对误差',xlabel='训练轮次')
    axs[1,1].plot(epoch,sel,'o-',color=BLUE,lw=1.5,ms=3); axs[1,1].scatter(epoch[best],sel[best],s=28,color=RED,zorder=3)
    axs[1,1].annotate(f'最优第{epoch[best]}轮',xy=(epoch[best],sel[best]),xytext=(5,8),textcoords='offset points',fontsize=7)
    axs[1,1].set(ylabel='选模分数',xlabel='训练轮次')
    for a in axs.ravel(): a.grid(axis='y',color='#E8EAED',lw=.6)
    export(fig,out,'问题2_DeBERTa逐轮训练曲线')


def comparison_figure(old,new,out):
    labels=['完整输入','连续缺失30%','三模态散点缺失']
    old_values=[old['clean_F1'],old['contiguous_30pct_mean_F1'],old['scattered_TAV_mean_F1']]
    new_values=[new['clean_F1'],new['contiguous_30pct_mean_F1'],new['scattered_TAV_mean_F1']]
    x=np.arange(3); width=.34
    fig,ax=plt.subplots(figsize=(7.2,3.35),constrained_layout=True)
    fig.suptitle('完整输入与模态缺失条件下的分类性能对比',fontsize=9,fontweight='bold')
    b1=ax.bar(x-width/2,old_values,width,color=GRAY,label='原跨模态模型')
    b2=ax.bar(x+width/2,new_values,width,color=BLUE,label='当前自适应融合模型')
    for bars in (b1,b2):
        for b in bars: ax.text(b.get_x()+b.get_width()/2,b.get_height()+.006,f'{b.get_height():.3f}',ha='center',fontsize=7)
    ax.set_xticks(x,labels); ax.set_ylabel('宏平均F1'); ax.set_ylim(.54,.71); ax.legend(ncol=2,loc='upper right')
    ax.grid(axis='y',color='#E8EAED',lw=.6)
    export(fig,out,'问题2_新旧模型鲁棒性对比')


def scenario_figures(rows,out):
    contiguous=[x for x in rows if x['family']=='contiguous']
    order=['none','T','A','V','AV','TAV']; label={'none':'完整输入','T':'文本','A':'语音','V':'视觉','AV':'语音+视觉','TAV':'三模态'}
    values=[]
    for key in order:
        row=next(x for x in contiguous if x['subset']==key); values.append(float(row['Macro_F1']))
    fig,ax=plt.subplots(figsize=(7.2,3.5),constrained_layout=True)
    fig.suptitle('不同模态缺失类型对分类性能的影响',fontsize=9,fontweight='bold')
    bars=ax.bar(np.arange(len(order)),values,color=[GREEN]+[BLUE]*5,width=.66)
    for b,v in zip(bars,values): ax.text(b.get_x()+b.get_width()/2,v+.006,f'{v:.3f}',ha='center',fontsize=7)
    ax.set_xticks(np.arange(len(order)),[label[x] for x in order],rotation=15,ha='right'); ax.set_ylabel('宏平均F1'); ax.set_ylim(.56,.71)
    ax.grid(axis='y',color='#E8EAED',lw=.6)
    export(fig,out,'问题2_缺失模态类型影响')

    scattered=[x for x in rows if x['family']=='scattered']; rates=sorted({float(x['rate']) for x in scattered})
    means=[np.mean([float(x['Macro_F1']) for x in scattered if float(x['rate'])==r]) for r in rates]
    lo=[min(float(x['Macro_F1']) for x in scattered if float(x['rate'])==r) for r in rates]
    hi=[max(float(x['Macro_F1']) for x in scattered if float(x['rate'])==r) for r in rates]
    fig,ax=plt.subplots(figsize=(7.2,3.45),constrained_layout=True)
    fig.suptitle('三模态共同散点缺失率与分类性能',fontsize=9,fontweight='bold')
    ax.plot(np.array(rates)*100,means,'o-',color=BLUE,lw=1.8,ms=5)
    ax.fill_between(np.array(rates)*100,lo,hi,color=BLUE,alpha=.16,label='两次固定随机遮挡的范围')
    for x,y in zip(np.array(rates)*100,means): ax.text(x,y+.004,f'{y:.3f}',ha='center',fontsize=7)
    ax.set(xlabel='三模态共同散点缺失比例（%）',ylabel='宏平均F1',xticks=np.array(rates)*100,ylim=(.57,.67))
    ax.grid(axis='y',color='#E8EAED',lw=.6); ax.legend()
    export(fig,out,'问题2_散点缺失率鲁棒曲线')


def confusion_figure(ensemble,out):
    matrix=np.asarray(ensemble['confusion_matrix_rows_true_columns_predicted'],dtype=float)
    normalized=matrix/matrix.sum(1,keepdims=True)
    labels=['负向','中性','正向']
    fig,ax=plt.subplots(figsize=(4.5,3.9),constrained_layout=True)
    fig.suptitle('验证集预测类别的归一化混淆矩阵',fontsize=9,fontweight='bold')
    image=ax.imshow(normalized,cmap='Blues',vmin=0,vmax=1)
    for i in range(3):
        for j in range(3):
            color='white' if normalized[i,j]>.55 else '#202124'
            ax.text(j,i,f'{normalized[i,j]:.1%}\n(n={int(matrix[i,j])})',ha='center',va='center',color=color,fontsize=7)
    ax.set_xticks(range(3),labels); ax.set_yticks(range(3),labels)
    ax.set(xlabel='预测类别',ylabel='真实类别')
    cbar=fig.colorbar(image,ax=ax,fraction=.047,pad=.04); cbar.set_label('行归一化比例')
    export(fig,out,'问题2_新最优归一化混淆矩阵')


def main(args):
    rows=read_csv(args.train_csv); robust_rows=read_csv(args.robust_csv)
    old=json.loads(Path(args.old_json).read_text(encoding='utf-8'))['cross_modal_profile']
    new=json.loads(Path(args.new_json).read_text(encoding='utf-8'))
    ensemble=json.loads(Path(args.ensemble_json).read_text(encoding='utf-8'))
    out=Path(args.out)
    training_figure(rows,out); comparison_figure(old,new,out); scenario_figures(robust_rows,out); confusion_figure(ensemble,out)
    qa={'core_conclusion':'自适应融合模型在完整输入与模态缺失条件下均提升宏平均F1。',
        'archetype':'quantitative grid','backend':'Python/matplotlib','validation_samples':1871,
        'split':'16,326 train; 1,871 video-disjoint valid','seeds':'one DeBERTa seed; deterministic missingness replicates',
        'source_data':[str(args.train_csv),str(args.robust_csv)],'excluded_rows':0,
        'exports':['SVG 中文字形轮廓','PDF 嵌入中文字体','PNG 600 dpi','TIFF 600 dpi']}
    (out/'问题2_新最优图形QA说明.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-csv',required=True); ap.add_argument('--robust-csv',required=True)
    ap.add_argument('--old-json',required=True); ap.add_argument('--new-json',required=True)
    ap.add_argument('--ensemble-json',required=True); ap.add_argument('--out',required=True)
    main(ap.parse_args())
