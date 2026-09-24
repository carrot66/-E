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
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans','sans-serif'],
    'svg.fonttype':'none','pdf.fonttype':42,'font.size':7,
    'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.8,
    'legend.frameon':False,'xtick.major.width':.7,'ytick.major.width':.7,
})

BLUE='#4472A8'; ORANGE='#E69F5B'; GREEN='#5A9A78'; GRAY='#A7ABB2'; RED='#C65D58'


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))


def export(fig,out,stem):
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/f'{stem}.svg',bbox_inches='tight')
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
    axs[0,0].plot(epoch,loss,'o-',color=GRAY,lw=1.5,ms=3); axs[0,0].set(ylabel='Training loss',xlabel='Epoch')
    axs[0,1].plot(epoch,acc,label='Accuracy',color=BLUE,lw=1.5); axs[0,1].plot(epoch,f1,label='Macro-F1',color=ORANGE,lw=1.5)
    axs[0,1].scatter(epoch[best],f1[best],s=28,color=RED,zorder=3); axs[0,1].legend(); axs[0,1].set(ylabel='Score',xlabel='Epoch')
    axs[1,0].plot(epoch,mae,'o-',color=GREEN,lw=1.5,ms=3); axs[1,0].set(ylabel='MAE',xlabel='Epoch')
    axs[1,1].plot(epoch,sel,'o-',color=BLUE,lw=1.5,ms=3); axs[1,1].scatter(epoch[best],sel[best],s=28,color=RED,zorder=3)
    axs[1,1].annotate(f'best epoch {epoch[best]}',xy=(epoch[best],sel[best]),xytext=(5,8),textcoords='offset points',fontsize=7)
    axs[1,1].set(ylabel='Selection score',xlabel='Epoch')
    for a,l in zip(axs.ravel(),'abcd'): panel(a,l); a.grid(axis='y',color='#E8EAED',lw=.6)
    export(fig,out,'问题2_DeBERTa逐轮训练曲线')


def comparison_figure(old,new,out):
    labels=['Clean','Contiguous 30%','Scattered TAV']
    old_values=[old['clean_F1'],old['contiguous_30pct_mean_F1'],old['scattered_TAV_mean_F1']]
    new_values=[new['clean_F1'],new['contiguous_30pct_mean_F1'],new['scattered_TAV_mean_F1']]
    x=np.arange(3); width=.34
    fig,ax=plt.subplots(figsize=(7.2,3.35),constrained_layout=True)
    b1=ax.bar(x-width/2,old_values,width,color=GRAY,label='Previous cross-modal')
    b2=ax.bar(x+width/2,new_values,width,color=BLUE,label='DeBERTa adaptive ensemble')
    for bars in (b1,b2):
        for b in bars: ax.text(b.get_x()+b.get_width()/2,b.get_height()+.006,f'{b.get_height():.3f}',ha='center',fontsize=7)
    ax.set_xticks(x,labels); ax.set_ylabel('Macro-F1'); ax.set_ylim(.54,.71); ax.legend(ncol=2,loc='upper right')
    ax.grid(axis='y',color='#E8EAED',lw=.6); panel(ax,'a')
    ax.text(.01,.98,'Higher across clean and missing-modality conditions',transform=ax.transAxes,va='top',fontweight='bold')
    export(fig,out,'问题2_新旧模型鲁棒性对比')


def scenario_figures(rows,out):
    contiguous=[x for x in rows if x['family']=='contiguous']
    order=['none','T','A','V','AV','TAV']; label={'none':'Clean','T':'Text','A':'Audio','V':'Vision','AV':'Audio+Vision','TAV':'All three'}
    values=[]
    for key in order:
        row=next(x for x in contiguous if x['subset']==key); values.append(float(row['Macro_F1']))
    fig,ax=plt.subplots(figsize=(7.2,3.5),constrained_layout=True)
    bars=ax.bar(np.arange(len(order)),values,color=[GREEN]+[BLUE]*5,width=.66)
    for b,v in zip(bars,values): ax.text(b.get_x()+b.get_width()/2,v+.006,f'{v:.3f}',ha='center',fontsize=7)
    ax.set_xticks(np.arange(len(order)),[label[x] for x in order],rotation=15,ha='right'); ax.set_ylabel('Macro-F1'); ax.set_ylim(.56,.71)
    ax.grid(axis='y',color='#E8EAED',lw=.6); panel(ax,'a')
    export(fig,out,'问题2_缺失模态类型影响')

    scattered=[x for x in rows if x['family']=='scattered']; rates=sorted({float(x['rate']) for x in scattered})
    means=[np.mean([float(x['Macro_F1']) for x in scattered if float(x['rate'])==r]) for r in rates]
    lo=[min(float(x['Macro_F1']) for x in scattered if float(x['rate'])==r) for r in rates]
    hi=[max(float(x['Macro_F1']) for x in scattered if float(x['rate'])==r) for r in rates]
    fig,ax=plt.subplots(figsize=(7.2,3.45),constrained_layout=True)
    ax.plot(np.array(rates)*100,means,'o-',color=BLUE,lw=1.8,ms=5)
    ax.fill_between(np.array(rates)*100,lo,hi,color=BLUE,alpha=.16,label='Two deterministic replicates')
    for x,y in zip(np.array(rates)*100,means): ax.text(x,y+.004,f'{y:.3f}',ha='center',fontsize=7)
    ax.set(xlabel='Scattered TAV missing rate (%)',ylabel='Macro-F1',xticks=np.array(rates)*100,ylim=(.57,.67))
    ax.grid(axis='y',color='#E8EAED',lw=.6); ax.legend(); panel(ax,'a')
    export(fig,out,'问题2_散点缺失率鲁棒曲线')


def main(args):
    rows=read_csv(args.train_csv); robust_rows=read_csv(args.robust_csv)
    old=json.loads(Path(args.old_json).read_text(encoding='utf-8'))['cross_modal_profile']
    new=json.loads(Path(args.new_json).read_text(encoding='utf-8'))
    out=Path(args.out)
    training_figure(rows,out); comparison_figure(old,new,out); scenario_figures(robust_rows,out)
    qa={'core_conclusion':'DeBERTa adaptive ensembling improves clean and missing-modality Macro-F1.',
        'archetype':'quantitative grid','backend':'Python/matplotlib','validation_samples':1871,
        'split':'16,326 train; 1,871 video-disjoint valid','seeds':'one DeBERTa seed; deterministic missingness replicates',
        'source_data':[str(args.train_csv),str(args.robust_csv)],'excluded_rows':0,
        'exports':['SVG editable text','PDF TrueType text','PNG 600 dpi','TIFF 600 dpi']}
    (out/'问题2_新最优图形QA说明.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-csv',required=True); ap.add_argument('--robust-csv',required=True)
    ap.add_argument('--old-json',required=True); ap.add_argument('--new-json',required=True); ap.add_argument('--out',required=True)
    main(ap.parse_args())
