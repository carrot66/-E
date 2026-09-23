"""绘制优化版问题2的验证集消融、缺失规律和错误分析。"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({'font.family':'sans-serif',
    'font.sans-serif':['WenQuanYi Micro Hei','Arial','DejaVu Sans'],
    'font.size':8,'axes.spines.right':False,'axes.spines.top':False,
    'axes.linewidth':.8,'svg.fonttype':'none','pdf.fonttype':42})
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/question2/问题2_优化实验结果'
DEST=BASE/'图表'
DEST.mkdir(parents=True,exist_ok=True)
SELECTION=json.loads((BASE/'问题2_集成选模结论.json').read_text(encoding='utf-8'))
SELECTED=SELECTION['selected']

def read(name):
    with (BASE/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def save(fig,stem):
    fig.savefig(DEST/f'{stem}.svg',bbox_inches='tight')
    fig.savefig(DEST/f'{stem}.pdf',bbox_inches='tight')
    fig.savefig(DEST/f'{stem}.png',dpi=600,bbox_inches='tight')
    fig.savefig(DEST/f'{stem}.tiff',dpi=600,bbox_inches='tight')
    plt.close(fig)

def main():
    rows=read('问题2_验证集缺失类型率位置.csv')
    base=[r for r in rows if r['模型']=='baseline']; optimized=[r for r in rows if r['模型']==SELECTED]
    fig,axes=plt.subplots(1,2,figsize=(7.4,2.9),constrained_layout=True)
    for m,label,color in [('baseline','等权融合','#7A8FA6'),(SELECTED,'优化模型','#2E8B8B')]:
        q=[r for r in rows if r['模型']==m]
        for sub,label2,style in [('T','文本缺失','-'),('AV','语音+视觉缺失','--'),('TAV','三模态缺失',':')]:
            rates=[.1,.3,.5,.7]
            yy=[float(np.mean([float(r['Macro_F1']) for r in q if r['subset']==sub and r['position']=='random' and abs(float(r['rate'])-rate)<1e-6])) for rate in rates]
            axes[0].plot(np.asarray(rates)*100,yy,color=color,linestyle=style,marker='o',markersize=3,label=label+' / '+label2)
    axes[0].set(xlabel='连续缺失比例 (%)',ylabel='Macro-F1',title='(a) 缺失比例与模态组合')
    axes[0].legend(fontsize=6,frameon=False,ncol=2,loc='lower left')
    axes[0].grid(color='#E8ECEF',linewidth=.6)
    groups=['none','T','A','V','TA','TV','AV','TAV']; labels=['完整','文本','语音','视觉','文+语','文+视','语+视','三模态']
    x=np.arange(len(groups)); width=.38
    for j,(q,label,color) in enumerate([(base,'等权融合','#A4B4C2'),(optimized,'优化模型','#2E8B8B')]):
        vals=[]
        for group in groups:
            subset=[float(r['Macro_F1']) for r in q if r['subset']==group and (group=='none' or (r['position']=='random' and abs(float(r['rate'])-.3)<1e-6))]
            vals.append(float(np.mean(subset)))
        axes[1].bar(x+(-.5+j)*width,vals,width,label=label,color=color)
    axes[1].set_xticks(x,labels,rotation=25)
    axes[1].set(ylabel='Macro-F1',title='(b) 缺失类型（30%）')
    axes[1].legend(fontsize=7,frameon=False,loc='upper center',
                   bbox_to_anchor=(.5,-.22),ncol=2)
    axes[1].grid(axis='y',color='#E8ECEF',linewidth=.6); axes[1].set_axisbelow(True)
    save(fig,'问题2_鲁棒性与缺失规律')

    pred=read(f'问题2_验证集_{SELECTED}_全量预测.csv')
    labels3=['Negative','Neutral','Positive']; matrix=np.zeros((3,3),int)
    for r in pred:matrix[labels3.index(r['真实极性']),labels3.index(r['预测极性'])]+=1
    fig,axes=plt.subplots(1,2,figsize=(7.4,2.9),constrained_layout=True)
    im=axes[0].imshow(matrix,cmap='Blues')
    fig.colorbar(im,ax=axes[0],fraction=.046,pad=.04)
    for i in range(3):
        for j in range(3):axes[0].text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()*.55 else '#183B56')
    axes[0].set_xticks(range(3),['负向','中性','正向']); axes[0].set_yticks(range(3),['负向','中性','正向'])
    axes[0].set(xlabel='预测极性',ylabel='真实极性',title='(c) 验证集混淆矩阵')
    true=np.asarray([float(r['真实强度']) for r in pred]); estimated=np.asarray([float(r['预测强度']) for r in pred])
    axes[1].scatter(true,estimated,s=8,alpha=.42,color='#2E8B8B',edgecolors='none')
    axes[1].plot([-3,3],[-3,3],'--',color='#555555',linewidth=.9)
    axes[1].set(xlim=(-3.1,3.1),ylim=(-3.1,3.1),xlabel='真实强度',ylabel='预测强度',title='(d) 验证集强度回归')
    axes[1].grid(color='#E8ECEF',linewidth=.6)
    save(fig,'问题2_验证集分类与强度预测')
    print('图表输出：',DEST)

if __name__=='__main__':main()
