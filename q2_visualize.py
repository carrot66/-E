"""问题2验证集图表：仅读取首轮实验CSV，不生成模拟数据。"""
from pathlib import Path
import csv
import collections
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['WenQuanYi Micro Hei', 'Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.spines.right': False, 'axes.spines.top': False,
    'axes.linewidth': .8, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
})

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / 'outputs' / '问题2_首轮实验结果'
OUT = RESULT / '图表'
COLORS = {'普通融合': '#7A8FA6', '连续缺失增强': '#B7A1C6', '动态门控': '#2E8B8B', '门控蒸馏': '#D98E5B'}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def mean(xs):
    return float(np.mean([float(x) for x in xs]))


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{stem}.png', dpi=450, bbox_inches='tight')
    fig.savefig(OUT / f'{stem}.svg', bbox_inches='tight')
    plt.close(fig)


def main():
    rows = read_csv(RESULT / '问题2_验证集缺失实验.csv')
    names = ['普通融合', '连续缺失增强', '动态门控', '门控蒸馏']
    # Panel A: clean vs mean random missing ablation.
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
    for j, metric in enumerate(['Macro_F1', 'MAE']):
        x = np.arange(len(names)); width = .36
        clean = []; missing = []
        for n in names:
            rr = [r for r in rows if r['模型'] == n]
            clean.append(mean([r[metric] for r in rr if r['subset'] == 'none']))
            missing.append(mean([r[metric] for r in rr if r['subset'] != 'none' and r['position'] == 'random']))
        ax[j].bar(x-width/2, clean, width, label='完整输入', color='#CBD5DF')
        ax[j].bar(x+width/2, missing, width, label='连续缺失', color=[COLORS[n] for n in names])
        ax[j].set_xticks(x, ['拼接', '增强', '门控', '蒸馏'], rotation=15)
        ax[j].set_ylabel('Macro-F1' if metric == 'Macro_F1' else 'MAE')
        ax[j].set_title('(a) ' + ('分类消融' if metric == 'Macro_F1' else '强度回归消融'))
        ax[j].grid(axis='y', color='#E8ECEF', linewidth=.6); ax[j].set_axisbelow(True)
    ax[0].legend(loc='upper right', frameon=False, fontsize=7)
    save(fig, '问题2_模型消融对比')

    # Panel B: dynamic gate versus missing rate/type.
    gate = [r for r in rows if r['模型'] == '动态门控']
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
    rates = [.1, .3, .5, .7]
    for subset, color, label in [('T', '#3D6FA5', '文本'), ('A', '#C66B3D', '语音'), ('V', '#5B8C5A', '视觉'), ('AV', '#8C63A8', '语音+视觉'), ('TAV', '#555555', '三模态')]:
        vals = []
        for rate in rates:
            q = [r for r in gate if r['subset'] == subset and r['position'] == 'random' and abs(float(r['rate'])-rate)<1e-6]
            vals.append(mean([r['Macro_F1'] for r in q]) if q else np.nan)
        ax[0].plot(np.array(rates)*100, vals, marker='o', linewidth=1.5, label=label, color=color)
    ax[0].set_xlabel('连续缺失比例 (%)'); ax[0].set_ylabel('Macro-F1'); ax[0].set_title('(b) 缺失比例规律')
    ax[0].grid(color='#E8ECEF', linewidth=.6); ax[0].legend(frameon=False, fontsize=7, ncol=2)
    subs = ['none', 'T', 'A', 'V', 'TA', 'TV', 'AV', 'TAV']; labels = ['完整', '文本', '语音', '视觉', '文+语', '文+视', '语+视', '三模态']
    vals = [mean([r['Macro_F1'] for r in gate if r['subset'] == s and (s == 'none' or (r['position'] == 'random' and abs(float(r['rate'])-.3)<1e-6))]) for s in subs]
    ax[1].bar(np.arange(len(subs)), vals, color=['#CBD5DF']+[COLORS['动态门控']]*7)
    ax[1].set_xticks(np.arange(len(subs)), labels, rotation=28); ax[1].set_ylabel('Macro-F1'); ax[1].set_title('(c) 缺失类型（30%）')
    ax[1].grid(axis='y', color='#E8ECEF', linewidth=.6); ax[1].set_axisbelow(True)
    save(fig, '问题2_缺失率与缺失类型规律')

    # Panel C: confusion matrix and regression scatter for selected model.
    pred = read_csv(RESULT / '问题2_验证集_动态门控_逐条预测.csv')
    names_cls = ['Negative', 'Neutral', 'Positive']; mat = np.zeros((3,3), dtype=int)
    for r in pred: mat[names_cls.index(r['真实极性']), names_cls.index(r['预测极性'])] += 1
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
    im = ax[0].imshow(mat, cmap='Blues'); fig.colorbar(im, ax=ax[0], fraction=.046, pad=.04)
    for i in range(3):
        for j in range(3): ax[0].text(j, i, str(mat[i,j]), ha='center', va='center', color='white' if mat[i,j] > mat.max()*.55 else '#183B56')
    ax[0].set_xticks(range(3), ['负向', '中性', '正向']); ax[0].set_yticks(range(3), ['负向', '中性', '正向'])
    ax[0].set_xlabel('预测'); ax[0].set_ylabel('真实'); ax[0].set_title('(d) 验证集混淆矩阵')
    true = np.asarray([float(r['真实强度']) for r in pred]); out = np.asarray([float(r['预测强度']) for r in pred]);
    ax[1].scatter(true, out, s=8, alpha=.42, color=COLORS['动态门控'], edgecolors='none')
    ax[1].plot([-3,3],[-3,3], '--', color='#555555', linewidth=.9); ax[1].set(xlim=(-3.1,3.1), ylim=(-3.1,3.1), xlabel='真实情感强度', ylabel='预测情感强度', title='(e) 强度回归')
    ax[1].grid(color='#E8ECEF', linewidth=.6); ax[1].set_axisbelow(True)
    save(fig, '问题2_验证集预测与错误分析')

    print('saved figures to', OUT)


if __name__ == '__main__': main()
