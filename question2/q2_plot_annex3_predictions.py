"""Figures for all 30 unlabeled Annex 3 predictions.

Conclusion: predictions lean positive, but several samples remain uncertain;
missingness must be shown alongside confidence rather than interpreted as error.
Evidence: all 30 per-sample probabilities, three-class counts, and observed
modality availability. Quantitative-grid archetype, Python/matplotlib backend.
Source data: 问题2_附件3全量预测.csv. No samples or classes are excluded.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from q2_figure_style import use_chinese_font


use_chinese_font()
mpl.rcParams["font.size"] = 8
COLORS = {"Negative": "#B86F69", "Neutral": "#8795A1", "Positive": "#4D8E87"}
LABELS = ("Negative", "Neutral", "Positive")
CHINESE = {"Negative": "负向", "Neutral": "中性", "Positive": "正向"}


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 30 or len({r["样本编号"] for r in rows}) != 30:
        raise ValueError("Expected the complete set of 30 unique predictions")
    return rows


def save(fig, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    svg = out / f"{stem}.svg"
    fig.savefig(svg, bbox_inches="tight")
    # Matplotlib can leave spaces at the ends of multiline SVG path commands.
    svg.write_text(re.sub(r"[ \t]+(?=\r?$)", "", svg.read_text(encoding="utf-8"),
                          flags=re.MULTILINE), encoding="utf-8")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(out / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def probabilities(rows: list[dict], out: Path) -> None:
    # A continuous three-color bar has no white edges or grid lines.
    fig, ax = plt.subplots(figsize=(7.2, 8.8))
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for label, key in zip(LABELS, ("负向概率", "中性概率", "正向概率")):
        values = np.array([float(r[key]) for r in rows])
        ax.barh(y, values, left=left, height=.72, color=COLORS[label],
                edgecolor="none", linewidth=0, label=CHINESE[label])
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels([r["样本编号"].split("_")[-1] for r in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0", ".25", ".50", ".75", "1"])
    ax.set_xlabel("预测类别概率")
    ax.set_ylabel("附件3样本编号")
    ax.set_title("附件3全部30条样本的情感类别概率", loc="left", fontweight="bold", pad=12)
    ax.text(0, 1.01, "无真实标签；右侧黑点表示最大类别概率低于0.50",
            transform=ax.transAxes, fontsize=7, color="#59656B")
    low = np.array([float(r["预测最大概率"]) < .5 for r in rows])
    ax.scatter(np.full(low.sum(), 1.035), y[low], s=17, color="#333A40", zorder=3, clip_on=False)
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, -.07))
    save(fig, out, "问题2_附件3逐样本类别概率")


def composition_and_missingness(rows: list[dict], out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1, 1.8]})
    fig.suptitle("附件3预测类别构成与模态缺失关系",fontsize=10,fontweight="bold",y=1.02)
    counts = Counter(r["预测极性"] for r in rows)
    vals = [counts.get(k, 0) for k in LABELS]
    ax1.barh(np.arange(3), vals, color=[COLORS[k] for k in LABELS],
             height=.55, edgecolor="none", linewidth=0)
    ax1.set_yticks(np.arange(3), [CHINESE[k] for k in LABELS])
    ax1.invert_yaxis()
    ax1.set_xlim(0, 18)
    ax1.set_xlabel("预测样本数（共30条）")
    ax1.set_title("预测类别分布", loc="left", fontweight="bold")
    for i, value in enumerate(vals):
        ax1.text(value + .35, i, str(value), va="center", fontsize=8)
    ax1.tick_params(axis="y", length=0)
    ax1.grid(False)
    ax1.spines["left"].set_visible(False)

    missing = np.array([np.mean([float(r[k]) for k in (
        "文本不可用比例", "语音不可用比例", "视觉不可用比例")]) for r in rows])
    conf = np.array([float(r["预测最大概率"]) for r in rows])
    for label in LABELS:
        indices = np.array([r["预测极性"] == label for r in rows])
        ax2.scatter(missing[indices], conf[indices], s=34, color=COLORS[label],
                    label=CHINESE[label], alpha=.9, edgecolor="none", zorder=3)
    for i, r in enumerate(rows):
        if conf[i] < .5:
            sample_number = r["样本编号"].split("_")[-1]
            label_offsets = {"08": (5, -13), "13": (-4, 7), "18": (4, 5),
                             "22": (4, 5), "24": (4, 5), "28": (4, 5)}
            ax2.annotate(r["样本编号"].split("_")[-1], (missing[i], conf[i]),
                         xytext=label_offsets.get(sample_number, (4, 5)),
                         textcoords="offset points", fontsize=6,
                         color="#39454B")
    ax2.set_xlim(-.03, .56)
    ax2.set_ylim(.32, 1.02)
    ax2.set_xlabel("三模态平均不可用比例")
    ax2.set_ylabel("最大类别概率")
    ax2.set_title("模态缺失与预测置信度", loc="left", fontweight="bold")
    ax2.grid(False)
    fig.subplots_adjust(wspace=.38,top=.80)
    save(fig, out, "问题2_附件3类别分布与缺失置信度")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = load(args.csv)
    probabilities(data, args.out)
    composition_and_missingness(data, args.out)
