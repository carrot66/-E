"""用已归档的审计 JSON 绘制问题二预处理图；不读取测试集或附件3标签。"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from q2_figure_style import use_chinese_font


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/question2/问题2_完整版MOSEI安全整合审计/问题2_完整版MOSEI与竞赛数据安全整合审计.json"
SHIFT_AUDIT = ROOT / "outputs/question2/问题2_训练集分组交叉验证/问题2_数据泄漏与质量审计.json"
OUT = ROOT / "outputs/question2/问题2_预处理结果图"
COLORS = {"负向": "#B86F69", "中性": "#8795A1", "正向": "#4D8E87"}
SPLITS = ("train", "valid", "test")
SPLIT_NAMES = ("训练集", "验证集", "保留测试集")


def load(audit_path: Path, shift_path: Path) -> tuple[dict, dict]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    shift = json.loads(shift_path.read_text(encoding="utf-8"))
    split = audit["zip_csv"]["split_counts"]
    assert sum(split.values()) == audit["zip_csv"]["samples"]
    assert sum(audit["zip_csv"]["train_class_counts"].values()) == split["train"]
    assert sum(audit["zip_csv"]["valid_class_counts"].values()) == split["valid"]
    assert audit["safe_train_sample_count_after_annex3_exclusion"] == split["train"]
    assert audit["safe_train_5fold_video_overlap_count"] == 0
    assert audit["annex3_unique_matches"] == 30
    assert audit["annex3_match_split_counts"] == {"test": 30}
    assert "置空" in audit["zip_csv"]["test_label_policy"]
    assert shift["annex3_unlabelled_input_shift"]["summary"]["content_positions"] == 605
    return audit, shift


def save(fig, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    svg = out / f"{stem}.svg"
    fig.savefig(svg, bbox_inches="tight")
    svg.write_text(re.sub(r"[ \t]+(?=\r?$)", "", svg.read_text(encoding="utf-8"),
                          flags=re.MULTILINE), encoding="utf-8")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def figure_sources(audit: dict, out: Path) -> None:
    info = audit["zip_csv"]
    totals = np.array([info["split_counts"][key] for key in SPLITS])
    added = np.array([info["additional_by_split"][key] for key in SPLITS])
    original = totals - added
    groups = [info["split_video_counts"][key] for key in SPLITS]
    y = np.arange(3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.45),
                                    gridspec_kw={"width_ratios": [1.35, 1]},
                                    constrained_layout=True)
    fig.suptitle("对齐数据的来源组成与视频组划分", fontsize=10, fontweight="bold")
    p_original = original / totals * 100
    p_added = added / totals * 100
    ax1.barh(y, p_original, color="#4472A8", height=.56, label="原竞赛数据")
    ax1.barh(y, p_added, left=p_original, color="#9FC5CF", height=.56,
             label="完整版新增片段")
    ax1.set_yticks(y, SPLIT_NAMES)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 114)
    ax1.set_xticks([0, 25, 50, 75, 100])
    ax1.set_xlabel("各划分中的样本来源比例（%）")
    ax1.set_title("样本来源（总数标于右侧）")
    for i, total in enumerate(totals):
        ax1.text(101, i, f"{total:,}", va="center", fontsize=8)
    ax1.legend(ncol=2, fontsize=7, loc="lower center", bbox_to_anchor=(.5, -.34))
    ax2.barh(y, groups, color=["#4472A8", "#789EBD", "#B1BAC2"], height=.56)
    ax2.set_yticks(y, SPLIT_NAMES)
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(groups) * 1.2)
    ax2.set_xlabel("视频组数量")
    ax2.set_title("按视频分组的划分")
    for i, value in enumerate(groups):
        ax2.text(value + 35, i, str(value), va="center", fontsize=8)
    fig.text(.5, -.055, "保留测试集仅作输入审计；其标签未用于训练、调参或绘图。",
             ha="center", fontsize=7, color="#59656B")
    save(fig, out, "问题2_预处理01_数据来源与视频组划分")


def figure_classes(audit: dict, out: Path) -> None:
    info = audit["zip_csv"]
    keys = ("Negative", "Neutral", "Positive")
    names = ("负向", "中性", "正向")
    splits = ("train", "valid")
    totals = np.array([info["split_counts"][key] for key in splits])
    fig, ax = plt.subplots(figsize=(7.2, 2.9), constrained_layout=True)
    fig.suptitle("训练与验证集的情感类别构成", fontsize=10, fontweight="bold")
    left = np.zeros(2)
    for key, name in zip(keys, names):
        counts = np.array([info[f"{split}_class_counts"][key]
                           for split in splits])
        share = counts / totals * 100
        ax.barh(np.arange(2), share, left=left, height=.55,
                color=COLORS[name], label=name)
        for i in range(2):
            ax.text(left[i] + share[i] / 2, i, f"{share[i]:.1f}%\n({counts[i]:,}条)",
                    ha="center", va="center", color="white", fontsize=8)
        left += share
    ax.set_yticks([0, 1], ["训练集", "验证集"])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("类别占比（%）")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(.5, -.37), fontsize=8)
    ax.grid(False)
    fig.text(.5, -.08, "仅统计训练集和验证集；保留测试集与附件3的标签均未读取。",
             ha="center", fontsize=7, color="#59656B")
    save(fig, out, "问题2_预处理02_训练验证类别构成")


def figure_native_missingness(audit: dict, shift: dict, out: Path) -> None:
    info = audit["full_input_feature_quality"]
    summary = shift["annex3_unlabelled_input_shift"]["summary"]
    positions = np.array([info[key]["content_positions"] for key in ("train", "valid")])
    values = np.array([
        [0, info[key]["audio_zero_content_positions"],
         info[key]["vision_zero_content_positions"]]
        for key in ("train", "valid")
    ]) / positions[:, None] * 100
    annex = np.array([summary["text_missing_attention_or_unknown"],
                      summary["audio_zero_content_positions"],
                      summary["vision_zero_content_positions"]]) / summary["content_positions"] * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.65),
                                    constrained_layout=True)
    fig.suptitle("原始输入的模态不可用位置比例", fontsize=10, fontweight="bold")
    labels = ["文本", "语音", "视觉"]
    x = np.arange(3)
    width = .32
    for i, (name, color) in enumerate((("训练集", "#4472A8"), ("验证集", "#78A7B7"))):
        bars = ax1.bar(x + (i - .5) * width, values[i], width,
                       color=color, label=name)
        for bar, value in zip(bars, values[i]):
            ax1.text(bar.get_x() + bar.get_width()/2, value + .15,
                     f"{value:.2f}", ha="center", fontsize=7)
    ax1.set_xticks(x, labels)
    ax1.set_ylim(0, 6.25)
    ax1.set_ylabel("不可用位置比例（%）")
    ax1.set_title("训练与验证输入")
    ax1.legend(fontsize=7, loc="upper left")
    bars = ax2.bar(labels, annex, color=["#4472A8", "#5A9A78", "#7B6AA8"], width=.6)
    ax2.set_ylim(0, 28)
    ax2.set_ylabel("不可用位置比例（%）")
    ax2.set_title("附件3无标签输入")
    for bar, value in zip(bars, annex):
        ax2.text(bar.get_x() + bar.get_width()/2, value + .5,
                 f"{value:.1f}%", ha="center", fontsize=8)
    fig.text(.5, -.035,
             "分母为有效内容位置。附件3中与语音、视觉零值同时出现的占位符视为文本缺失；两幅图纵轴刻度不同。",
             ha="center", fontsize=7, color="#59656B")
    save(fig, out, "问题2_预处理03_原始模态不可用比例")


def figure_folds(audit: dict, out: Path) -> None:
    folds = audit["safe_train_grouped_5fold_summary"]
    order = sorted(folds, key=int)
    counts = np.array([folds[key]["validation_class_counts_negative_neutral_positive"]
                       for key in order])
    totals = np.array([folds[key]["validation_samples"] for key in order])
    groups = np.array([folds[key]["validation_video_groups"] for key in order])
    assert np.all(counts.sum(axis=1) == totals)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.6),
                                    gridspec_kw={"width_ratios": [1.4, 1]},
                                    constrained_layout=True)
    fig.suptitle("视频组隔离的五折划分与类别稳定性", fontsize=10, fontweight="bold")
    y = np.arange(5)
    left = np.zeros(5)
    for j, name in enumerate(("负向", "中性", "正向")):
        share = counts[:, j] / totals * 100
        ax1.barh(y, share, left=left, height=.56, color=COLORS[name], label=name)
        left += share
    ax1.set_yticks(y, [f"第{int(key)+1}折" for key in order])
    ax1.invert_yaxis()
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("验证折类别占比（%）")
    ax1.set_title("验证折的类别构成")
    ax1.legend(ncol=3, fontsize=7, loc="lower center", bbox_to_anchor=(.5, -.35))
    ax1.grid(False)
    ax2.barh(y, groups, height=.56, color="#789EBD")
    ax2.set_yticks(y, [f"第{int(key)+1}折" for key in order])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 540)
    ax2.set_xlabel("验证折视频组数量")
    ax2.set_title("分组规模")
    for i, value in enumerate(groups):
        ax2.text(value + 8, i, f"{value}组", va="center", fontsize=8)
    fig.text(.5, -.03, "五折均从安全训练集内按视频分组；折间视频组重合数为0。",
             ha="center", fontsize=7, color="#59656B")
    save(fig, out, "问题2_预处理04_视频组五折划分")


def figure_flow(audit: dict, out: Path) -> None:
    info = audit["zip_csv"]
    fig, ax = plt.subplots(figsize=(7.2, 2.15), constrained_layout=True)
    fig.suptitle("问题二从对齐特征到鲁棒模型输入的预处理流程", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        ("对齐特征审计", f"{info['samples']:,}条；最长50步"),
        ("视频组隔离", f"训练{info['split_counts']['train']:,}条"),
        ("测试标签屏蔽", "置空；不参与选模"),
        ("原生缺失识别", "掩码记录缺失位置"),
        ("三模态编码", "文本、语音、视觉"),
    ]
    centers = np.linspace(.11, .89, len(stages))
    fills = ["#E8EEF2", "#DEEAF4", "#F0E9DB", "#E2EFEA", "#E7E3F1"]
    for i, ((title, sub), x) in enumerate(zip(stages, centers)):
        ax.text(x, .58, title, ha="center", va="center", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=.62", fc=fills[i], ec="#6A7882", lw=.8))
        ax.text(x, .34, sub, ha="center", va="center", fontsize=7.2, color="#4D5B64")
        if i < len(stages) - 1:
            ax.annotate("", xy=(centers[i+1]-.075, .58), xytext=(x+.075, .58),
                        arrowprops=dict(arrowstyle="->", lw=1.1, color="#667781"))
    ax.text(.5, .09, "附件3只作为无标签输入；训练期模拟缺失另行施加，原始缺失不被删除。",
            ha="center", va="center", fontsize=7.4, color="#59656B")
    save(fig, out, "问题2_预处理05_安全整合与编码流程")


def main(args: argparse.Namespace) -> None:
    use_chinese_font()
    plt.rcParams.update({"font.size": 8, "savefig.facecolor": "white"})
    audit, shift = load(args.audit, args.shift_audit)
    figure_sources(audit, args.out)
    figure_classes(audit, args.out)
    figure_native_missingness(audit, shift, args.out)
    figure_folds(audit, args.out)
    figure_flow(audit, args.out)
    manifest = {
        "figure_count": 5,
        "source_json": [str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
                        for path in (args.audit, args.shift_audit)],
        "test_labels_used": False,
        "annex3_labels_used": False,
        "png_pdf_svg_count_each": 5,
    }
    (args.out / "绘图来源与核验.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--shift-audit", type=Path, default=SHIFT_AUDIT)
    parser.add_argument("--out", type=Path, default=OUT)
    main(parser.parse_args())
