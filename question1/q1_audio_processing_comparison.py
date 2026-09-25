#!/usr/bin/env python3
"""用问题一真实视频与已保存的词级特征绘制语音处理前后对比。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
import numpy as np

from q1_feature_extract import SAMPLE_RATE, decode_audio, mel_mfcc_prosody, pool_by_word


ROOT = Path(__file__).resolve().parents[1]
# Route-B example: complete text coverage (first word ≈0.16 s, last word
# ≈7.77 s) leaves no artificial blank region at the left of the raw panel.
DEFAULT_SAMPLE = "-wny0OAz3g8__7"
DEFAULT_DATA = ROOT / "E题数据" / "E题数据"
DEFAULT_FEATURES = ROOT / "outputs" / "question1" / "问题1_全量特征结果" / "features"
DEFAULT_OUT = ROOT / "outputs" / "question1" / "问题1_补充结果图"
STEM = "q1_fig09_语音特征处理前后对比"
BLUE = "#4472A8"
RED = "#B86F69"
GREEN = "#4D8E87"
PURPLE = "#7B6AA8"
GREY = "#59656B"


def configure_font() -> None:
    chars = "问题语音特征处理前后对比原始帧级序列仅展示标准化逐词池化结果时间秒振幅系数均值对齐共"
    required = {ord(c) for c in chars}
    for family in ("Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei"):
        fonts = [font for font in font_manager.fontManager.ttflist if font.name == family]
        if any(required.issubset(FT2Font(font.fname).get_charmap()) for font in fonts):
            mpl.rcParams.update({
                "font.family": "sans-serif", "font.sans-serif": [family, "DejaVu Sans"],
                "axes.unicode_minus": False, "svg.fonttype": "path", "pdf.fonttype": 42,
                "axes.spines.top": False, "axes.spines.right": False,
                "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white",
                "savefig.facecolor": "white",
            })
            return
    raise RuntimeError("未找到可绘制本图中文字符的字体")


def source_video(data_root: Path, sample_id: str) -> Path:
    video_id, clip_id = sample_id.rsplit("__", 1)
    candidates = list(data_root.rglob(f"{video_id}/{clip_id}.mp4"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"样本 {sample_id} 应匹配一条原始视频，实际找到 {len(candidates)} 条")
    return candidates[0]


def prepare(video: Path, npz_path: Path) -> dict:
    if not video.is_file() or not npz_path.is_file():
        raise FileNotFoundError("原始视频或对应的第一问特征 NPZ 不存在")
    # Route B: always decode the audio from the original MP4.  The NPZ is used
    # only for the saved word intervals and the stored word-level audit values;
    # it is never used as a substitute for the source waveform.
    wave = decode_audio(video)
    frame, times = mel_mfcc_prosody(wave)
    with np.load(npz_path, allow_pickle=False) as z:
        if str(z["sample_id"].item()) != npz_path.stem:
            raise AssertionError("NPZ 内的样本编号与请求的原始视频不一致")
        words = z["words"].astype(str)
        intervals = z["word_times"].astype(np.float64)
        saved = z["audio"][:, [40, 66]].astype(np.float64)
        stored_times = z["audio_frame_times"].astype(np.float64)
        audio_present = bool(z["audio_present"])
        aligned = bool(np.all(z["alignment_reliable"]))
    if not audio_present or not aligned:
        raise ValueError("该示例需要有效语音和 CTC 强制对齐，以免把回退位置当作精确词界")
    if frame.shape[1] != 74 or len(times) != len(stored_times):
        raise AssertionError("帧级特征维度或时间轴与保存结果不一致")
    if not np.allclose(times, stored_times, rtol=0, atol=2e-5):
        raise AssertionError("重算的 10 ms 音频帧时间轴与保存结果不一致")
    raw = frame[:, [40, 66]].astype(np.float64)
    pooled, counts = pool_by_word(frame, times, intervals)
    max_diff = float(np.max(np.abs(pooled[:, [40, 66]] - saved)))
    if max_diff > 1e-4:
        raise AssertionError(f"重算与已保存的逐词 MFCC-1/RMS 不一致：最大差 {max_diff:.6g}")
    center = raw.mean(axis=0)
    scale = raw.std(axis=0)
    if np.any(scale <= 1e-10):
        raise AssertionError("帧级特征方差过小，无法仅供展示地做 z 标准化")
    frame_z = (raw - center) / scale
    word_z = (saved - center) / scale
    pooled_z, _ = pool_by_word(frame_z.astype(np.float32), times, intervals)
    if not np.allclose(pooled_z, word_z, atol=2e-4, rtol=2e-4):
        raise AssertionError("标准化后的帧均值与已保存词级特征的同尺度表示不一致")
    return {
        "times": times, "raw": raw, "frame_z": frame_z, "word_raw": saved, "word_z": word_z,
        "words": words, "intervals": intervals, "counts": counts,
        "center": center, "scale": scale, "max_diff": max_diff,
        "audio_sample_rate": SAMPLE_RATE, "audio_sample_count": int(len(wave)),
        "audio_duration_sec": float(len(wave) / SAMPLE_RATE),
        "audio_rms": float(np.sqrt(np.mean(np.square(wave, dtype=np.float64)))),
        "audio_peak": float(np.max(np.abs(wave))) if len(wave) else 0.0,
    }


def render(data: dict, sample_id: str, out: Path) -> None:
    configure_font()
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 1.0, "legend.frameon": False})
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.15), gridspec_kw={"width_ratios": [1, 1, 1.07]})
    t, raw = data["times"], data["raw"]
    ax = axes[0]
    h1, = ax.plot(t, raw[:, 0], color=BLUE, lw=.9, label="原始 MFCC-1")
    ax.set(xlabel="时间（秒）", ylabel="原始 MFCC-1 特征值", title="a  原始帧级序列")
    ax.set_xlim(0, float(t[-1]))
    ax2 = ax.twinx()
    h2, = ax2.plot(t, raw[:, 1], color=RED, lw=1.0, ls="--", alpha=.82, label="原始 RMS（右轴）")
    ax2.set_ylabel("原始 RMS 振幅", color=RED)
    ax2.tick_params(axis="y", colors=RED)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(RED)

    ax = axes[1]
    h3, = ax.plot(t, data["frame_z"][:, 0], color=GREEN, lw=.9, label="标准化 MFCC-1")
    h4, = ax.plot(t, data["frame_z"][:, 1], color=PURPLE, lw=.9,
                  ls="--", alpha=.9, label="标准化 RMS")
    ax.axhline(0, color="#AAB4BD", lw=.65, zorder=0)
    ax.set(xlabel="时间（秒）", ylabel="标准化特征值", title="b  标准化后序列")
    ax.set_xlim(0, float(t[-1]))

    ax = axes[2]
    k = np.arange(1, len(data["words"]) + 1)
    h5, = ax.plot(k, data["word_z"][:, 0], color=BLUE, marker="o", ms=3.5,
                  lw=1.4, label="词区间池化 MFCC-1")
    h6, = ax.plot(k, data["word_z"][:, 1], color=RED, marker="s", ms=3.5,
                  lw=1.4, ls="--", label="词区间池化 RMS")
    ax.axhline(0, color="#AAB4BD", lw=.65, zorder=0)
    ax.set(xlabel=f"对齐位置 k（词元，共 {len(k)} 个）", ylabel="标准化特征值",
           title="c  词区间池化结果")
    ax.set_xlim(.5, len(k) + .5)
    ax.set_xticks(k if len(k) <= 20 else k[::2])

    fig.suptitle("语音特征预处理前后对比", y=.985,
                 fontsize=12, fontweight="bold")
    fig.legend([h1, h2, h3, h4, h5, h6],
               ["原始 MFCC-1", "原始 RMS（右轴）", "标准化 MFCC-1",
                "标准化 RMS", "词区间池化 MFCC-1", "词区间池化 RMS"],
               loc="upper center", bbox_to_anchor=(.5, .93), ncol=3, fontsize=8.5)
    fig.text(.5, .045,
             f"样本 {sample_id}；25 ms 窗长 / 10 ms 帧移。标准化仅用于图中比较；保存的 74 维声学特征仍为原尺度逐词均值。",
             ha="center", va="center", fontsize=8, color=GREY)
    fig.subplots_adjust(left=.065, right=.93, bottom=.18, top=.73, wspace=.43)

    out.mkdir(parents=True, exist_ok=True)
    svg = out / f"{STEM}.svg"
    fig.savefig(svg, bbox_inches="tight")
    svg.write_text(re.sub(r"[ \t]+(?=\r?$)", "", svg.read_text(encoding="utf-8"),
                   flags=re.MULTILINE), encoding="utf-8")
    fig.savefig(out / f"{STEM}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{STEM}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{STEM}.tif", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_audit(data: dict, sample_id: str, video: Path, data_root: Path, out: Path) -> None:
    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def word_for_time(t: float) -> int:
        hit = np.flatnonzero((data["intervals"][:, 0] <= t) & (t < data["intervals"][:, 1]))
        return int(hit[0]) if len(hit) else -1

    # Keep the data used for the figure beside it.  This makes the plotted
    # values independently checkable without re-reading the original video.
    frame_csv = out / f"{STEM}_逐帧.csv"
    with frame_csv.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["帧序号", "时间_秒", "原始_MFCC1", "原始_RMS", "标准化_MFCC1", "标准化_RMS", "词序号", "词语"])
        for i, t in enumerate(data["times"]):
            wi = word_for_time(float(t))
            writer.writerow([
                i, f"{float(t):.6f}", f"{data['raw'][i, 0]:.9g}", f"{data['raw'][i, 1]:.9g}",
                f"{data['frame_z'][i, 0]:.9g}", f"{data['frame_z'][i, 1]:.9g}",
                wi + 1 if wi >= 0 else "", data["words"][wi] if wi >= 0 else "",
            ])

    word_csv = out / f"{STEM}_逐词.csv"
    with word_csv.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["对齐位置_k", "词语", "开始_秒", "结束_秒", "帧数", "原始_MFCC1", "原始_RMS", "标准化_MFCC1", "标准化_RMS"])
        for i, (word, (a, b), n) in enumerate(zip(data["words"], data["intervals"], data["counts"]), 1):
            writer.writerow([
                i, word, f"{float(a):.6f}", f"{float(b):.6f}", n,
                f"{data['word_raw'][i-1, 0]:.9g}", f"{data['word_raw'][i-1, 1]:.9g}",
                f"{data['word_z'][i-1, 0]:.9g}", f"{data['word_z'][i-1, 1]:.9g}",
            ])

    audit = {
        "route": "B",
        "route_description": "原始 MP4 直接解码语音；NPZ 仅提供已核验的词区间和保存特征作一致性核验",
        "sample_id": sample_id,
        "source_video": str(video.relative_to(data_root)),
        "source_video_size_bytes": int(video.stat().st_size),
        "source_video_sha256": sha256_file(video),
        "audio_decode": {"sample_rate_hz": int(data["audio_sample_rate"]), "channels": 1,
                         "pcm_format": "float32 little-endian", "sample_count": int(data["audio_sample_count"]),
                         "duration_sec": round(data["audio_duration_sec"], 6),
                         "ffmpeg_options": ["-vn", "-ac", "1", "-ar", str(data["audio_sample_rate"]), "-f", "f32le"]},
        "source_waveform_rms": data["audio_rms"],
        "source_waveform_peak": data["audio_peak"],
        "frame_count": int(len(data["times"])),
        "word_count": int(len(data["words"])),
        "frame_window_ms": 25, "frame_hop_ms": 10,
        "features": {"mfcc_first_coefficient": 40, "rms": 66},
        "z_score_for_display_only": True,
        "saved_audio_is_raw_scale_word_mean": True,
        "frame_mean": data["center"].tolist(),
        "frame_std": data["scale"].tolist(),
        "pooled_vs_saved_max_abs_diff": data["max_diff"],
        "words_without_interval_frame": int(sum(count == 0 for count in data["counts"])),
        "data_files": [frame_csv.name, word_csv.name],
        "formats": ["png", "pdf", "svg", "tif"],
    }
    (out / f"{STEM}_核验.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = out / "figure_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["figure_count"] = len(list(out.glob("q1_fig*.png")))
        manifest["audio_comparison_figure"] = STEM
        manifest["audio_comparison_data_files"] = [
            f"{STEM}_核验.json", frame_csv.name, word_csv.name,
        ]
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE)
    args = parser.parse_args()
    video = source_video(args.data_root, args.sample_id)
    npz = args.features / f"{args.sample_id}.npz"
    data = prepare(video, npz)
    render(data, args.sample_id, args.out)
    write_audit(data, args.sample_id, video, args.data_root, args.out)
    print(json.dumps({"sample_id": args.sample_id, "frames": len(data["times"]),
                      "words": len(data["words"]), "max_abs_diff": data["max_diff"],
                      "figure": str(args.out / f"{STEM}.png")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
