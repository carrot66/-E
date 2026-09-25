#!/usr/bin/env python3
"""以现代四联版式绘制问题一的三模态时序对齐示例。

版式参考用户提供的第二张图，但使用问题一自己的路线 B 原始媒体和 NPZ。
默认样本 ``-wny0OAz3g8__7`` 与参考图示例不同，避免把示例内容误当成本题数据。
原始 MP4 不进入仓库；图、核验 JSON 和逐词 CSV 是可提交的派生结果。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "E题数据" / "E题数据"
DEFAULT_FEATURES = ROOT / "outputs" / "question1" / "问题1_全量特征结果" / "features"
DEFAULT_OUT = ROOT / "outputs" / "question1" / "问题1_补充结果图"
DEFAULT_SAMPLE = "-wny0OAz3g8__7"
SAMPLE_RATE = 16_000
# Nature-inspired scientific palette supplied by the user.
LAVENDER = "#6A5C95"      # Lavender Dusk
CORAL_BLOOM = "#D99AAE"   # Coral Bloom
CORAL_PEACH = "#E7A889"   # Coral Peach / anchor highlight
SEAFOAM = "#A9CBB8"       # Seafoam Mist / RMS
PALE_AQUA = "#CFE3E6"      # Pale Aqua / grids
BLUE = "#8FA2D6"           # Periwinkle Blue / valid observations
SLATE = "#5B608C"          # Slate Violet / waveform
RED = CORAL_PEACH
LIGHT_RED = "#FBEFEA"
GRID = PALE_AQUA
GREY = "#6E7185"
INK = "#2E3142"
MISSING = "#EFF3F5"
BACKGROUND = "#F6FAFB"


def configure_font() -> None:
    required = {ord(c) for c in "问题一三模态时序对齐样本秒情感标注锚点词时间箱文本语音视频帧共享表示时间振幅"}
    for family in ("Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei"):
        fonts = [f for f in font_manager.fontManager.ttflist if f.name == family]
        if any(required.issubset(FT2Font(f.fname).get_charmap()) for f in fonts):
            mpl.rcParams.update({
                "font.family": "sans-serif", "font.sans-serif": [family, "DejaVu Sans"],
                "axes.unicode_minus": False, "svg.fonttype": "path", "pdf.fonttype": 42,
                "figure.facecolor": BACKGROUND, "axes.facecolor": BACKGROUND,
                "axes.spines.top": False, "axes.spines.right": False,
                "savefig.facecolor": "white",
            })
            return
    mpl.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_video(data_root: Path, sample_id: str) -> Path:
    video_id, clip_id = sample_id.rsplit("__", 1)
    media = data_root / "附件1-数据集原始多模态样本" / "MOSEI数据集部分原始视频-100条"
    direct = media / video_id / f"{clip_id}.mp4"
    if direct.is_file():
        return direct
    found = list(media.rglob(f"{video_id}/{clip_id}.mp4"))
    if len(found) != 1:
        raise FileNotFoundError(f"样本 {sample_id} 未唯一匹配原始 MP4，找到 {len(found)} 条")
    return found[0]


def decode_audio(path: Path) -> np.ndarray:
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path),
           "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode(errors="replace")[-500:])
    return np.nan_to_num(np.frombuffer(p.stdout, dtype="<f4").copy()).astype(np.float32)


def rms_series(wave: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nfft, hop = 400, 160
    if len(wave) < nfft:
        wave = np.pad(wave, (0, nfft - len(wave)))
    pad = nfft // 2
    padded = np.pad(wave, (pad, pad), mode="reflect")
    count = max(1, 1 + (len(padded) - nfft) // hop)
    frames = np.stack([padded[i * hop:i * hop + nfft] for i in range(count)])
    return np.arange(count, dtype=float) * hop / SAMPLE_RATE, np.sqrt(np.mean(frames.astype(float) ** 2, axis=1))


def read_frame(path: Path, time_sec: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(time_sec)) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def assign_lanes(intervals: np.ndarray, gap: float = 0.015) -> np.ndarray:
    """Distribute labels over four visual lanes.

    Consecutive words rarely overlap as intervals, but their *labels* do. A
    cyclic lane assignment therefore communicates the sequence more clearly
    than interval-only packing, while retaining a deterministic four-lane
    layout for long transcripts.
    """
    return np.arange(len(intervals), dtype=int) % 4


def downsample_wave(wave: np.ndarray, max_points: int = 30_000) -> tuple[np.ndarray, np.ndarray]:
    stride = max(1, int(math.ceil(2 * len(wave) / max_points)))
    indices = []
    for start in range(0, len(wave), stride):
        block = wave[start:start + stride]
        indices.extend([start + int(block.argmin()), start + int(block.argmax())])
    indices = np.unique(indices)
    return indices / SAMPLE_RATE, wave[indices]


def build_payload(video: Path, npz_path: Path, anchor_word: str) -> dict:
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    wave = decode_audio(video)
    rms_t, rms = rms_series(wave)
    with np.load(npz_path, allow_pickle=False) as z:
        words = z["words"].astype(str)
        intervals = z["word_times"].astype(float)
        sample_id = str(z["sample_id"].item())
        label = float(np.asarray(z["label"]).reshape(-1)[0])
        annotation = str(z["annotation"].item())
        audio_frame_times = z["audio_frame_times"].astype(float)
        video_frame_times = z["video_frame_times"].astype(float)
        vision_valid = z["vision_valid"].astype(bool)
        face_valid = z["face_blendshape_valid"].astype(bool)
    if sample_id != npz_path.stem:
        raise ValueError(f"NPZ 编号 {sample_id} 与文件名 {npz_path.stem} 不一致")
    duration = float(len(wave) / SAMPLE_RATE)
    hit = np.flatnonzero(np.char.lower(words) == anchor_word.lower())
    if not len(hit):
        raise ValueError(f"样本中没有锚点词 {anchor_word!r}")
    anchor_i = int(hit[0])
    anchor = intervals[anchor_i]
    bin_edges = np.linspace(0.0, duration, 51)
    lanes = assign_lanes(intervals)
    # Validity in the shared 50-bin strip is derived from the same timestamps
    # shown above, rather than from arbitrary per-bin imputation.
    text_valid = np.asarray([
        bool(np.any((intervals[:, 1] > bin_edges[j]) & (intervals[:, 0] < bin_edges[j + 1])))
        for j in range(50)
    ])
    audio_valid = np.asarray([
        bool(np.any((rms_t >= bin_edges[j]) & (rms_t < bin_edges[j + 1]) & (rms > 1e-5)))
        for j in range(50)
    ])
    video_valid = np.asarray([
        bool(np.any((video_frame_times >= bin_edges[j]) & (video_frame_times < min(bin_edges[j + 1], duration))))
        for j in range(50)
    ])
    return {
        "wave": wave, "rms_t": rms_t, "rms": rms, "wave_t": downsample_wave(wave)[0],
        "wave_y": downsample_wave(wave)[1], "words": words, "intervals": intervals,
        "sample_id": sample_id, "label": label, "annotation": annotation,
        "audio_frame_times": audio_frame_times, "video_frame_times": video_frame_times,
        "vision_valid": vision_valid, "face_valid": face_valid,
        "duration": duration, "anchor_i": anchor_i, "anchor": anchor, "anchor_word": words[anchor_i],
        "anchor_bins": np.flatnonzero((bin_edges[:-1] < anchor[1]) & (bin_edges[1:] > anchor[0])),
        "bin_edges": bin_edges, "lanes": lanes,
        "validity": [text_valid, audio_valid, video_valid],
    }


def style_axis(ax, duration: float, grids: np.ndarray) -> None:
    ax.set_xlim(0, duration)
    ax.set_xticks(np.linspace(0, duration, 8))
    ax.grid(axis="x", color=GRID, lw=.55, alpha=.9)
    ax.set_axisbelow(True)
    for x in grids:
        ax.axvline(x, color=GRID, lw=.45, alpha=.75, zorder=0)
    ax.tick_params(axis="both", colors=INK, labelsize=8, length=2)
    ax.spines["left"].set_color(INK); ax.spines["bottom"].set_color(INK)


def highlight(ax, a: float, b: float, ymin: float, ymax: float) -> None:
    ax.axvspan(a, b, color=LIGHT_RED, alpha=.9, zorder=0)
    ax.vlines([a, b], ymin, ymax, color=RED, lw=1.0, ls=(0, (4, 2)), zorder=5)


def render(payload: dict, video: Path, out: Path, anchor_word: str) -> None:
    configure_font()
    d = payload["duration"]
    anchor_a, anchor_b = payload["anchor"]
    grids = payload["bin_edges"][1:-1]
    fig = plt.figure(figsize=(16.5, 10.7), facecolor=BACKGROUND)
    gs = fig.add_gridspec(4, 1, left=.075, right=.965, top=.82, bottom=.115,
                          height_ratios=[1.16, .92, 1.16, .54], hspace=.44)

    # Header mirrors the reference's strong navy title and red anchor callout.
    fig.text(.075, .964, "问题一｜三模态时序对齐", color=INK, fontsize=22, fontweight="bold", va="top")
    fig.text(.075, .924, f"{payload['sample_id']}   ·   {d:.2f} 秒   ·   情感标注 {payload['label']:+.2f}（{payload['annotation']}）",
             color=GREY, fontsize=10.5, va="top")
    bins = ", ".join(str(int(x) + 1) for x in payload["anchor_bins"])
    fig.text(.075, .887, f"锚点词“{payload['anchor_word']}”  [{anchor_a:.3f}, {anchor_b:.3f}] 秒  →  时间箱 {bins}",
             color=RED, fontsize=12.5, fontweight="bold", va="top")

    # A. Text bars with the exact aligned intervals.
    ax = fig.add_subplot(gs[0])
    style_axis(ax, d, grids)
    ax.set_ylim(-.05, 1.0); ax.set_yticks([]); ax.set_ylabel("A  文本", rotation=0, labelpad=35,
                                                            color=INK, fontsize=11, fontweight="bold", va="top")
    highlight(ax, anchor_a, anchor_b, -.05, 1.0)
    lane_y = [.18, .42, .66, .88]
    bar_h = .026
    for i, (word, (a, b), lane) in enumerate(zip(payload["words"], payload["intervals"], payload["lanes"])):
        selected = i == payload["anchor_i"]
        color = RED if selected else BLUE
        y = lane_y[int(lane)]
        ax.plot([a, b], [y, y], color=color, lw=4.5, solid_capstyle="butt", zorder=7)
        ax.text((a + b) / 2, y + .045, word, ha="center", va="bottom", fontsize=7.6,
                color=color, fontweight="bold" if selected else "normal", zorder=8)
        ax.text((a + b) / 2, y - .052, f"{a:.2f}–{b:.2f}", ha="center", va="top", fontsize=5.8,
                color=GREY, zorder=8)
    ax.set_xlabel("片段时间（秒）", color=GREY, fontsize=8.5, labelpad=6)

    # B. True waveform and RMS from the embedded audio track.
    ax = fig.add_subplot(gs[1], sharex=ax)
    style_axis(ax, d, grids)
    amplitude_limit = max(float(np.max(np.abs(payload["wave"]))), .01) * 1.05
    ax.set_ylim(-amplitude_limit, amplitude_limit); ax.set_ylabel("B  语音", rotation=0, labelpad=35, color=INK, fontsize=11,
                                           fontweight="bold", va="top"); ax.set_xlabel("")
    highlight(ax, anchor_a, anchor_b, -amplitude_limit, amplitude_limit)
    ax.plot(payload["wave_t"], payload["wave_y"], color=SLATE, lw=.42, alpha=.7, label="原始波形", zorder=2)
    ax.plot(payload["rms_t"], payload["rms"], color=SEAFOAM, lw=1.55, label="RMS（25 ms / 10 ms）", zorder=4)
    ax.legend(loc="upper right", frameon=True, framealpha=.9, edgecolor=GRID, fontsize=8)
    # The source/reproducibility statement is kept in the accompanying JSON;
    # leaving the waveform panel clean prevents annotation text from covering
    # high-amplitude regions.

    # C. Four raw video frames, one of which is the selected interval.
    ax = fig.add_subplot(gs[2], sharex=ax)
    style_axis(ax, d, grids)
    ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_ylabel("C  视频帧", rotation=0, labelpad=35, color=INK,
                                                      fontsize=11, fontweight="bold", va="top")
    ax.set_xlabel("")
    highlight(ax, anchor_a, anchor_b, 0, 1)
    frame_times = np.array([.55, 2.35, float((anchor_a + anchor_b) / 2), min(d - .35, 7.35)])
    frame_times = np.unique(np.clip(frame_times, .05, max(.06, d - .05)))
    for j, ts in enumerate(frame_times):
        frame = read_frame(video, float(ts))
        if frame is None:
            continue
        selected = anchor_a <= ts <= anchor_b
        x_norm = float(ts / d)
        iax = ax.inset_axes([x_norm - .085, .19, .17, .68], transform=ax.transAxes)
        iax.imshow(frame); iax.axis("off")
        for spine in iax.spines.values():
            spine.set_visible(True); spine.set_color(RED if selected else BLUE); spine.set_linewidth(1.6)
            iax.set_title(f"视频帧｜{ts:.2f} 秒", fontsize=7.3,
                          color=CORAL_BLOOM if selected else LAVENDER, pad=3)
        ax.plot([ts, ts], [.08, .19], color=RED if selected else BLUE, lw=.9)
        ax.scatter([ts], [.07], s=15, color=RED if selected else BLUE, zorder=7)
    ax.text(.005, 1.08, "帧图取自原始视频；桃色标记为锚点词区间中点", transform=ax.transAxes,
            fontsize=7.6, color=GREY, va="top")

    # D. Shared 50-bin representation with explicit missingness.
    # The bottom strip uses bin coordinates (0..50), so it intentionally has a
    # separate x-axis from the three panels whose x-axis is seconds.
    ax = fig.add_subplot(gs[3])
    ax.set_xlim(0, 50); ax.set_ylim(-.5, 2.5); ax.set_yticks([2, 1, 0], ["文本", "语音", "视觉"])
    ax.tick_params(axis="y", labelsize=8, colors=INK, length=0)
    ax.set_xlabel("共享时间箱编号（1—50）；蓝色=有效观测，浅灰=缺失", color=GREY, fontsize=8.5, labelpad=7)
    for r, valid in enumerate(payload["validity"]):
        y = 2 - r
        for k in range(50):
            ax.add_patch(Rectangle((k, y - .38), .96, .76, facecolor=BLUE if valid[k] else MISSING,
                                   edgecolor="white", linewidth=.25))
    for k in payload["anchor_bins"]:
        for y in (0, 1, 2):
            ax.add_patch(Rectangle((k, y - .38), .96, .76, fill=False, edgecolor=RED, linewidth=1.3))
    tick_bins = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax.set_xticks(tick_bins - .5, [str(x) for x in tick_bins], fontsize=7.5)
    ax.grid(False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK); ax.spines["bottom"].set_color(INK)
    ax.text(0, 2.46, "D  共享 50 时间箱表示", color=INK, fontsize=11, fontweight="bold", va="bottom")

    fig.text(.075, .045,
             "桃色=锚点区间   ·   蓝色=有效观测   ·   浅灰=缺失   ·   词界来自 wav2vec2 CTC 强制对齐",
             fontsize=8, color=GREY, ha="left")
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "q1_fig11_典型样本三模态时序对齐_现代版"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_data(payload: dict, video: Path, data_root: Path, out: Path) -> None:
    stem = "q1_fig11_典型样本三模态时序对齐_现代版"
    with (out / f"{stem}_逐词.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["词序号", "词语", "开始_秒", "结束_秒", "文本显示层", "是否锚点"])
        for i, (word, (a, b), lane) in enumerate(zip(payload["words"], payload["intervals"], payload["lanes"]), 1):
            writer.writerow([i, word, f"{a:.6f}", f"{b:.6f}", int(lane) + 1, int(i - 1 == payload["anchor_i"])])
    bins = payload["bin_edges"]
    np.savez_compressed(out / f"{stem}_共享时间箱.npz", sample_id=np.asarray(payload["sample_id"]),
                        bin_edges=bins.astype(np.float32), text_valid=payload["validity"][0],
                        audio_valid=payload["validity"][1], vision_valid=payload["validity"][2],
                        anchor_bins=payload["anchor_bins"].astype(np.int32))
    audit = {
        "route": "B", "sample_id": payload["sample_id"], "anchor_word": str(payload["anchor_word"]),
        "anchor_interval_sec": [float(x) for x in payload["anchor"]],
        "anchor_bins_1_based": [int(x) + 1 for x in payload["anchor_bins"]],
        "duration_sec": payload["duration"], "word_count": int(len(payload["words"])),
        "audio_sample_rate_hz": SAMPLE_RATE, "audio_sample_count": int(len(payload["wave"])),
        "waveform_rms": float(np.sqrt(np.mean(payload["wave"].astype(float) ** 2))),
        "waveform_peak": float(np.max(np.abs(payload["wave"]))),
        "video_frame_count_in_effective_duration": int(np.sum(payload["video_frame_times"] < payload["duration"])),
        "face_blendshape_valid_word_count": int(payload["face_valid"].sum()),
        "source_video_sha256": sha256_file(video),
        "source_video_relative": video.relative_to(data_root).as_posix(),
        "derived_only": True,
    }
    (out / f"{stem}_核验.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--sample-id", default=DEFAULT_SAMPLE)
    p.add_argument("--anchor-word", default="economic")
    args = p.parse_args()
    video = source_video(args.data_root, args.sample_id)
    npz_path = args.features / f"{args.sample_id}.npz"
    payload = build_payload(video, npz_path, args.anchor_word)
    render(payload, video, args.out, args.anchor_word)
    write_data(payload, video, args.data_root, args.out)
    print(json.dumps({"sample_id": payload["sample_id"], "duration_sec": payload["duration"],
                      "word_count": len(payload["words"]), "anchor_word": payload["anchor_word"],
                      "figure": str(args.out / "q1_fig11_典型样本三模态时序对齐_现代版.png")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
