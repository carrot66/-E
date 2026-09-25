#!/usr/bin/env python3
"""路线 B：从附件 1 原始 MP4 复核问题一的音视频特征与时序对齐。

本脚本不把原始视频复制到仓库。它逐条读取原始 MP4（音频从容器中解码为
16 kHz 单声道 PCM），保存来源哈希、容器时长、解码波形统计、视频采样统计，
并与问题一已经生成的 NPZ 和逐词时间区间核对。脚本还生成一个典型样本证据图：
真实波形、逐帧 RMS 与词区间、以及原始视频帧缩略图均来自同一个 MP4。

在 38001 服务器上运行示例：
  PYTHONPATH=question1 /root/user/.../venvs/q1mp/bin/python \\
    question1/q1_routeb_raw_media_audit.py --data-root 'E题数据/E题数据'
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "E题数据" / "E题数据"
DEFAULT_FEATURES = ROOT / "outputs" / "question1" / "问题1_全量特征结果" / "features"
DEFAULT_MANIFEST = ROOT / "outputs" / "question1" / "问题1_全量特征结果" / "sample_manifest.csv"
DEFAULT_OUT = ROOT / "outputs" / "question1" / "问题1_路线B原始媒体审计"
DEFAULT_SAMPLE = "-wny0OAz3g8__7"
SAMPLE_RATE = 16_000
HOP = 160
WINDOW = 400
BLUE = "#4472A8"
RED = "#B86F69"
GREEN = "#4D8E87"
PURPLE = "#7B6AA8"
INK = "#263238"
GREY = "#59656B"


def configure_font() -> None:
    required = {ord(c) for c in "问题一路线原始媒体审计典型样本真实波形音频视频时间秒振幅词语"}
    for family in ("Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei"):
        fonts = [f for f in font_manager.fontManager.ttflist if f.name == family]
        if any(required.issubset(FT2Font(f.fname).get_charmap()) for f in fonts):
            mpl.rcParams.update({
                "font.family": "sans-serif", "font.sans-serif": [family, "DejaVu Sans"],
                "axes.unicode_minus": False, "svg.fonttype": "path", "pdf.fonttype": 42,
                "axes.spines.top": False, "axes.spines.right": False,
                "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white",
                "savefig.facecolor": "white",
            })
            return
    # DejaVu is a usable fallback for headless systems; the audit still remains valid.
    mpl.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def decode_audio(path: Path) -> np.ndarray:
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path),
           "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if p.returncode != 0 or len(p.stdout) < SAMPLE_RATE // 2 * 4:
        msg = p.stderr.decode(errors="replace")[-500:]
        raise RuntimeError(f"ffmpeg 音频解码失败: {msg}")
    wave = np.frombuffer(p.stdout, dtype="<f4").copy()
    return np.nan_to_num(wave).astype(np.float32, copy=False)


def ffprobe(path: Path) -> dict[str, Any]:
    cmd = ["ffprobe", "-v", "error", "-show_entries",
           "format=duration,size:stream=codec_type,codec_name,sample_rate,channels,width,height,r_frame_rate,nb_frames",
           "-of", "json", str(path)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if p.returncode != 0:
        return {}
    try:
        raw = json.loads(p.stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}
    streams = raw.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    fmt = raw.get("format", {})

    def num(x: Any, default: float = float("nan")) -> float:
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    return {
        "container_duration_sec": num(fmt.get("duration")),
        "container_size_bytes": int(num(fmt.get("size"), 0)),
        "video_codec": video.get("codec_name", ""),
        "video_width": int(num(video.get("width"), 0)),
        "video_height": int(num(video.get("height"), 0)),
        "video_fps": num(video.get("r_frame_rate", "0/1").split("/")[0]) /
                     max(num(video.get("r_frame_rate", "0/1").split("/")[1], 1.0), 1.0)
                     if "/" in str(video.get("r_frame_rate", "")) else num(video.get("r_frame_rate")),
        "video_frames_probe": int(num(video.get("nb_frames"), 0)),
        "audio_codec": audio.get("codec_name", ""),
        "audio_sample_rate_hz": int(num(audio.get("sample_rate"), 0)),
        "audio_channels": int(num(audio.get("channels"), 0)),
    }


def video_info(path: Path) -> tuple[float, float, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV 无法打开视频: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0 or frames <= 0:
        raise RuntimeError(f"视频帧率或帧数无效: {path}")
    return frames / fps, fps, frames, width * 100000 + height


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"清单为空: {path}")
    return rows


def source_video(media_root: Path, sample_id: str) -> Path:
    video_id, clip_id = sample_id.rsplit("__", 1)
    direct = media_root / video_id / f"{clip_id}.mp4"
    if direct.is_file():
        return direct
    found = list(media_root.rglob(f"{video_id}/{clip_id}.mp4"))
    if len(found) != 1:
        raise FileNotFoundError(f"样本 {sample_id} 未唯一匹配原始 MP4，找到 {len(found)} 条")
    return found[0]


def wave_stats(wave: np.ndarray) -> dict[str, float]:
    if not len(wave):
        return {"rms": 0.0, "peak": 0.0, "active_start": 0.0, "active_end": 0.0}
    rms = float(np.sqrt(np.mean(np.square(wave, dtype=np.float64))))
    peak = float(np.max(np.abs(wave)))
    n = len(wave) // HOP
    if n:
        padded = wave[: n * HOP].reshape(n, HOP)
        env = np.sqrt(np.mean(np.square(padded, dtype=np.float64), axis=1))
        active = np.flatnonzero(env >= 1e-5)
    else:
        active = np.empty(0, dtype=int)
    if len(active):
        lo = float(active[0] * HOP / SAMPLE_RATE)
        hi = float(min(len(wave) / SAMPLE_RATE, (active[-1] + 1) * HOP / SAMPLE_RATE))
    else:
        lo, hi = 0.0, float(len(wave) / SAMPLE_RATE)
    return {"rms": rms, "peak": peak, "active_start": lo, "active_end": hi}


def safe_float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def npz_check(npz_path: Path, raw_duration: float, wave: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "feature_npz_exists": npz_path.is_file(),
        "word_count": 0, "audio_frame_count": 0, "video_frame_count": 0,
        "word_times_in_bounds": False, "audio_stats_match": False,
        "npz_duration_sec": float("nan"), "npz_audio_rms": float("nan"),
        "npz_audio_peak": float("nan"), "npz_audio_present": False,
        "first_word_sec": float("nan"), "last_word_sec": float("nan"),
    }
    if not npz_path.is_file():
        return result
    with np.load(npz_path, allow_pickle=False) as z:
        keys = set(z.files)
        words = z["words"].astype(str) if "words" in keys else np.empty(0, dtype=str)
        intervals = z["word_times"].astype(float) if "word_times" in keys else np.empty((0, 2))
        at = z["audio_frame_times"].astype(float) if "audio_frame_times" in keys else np.empty(0)
        vt = z["video_frame_times"].astype(float) if "video_frame_times" in keys else np.empty(0)
        result.update({
            "word_count": int(len(words)), "audio_frame_count": int(len(at)),
            "video_frame_count": int(len(vt)),
            "npz_audio_rms": float(np.asarray(z["audio_rms"]).reshape(-1)[0]) if "audio_rms" in keys else float("nan"),
            "npz_audio_peak": float(np.asarray(z["audio_peak"]).reshape(-1)[0]) if "audio_peak" in keys else float("nan"),
            "npz_audio_present": bool(np.asarray(z["audio_present"]).reshape(-1)[0]) if "audio_present" in keys else False,
        })
        if intervals.size:
            result["first_word_sec"] = float(intervals[0, 0])
            result["last_word_sec"] = float(intervals[-1, 1])
            result["word_times_in_bounds"] = bool(
                np.all(np.isfinite(intervals)) and np.all(intervals[:, 0] >= -1e-4) and
                np.all(intervals[:, 1] <= raw_duration + 0.05) and
                np.all(intervals[:, 1] >= intervals[:, 0]) and
                np.all(np.diff(intervals[:, 0]) >= -1e-4) and
                np.all(intervals[:-1, 1] <= intervals[1:, 0] + 0.05)
            )
        raw_rms = float(np.sqrt(np.mean(np.square(wave, dtype=np.float64)))) if len(wave) else 0.0
        raw_peak = float(np.max(np.abs(wave))) if len(wave) else 0.0
        result["audio_stats_match"] = bool(
            np.isfinite(result["npz_audio_rms"]) and np.isfinite(result["npz_audio_peak"]) and
            abs(result["npz_audio_rms"] - raw_rms) <= 2e-5 and
            abs(result["npz_audio_peak"] - raw_peak) <= 2e-4
        )
        if "duration_video_sec" in keys:
            result["npz_duration_sec"] = float(np.asarray(z["duration_video_sec"]).reshape(-1)[0])
        elif "duration_sec" in keys:
            result["npz_duration_sec"] = float(np.asarray(z["duration_sec"]).reshape(-1)[0])
    return result


def sample_frame(path: Path, time_sec: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(time_sec)) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def representative_payload(video: Path, npz_path: Path) -> dict[str, Any]:
    wave = decode_audio(video)
    with np.load(npz_path, allow_pickle=False) as z:
        words = z["words"].astype(str)
        intervals = z["word_times"].astype(float)
        video_times = z["video_frame_times"].astype(float)
    # Pick four evenly spaced word intervals so the frame montage covers the clip.
    ids = np.unique(np.linspace(0, max(0, len(words) - 1), min(4, len(words))).round().astype(int))
    frame_times = np.asarray([float(intervals[i].mean()) for i in ids], dtype=float)
    frames: list[np.ndarray] = []
    valid_times: list[float] = []
    for t in frame_times:
        f = sample_frame(video, t)
        if f is not None:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            valid_times.append(t)
    return {
        "wave": wave, "words": words, "intervals": intervals,
        "video_times": video_times, "frame_word_ids": ids,
        "frames": frames, "frame_times": np.asarray(valid_times, dtype=float),
    }


def render_evidence(payload: dict[str, Any], sample_id: str, out: Path) -> None:
    configure_font()
    wave = payload["wave"]
    intervals = payload["intervals"]
    words = payload["words"]
    duration = len(wave) / SAMPLE_RATE
    # Plot at most 20k points for a crisp PDF and a light SVG.
    stride = max(1, int(math.ceil(len(wave) / 20_000)))
    t = np.arange(0, len(wave), stride) / SAMPLE_RATE
    y = wave[::stride]
    hop_n = max(1, len(wave) // HOP)
    rms = np.sqrt(np.mean(np.square(wave[:hop_n * HOP].reshape(hop_n, HOP), dtype=np.float64), axis=1))
    rt = np.arange(len(rms)) * HOP / SAMPLE_RATE

    fig = plt.figure(figsize=(13.8, 8.2), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=.38, wspace=.22)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t, y, lw=.45, color=BLUE, alpha=.82, rasterized=True)
    ax.plot(rt, rms, lw=1.15, color=RED, alpha=.95, label="10 ms RMS 包络")
    for i, (a, b) in enumerate(intervals):
        ax.axvspan(a, b, color=GREEN, alpha=.07, lw=0)
        if i in payload["frame_word_ids"]:
            ax.text((a + b) / 2, ax.get_ylim()[1] * .82, words[i], fontsize=8,
                    ha="center", va="center", color=INK, rotation=35)
    ax.set(xlim=(0, duration), xlabel="时间（秒）", ylabel="波形振幅", title="a  原始 MP4 解码波形与词区间")
    ax.legend(loc="upper right", fontsize=8)
    ax.text(0.005, .93, "绿色阴影：CTC 词级时间区间；红线：同一波形计算的 RMS", transform=ax.transAxes,
            fontsize=8, color=GREY, va="top")

    ax2 = fig.add_subplot(gs[1, 0])
    k = np.arange(len(intervals)) + 1
    ax2.plot(k, intervals[:, 0], "o-", ms=3.5, lw=1.1, color=PURPLE, label="开始时刻")
    ax2.plot(k, intervals[:, 1], "s--", ms=3.2, lw=1.0, color=RED, label="结束时刻")
    ax2.set(xlabel="对齐位置 k（词元）", ylabel="时间（秒）", title="b  词级边界与原始媒体时长")
    ax2.set_xlim(.5, len(k) + .5)
    ax2.legend(fontsize=8, loc="best")
    # Keep the panel clean for manuscript export; the two curves and labeled
    # axes already carry the timing information, so no gridline is needed.
    ax2.grid(False)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    frames = payload["frames"]
    if frames:
        n = len(frames)
        inner = ax3.get_position()
        # inset axes avoid any artificial white divider lines between image panels
        for j, (frame, ts) in enumerate(zip(frames, payload["frame_times"])):
            x0 = inner.x0 + j * inner.width / n
            w = inner.width / n * .96
            iax = fig.add_axes([x0, inner.y0 + .06 * inner.height, w, inner.height * .83])
            iax.imshow(frame)
            iax.set_title(f"{words[int(payload['frame_word_ids'][j])]}  {ts:.2f} 秒", fontsize=8, pad=4)
            iax.axis("off")
        ax3.set_title("c  原始视频帧（词区间中点）", pad=12)
    else:
        ax3.text(.5, .5, "未读取到视频帧", ha="center", va="center")

    fig.suptitle("问题一：路线 B 原始音视频取证与时序对应", fontsize=14, fontweight="bold", y=.985, color=INK)
    fig.text(.5, .018, f"样本 {sample_id}；波形由原始 MP4 解码为 16 kHz 单声道 PCM；视频帧取词区间中点",
             ha="center", va="center", fontsize=8.5, color=GREY)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "q1_fig10_路线B原始音视频取证"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_representative_data(payload: dict[str, Any], sample_id: str, video: Path, media_root: Path, out: Path) -> None:
    np.savez_compressed(out / "问题1_路线B典型样本波形数据.npz", sample_id=np.asarray(sample_id),
                        waveform=payload["wave"].astype(np.float32), sample_rate_hz=np.asarray(SAMPLE_RATE),
                        words=payload["words"], word_times=payload["intervals"].astype(np.float32),
                        video_frame_times=payload["video_times"].astype(np.float32),
                        evidence_frame_times=payload["frame_times"].astype(np.float32),
                        evidence_word_indices=payload["frame_word_ids"].astype(np.int32))
    row = {
        "样本编号": sample_id,
        "原始视频相对路径": video.relative_to(media_root.parent.parent).as_posix(),
        "原始视频SHA256": sha256_file(video),
        "证据帧时刻_秒": ";".join(f"{float(t):.4f}" for t in payload["frame_times"]),
        "证据词元": ";".join(str(payload["words"][int(i)]) for i in payload["frame_word_ids"]),
        "说明": "波形和视频帧均由原始MP4现场解码；仓库不包含原始MP4",
    }
    with (out / "问题1_路线B典型样本证据清单.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row), lineterminator="\n")
        writer.writeheader(); writer.writerow(row)


def audit(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    media_root = args.data_root / "附件1-数据集原始多模态样本" / "MOSEI数据集部分原始视频-100条"
    rows = read_manifest(args.manifest)
    out_rows: list[dict[str, Any]] = []
    raw_found = 0
    npz_found = 0
    duration_diffs: list[float] = []
    stat_matches = 0
    bounds_ok = 0
    for index, manifest_row in enumerate(rows, 1):
        sid = manifest_row.get("sample_id", "").strip()
        raw_path = source_video(media_root, sid)
        raw_found += 1
        probe = ffprobe(raw_path)
        vdur, vfps, vframes, wh = video_info(raw_path)
        width, height = divmod(wh, 100000)
        wave = decode_audio(raw_path)
        wstat = wave_stats(wave)
        npz = npz_check(args.features / f"{sid}.npz", min(vdur, len(wave) / SAMPLE_RATE), wave)
        npz_found += int(npz["feature_npz_exists"])
        q1_duration = safe_float(manifest_row, "duration_video_sec")
        duration_diff = float(q1_duration - min(vdur, len(wave) / SAMPLE_RATE)) if np.isfinite(q1_duration) else float("nan")
        if np.isfinite(duration_diff): duration_diffs.append(abs(duration_diff))
        stat_matches += int(npz["audio_stats_match"])
        bounds_ok += int(npz["word_times_in_bounds"])
        row = {
            "样本编号": sid,
            "原始视频相对路径": raw_path.relative_to(args.data_root).as_posix(),
            "原始文件SHA256": sha256_file(raw_path),
            "文件字节数": raw_path.stat().st_size,
            "容器时长_秒": round(probe.get("container_duration_sec", float("nan")), 6),
            "OpenCV视频时长估计_秒": round(vdur, 6), "音频解码时长_秒": round(len(wave) / SAMPLE_RATE, 6),
            "问题一有效时长_秒": round(min(vdur, len(wave) / SAMPLE_RATE), 6),
            "问题一清单时长_秒": q1_duration,
            "有效时长差_秒": round(duration_diff, 6) if np.isfinite(duration_diff) else "",
            "原始视频帧率": round(vfps, 6), "原始视频帧数": vframes,
            "有效时长内10fps帧数": int(math.ceil(min(vdur, len(wave) / SAMPLE_RATE) * 10 - 1e-9)),
            "视频宽": width, "视频高": height,
            "原始音频编码": probe.get("audio_codec", ""), "原始音频采样率_Hz": probe.get("audio_sample_rate_hz", 0),
            "原始音频声道数": probe.get("audio_channels", 0), "解码采样率_Hz": SAMPLE_RATE,
            "解码音频采样点数": len(wave), "波形RMS": round(wstat["rms"], 10),
            "波形峰值": round(wstat["peak"], 10), "波形有效起点_秒": round(wstat["active_start"], 6),
            "波形有效终点_秒": round(wstat["active_end"], 6), "NPZ是否存在": bool(npz["feature_npz_exists"]),
            "词数": npz["word_count"], "NPZ音频帧数": npz["audio_frame_count"],
            "NPZ视频帧数": npz["video_frame_count"], "首词起点_秒": npz["first_word_sec"],
            "末词终点_秒": npz["last_word_sec"], "词区间均在原始时长内": bool(npz["word_times_in_bounds"]),
            "NPZ音频统计与原始波形一致": bool(npz["audio_stats_match"]),
            "核验状态": "通过" if npz["feature_npz_exists"] and npz["word_times_in_bounds"] and npz["audio_stats_match"] else "需复核",
        }
        out_rows.append(row)
        if index % 10 == 0: print(f"[路线B] 已核验 {index}/{len(rows)}")
    passed = (len(rows) == 100 and raw_found == 100 and npz_found == 100 and
              bounds_ok == len(rows) and stat_matches == len(rows))
    summary = {
        "route": "B",
        "route_description": "原始 MP4/内嵌音频为来源；所有派生特征与词级区间均逐条回指原始媒体",
        "sample_count_in_manifest": len(rows), "raw_media_found": raw_found,
        "feature_npz_found": npz_found, "word_alignment_in_bounds": bounds_ok,
        "waveform_statistics_match": stat_matches,
        "max_abs_effective_duration_diff_sec": max(duration_diffs) if duration_diffs else None,
        "mean_abs_effective_duration_diff_sec": float(np.mean(duration_diffs)) if duration_diffs else None,
        "passed": passed, "source_media_uploaded": False,
        "source_media_note": "原始 MP4 仅保留在 38001 数据目录；GitHub 仅提交哈希、汇总和可复现实验脚本",
        "representative_sample": args.sample_id,
    }
    return out_rows, summary, {"media_root": media_root, "rows": rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--sample-id", default=DEFAULT_SAMPLE)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, summary, context = audit(args)
    fields = list(rows[0]) if rows else []
    with (args.out / "问题1_路线B原始媒体审计.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    (args.out / "问题1_路线B原始媒体审计.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    sample_video = source_video(context["media_root"], args.sample_id)
    sample_npz = args.features / f"{args.sample_id}.npz"
    payload = representative_payload(sample_video, sample_npz)
    render_evidence(payload, args.sample_id, args.out)
    write_representative_data(payload, args.sample_id, sample_video, context["media_root"], args.out)
    (args.out / "README.md").write_text(
        "# 问题一：路线 B 原始媒体审计\n\n"
        "本目录的 CSV/JSON 是附件 1 的 100 条原始 MP4 逐条核验结果。原始媒体不随 Git 提交，"
        "可在 38001 的题目数据目录按 `原始视频相对路径` 复现；SHA256 用于确认来源没有被替换。"
        "CSV 中的 `OpenCV视频时长估计_秒` 只是与特征提取器一致的帧数/帧率估计，音频时长以实际解码采样点数为准。\n\n"
        "- `问题1_路线B原始媒体审计.csv`：100 条样本的容器信息、16 kHz 解码波形统计、NPZ/词界核验。\n"
        "- `问题1_路线B原始媒体审计.json`：通过条件与总体统计。\n"
        "- `q1_fig10_路线B原始音视频取证.*`：典型样本的真实波形、词界和原始视频帧。\n"
        "- `问题1_路线B典型样本波形数据.npz`：仅保存典型样本的派生波形与时间轴，不含原始 MP4。\n\n"
        "复现：`PYTHONPATH=question1 python question1/q1_routeb_raw_media_audit.py --data-root 'E题数据/E题数据'`。\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
