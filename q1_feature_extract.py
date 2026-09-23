#!/usr/bin/env python3
"""Extract word-aligned text, audio, and visual features for MOSEI E Question 1.

Models are generic pretrained representation models, not emotion classifiers:
  * google-bert/bert-base-uncased: token-level language representation (768)
  * facebook/wav2vec2-base-960h: speech representation (768) and CTC alignment
  * torchvision ResNet-18 ImageNet weights: face/frame appearance representation (512)

In addition, a transparent 74-D acoustic descriptor and 49-D face appearance
descriptor are extracted. CTC alignment uses the supplied transcript and speech
recognizer vocabulary. Every feature row is aligned to a word interval in the
original clip. Missing/failed steps are logged; no sample is silently dropped.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import openpyxl
import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel, AutoModelForCTC, AutoTokenizer
from torchvision.models import ResNet18_Weights, resnet18

TEXT_MODEL = "google-bert/bert-base-uncased"
SPEECH_MODEL = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000
CTC_MIN_MEAN_LOGPROB = -3.0
FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
BLENDSHAPE_DIM = 52
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*")
LOGGER = logging.getLogger("q1")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure_face_landmarker(path: Path) -> Path:
    """Download the official MediaPipe face landmark/blendshape task if absent."""
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.is_file() or path.stat().st_size < 1_000_000:
        tmp=path.with_suffix(path.suffix+".download")
        LOGGER.info("downloading MediaPipe Face Landmarker model to %s",path)
        urllib.request.urlretrieve(FACE_LANDMARKER_URL,tmp)
        if tmp.stat().st_size < 1_000_000:
            tmp.unlink(missing_ok=True)
            raise RuntimeError("downloaded Face Landmarker model is unexpectedly small")
        tmp.replace(path)
    return path


def find_inputs(data_root: Path) -> tuple[Path, Path]:
    xlsx = list(data_root.rglob("label-100.xlsx"))
    if len(xlsx) != 1:
        raise FileNotFoundError(f"Expected one label-100.xlsx under {data_root}; found {len(xlsx)}")
    video_dirs = [p for p in data_root.rglob("MOSEI数据集部分原始视频-100条") if p.is_dir()]
    if len(video_dirs) != 1:
        raise FileNotFoundError(f"Expected one raw-video folder; found {len(video_dirs)}")
    return xlsx[0], video_dirs[0]


def clean_id(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def load_labels(path: Path) -> list[dict[str, Any]]:
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = ws.iter_rows(values_only=True)
    header = [str(v).strip() if v is not None else "" for v in next(rows)]
    index = {name: i for i, name in enumerate(header)}
    required = {"video_id", "clip_id", "text", "label", "annotation"}
    if not required.issubset(index):
        raise ValueError(f"label file lacks columns: {sorted(required - set(index))}")
    records = []
    seen = set()
    for row in rows:
        vid, clip = clean_id(row[index["video_id"]]), clean_id(row[index["clip_id"]])
        if not vid or not clip:
            continue
        key = (vid, clip)
        if key in seen:
            raise ValueError(f"Duplicate video_id/clip_id in labels: {key}")
        seen.add(key)
        records.append({"video_id": vid, "clip_id": clip,
                        "text": str(row[index["text"]] or ""),
                        "label": row[index["label"]],
                        "annotation": str(row[index["annotation"]] or "")})
    wb.close()
    if len(records) != 100:
        LOGGER.warning("Label rows=%d; competition specification says 100", len(records))
    return records


def resolve_video(video_root: Path, video_id: str, clip_id: str) -> Path | None:
    candidates = [video_root / video_id / f"{clip_id}.mp4",
                  video_root / video_id / f"{clip_id}.MP4"]
    for p in candidates:
        if p.is_file():
            return p
    return None


class Encoders:
    def __init__(self, device: torch.device, cache_dir: Path | None = None):
        self.device = device
        # If HF_ENDPOINT is configured (e.g. hf-mirror.com), both downloads and
        # cached model resolution use the same standard Transformers API.
        common = {"cache_dir": str(cache_dir) if cache_dir else None}
        self.text_tok = AutoTokenizer.from_pretrained(TEXT_MODEL, use_fast=True, **common)
        self.text_model = AutoModel.from_pretrained(TEXT_MODEL, **common).to(device).eval()
        self.speech_proc = AutoTokenizer.from_pretrained(SPEECH_MODEL, **common)
        self.speech_model = AutoModelForCTC.from_pretrained(SPEECH_MODEL, **common).to(device).eval()
        weights = ResNet18_Weights.DEFAULT
        self.vision_model = resnet18(weights=weights).to(device).eval()
        self.vision_model.fc = torch.nn.Identity()
        self.vision_transform = weights.transforms()
        self.max_text_tokens = min(510, int(getattr(self.text_model.config, "max_position_embeddings", 512)) - 2)

    @torch.inference_mode()
    def text_words(self, text: str, words: list[str]) -> tuple[np.ndarray, list[int]]:
        dim = int(self.text_model.config.hidden_size)
        if not words:
            return np.zeros((0, dim), np.float32), []
        enc = self.text_tok(text, add_special_tokens=False, return_offsets_mapping=True,
                            truncation=False)
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        pooled: list[list[np.ndarray]] = [[] for _ in words]
        word_spans = [(m.start(), m.end()) for m in WORD_RE.finditer(text)]
        if len(word_spans) != len(words):
            raise ValueError("word tokenization mismatch")
        for start in range(0, len(ids), self.max_text_tokens):
            stop = min(len(ids), start + self.max_text_tokens)
            chunk_ids = ids[start:stop]
            model_ids = self.text_tok.build_inputs_with_special_tokens(chunk_ids)
            input_ids = torch.tensor([model_ids], dtype=torch.long, device=self.device)
            mask = torch.ones_like(input_ids)
            out = self.text_model(input_ids=input_ids, attention_mask=mask).last_hidden_state[0, 1:1 + len(chunk_ids)]
            token_np = out.float().cpu().numpy()
            for j, (a, b) in enumerate(offsets[start:stop]):
                if b <= a:
                    continue
                # A WordPiece can intersect a token boundary in rare punctuation cases.
                for wi, (wa, wb) in enumerate(word_spans):
                    if a < wb and b > wa:
                        pooled[wi].append(token_np[j])
                        break
        features = np.zeros((len(words), dim), dtype=np.float32)
        counts = []
        for i, pieces in enumerate(pooled):
            counts.append(len(pieces))
            if pieces:
                features[i] = np.mean(pieces, axis=0, dtype=np.float64).astype(np.float32)
        return features, counts

    @torch.inference_mode()
    def audio(self, waveform: np.ndarray, transcript: str, duration: float):
        audio_rms=float(np.sqrt(np.mean(np.square(waveform,dtype=np.float64))))
        audio_peak=float(np.max(np.abs(waveform))) if len(waveform) else 0.0
        if audio_rms < 1e-5 or audio_peak < 1e-4:
            # Silence cannot carry a word alignment. Preserve the row count and
            # mark the entire audio modality missing instead of aligning noise.
            words=list(WORD_RE.finditer(transcript))
            times=proportional_times(transcript,duration)
            active=(0.0,duration)
            hidden=np.zeros((max(1,int(math.ceil(duration/0.02))),768),dtype=np.float32)
            word_conf=np.full(len(words),-1.0,np.float32); word_conf_valid=np.zeros(len(words),bool)
            return hidden,times,np.asarray([duration/len(hidden)],np.float32),"proportional_fallback_silent_audio",float("nan"),word_conf,word_conf_valid,active,audio_rms,audio_peak,False
        active=speech_activity_interval(waveform,duration)
        # Wav2Vec2FeatureExtractor normalizes each waveform to zero mean/unit
        # variance. Applying the same transform is required for its published
        # CTC probabilities and hidden representations.
        norm=(waveform-waveform.mean())/np.sqrt(waveform.var()+1e-7)
        x = torch.from_numpy(norm.astype(np.float32,copy=False)).to(self.device).unsqueeze(0)
        # Compute the encoder representation once; the CTC head supplies a
        # transcript-conditioned forced alignment without a separate aligner.
        with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                            enabled=self.device.type == "cuda"):
            base = self.speech_model.wav2vec2(x, return_dict=True).last_hidden_state
            logits = self.speech_model.lm_head(self.speech_model.dropout(base))
        hidden = base[0].float().cpu().numpy().astype(np.float32)
        logp = logits[0].float().log_softmax(-1).cpu().numpy()
        frame_sec = len(waveform) / SAMPLE_RATE / max(1, len(hidden))
        word_times,method,confidence,word_conf,word_conf_valid=ctc_word_alignment(
            logp, self.speech_proc.get_vocab(), self.speech_proc.pad_token_id,
            transcript, len(list(WORD_RE.finditer(transcript))), duration, active)
        return hidden,word_times,np.asarray([frame_sec],np.float32),method,confidence,word_conf,word_conf_valid,active,audio_rms,audio_peak,True

    @torch.inference_mode()
    def vision(self, crops: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)
        chunks = []
        for i in range(0, len(crops), batch_size):
            tensors = []
            for crop in crops[i:i + batch_size]:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                im = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.
                tensors.append(self.vision_transform(im))
            x = torch.stack(tensors).to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                enabled=self.device.type == "cuda"):
                z = self.vision_model(x)
            chunks.append(z.float().cpu().numpy())
        return np.concatenate(chunks, axis=0).astype(np.float32)


def ctc_word_alignment(logp: np.ndarray, vocab: dict[str, int], blank: int,
                       transcript: str, nwords: int, duration: float,
                       active_bounds: tuple[float,float] | None = None) -> tuple[np.ndarray,str,float,np.ndarray,np.ndarray]:
    words = list(WORD_RE.finditer(transcript))
    if len(words) != nwords:
        nwords = len(words)
    char_ids: list[int] = []
    char_word: list[int] = []
    for wi, match in enumerate(words):
        # The published wav2vec2-base-960h CTC vocabulary uses uppercase
        # English labels. Matching case is required for a true forced path.
        letters = match.group(0).upper()
        for ch in letters:
            if ch in vocab and vocab[ch] != blank:
                char_ids.append(int(vocab[ch])); char_word.append(wi)
        if wi != len(words) - 1 and "|" in vocab:
            char_ids.append(int(vocab["|"])); char_word.append(-1)
    t_count = logp.shape[0]
    fallback = proportional_times(transcript, duration)
    word_conf=np.full(len(words),-1.0,np.float32); word_conf_valid=np.zeros(len(words),bool)
    if not words or not char_ids or len(char_ids) > t_count:
        return fallback,"proportional_fallback",float("nan"),word_conf,word_conf_valid
    ext = np.full(2 * len(char_ids) + 1, blank, dtype=np.int64)
    ext[1::2] = np.asarray(char_ids, dtype=np.int64)
    state_n = len(ext)
    emit = logp[:, ext]
    neg = -1e30
    dp = np.full(state_n, neg, dtype=np.float32)
    dp[0] = emit[0, 0]
    dp[1] = emit[0, 1]
    back = np.zeros((t_count, state_n), dtype=np.int8)
    skip_ok = np.zeros(state_n, dtype=bool)
    if state_n > 2:
        idx = np.arange(2, state_n)
        skip_ok[idx] = (ext[idx] != blank) & (ext[idx] != ext[idx - 2])
    for ti in range(1, t_count):
        stay = dp
        step1 = np.full(state_n, neg, np.float32); step1[1:] = dp[:-1]
        step2 = np.full(state_n, neg, np.float32); step2[2:] = dp[:-2]
        step2[~skip_ok] = neg
        cand = np.stack((stay, step1, step2))
        move = np.argmax(cand, axis=0).astype(np.int8)
        dp = cand[move, np.arange(state_n)] + emit[ti]
        back[ti] = move
    end_candidates = [state_n - 1, max(0, state_n - 2)]
    state = max(end_candidates, key=lambda s: float(dp[s]))
    score = float(dp[state])
    if score <= neg / 2 or not np.isfinite(score):
        return fallback,"proportional_fallback",float("nan"),word_conf,word_conf_valid
    frames: list[list[int]] = [[] for _ in char_ids]
    s = state
    for ti in range(t_count - 1, -1, -1):
        if s % 2 == 1:
            frames[s // 2].append(ti)
        if ti > 0:
            s -= int(back[ti, s])
    intervals = np.full((len(words), 2), np.nan, dtype=np.float32)
    char_conf = []; per_word_char_conf=[[] for _ in words]
    for ci, fr in enumerate(frames):
        if fr:
            wi = char_word[ci]
            if wi >= 0:
                lo, hi = min(fr), max(fr) + 1
                if np.isnan(intervals[wi, 0]): intervals[wi] = [lo, hi]
                else:
                    intervals[wi, 0] = min(intervals[wi, 0], lo)
                    intervals[wi, 1] = max(intervals[wi, 1], hi)
                values=logp[fr,char_ids[ci]].tolist()
                char_conf.extend(values); per_word_char_conf[wi].extend(values)
    intervals *= duration / max(1, t_count)
    intervals = fill_missing_intervals(intervals, transcript, duration)
    if len(intervals):
        intervals[:, 0] = np.clip(intervals[:, 0], 0, max(0, duration - 0.005))
        intervals[:, 1] = np.clip(intervals[:, 1], intervals[:, 0] + 0.005, duration)
    conf = float(np.mean(char_conf)) if char_conf else float("nan")
    for wi,values in enumerate(per_word_char_conf):
        if values:
            word_conf[wi]=float(np.mean(values)); word_conf_valid[wi]=True
    if conf < CTC_MIN_MEAN_LOGPROB:
        # Weak transcript/audio agreement produces degenerate CTC durations.
        # Place words proportionally over the detected speech window and record
        # the fallback so it can be inspected instead of presented as CTC truth.
        lo,hi=active_bounds if active_bounds is not None else (0.0,duration)
        return proportional_times(transcript,duration,lo,hi),"proportional_fallback_low_ctc_confidence",conf,word_conf,word_conf_valid
    return intervals,"wav2vec2_ctc_forced_alignment",conf,word_conf,word_conf_valid


def proportional_times(text: str, duration: float, start: float = 0.0,
                       end: float | None = None) -> np.ndarray:
    words = list(WORD_RE.finditer(text))
    if not words:
        return np.zeros((0, 2), np.float32)
    end=duration if end is None else float(np.clip(end,start,duration))
    start=float(np.clip(start,0,duration))
    weights = np.asarray([max(1, len(re.sub(r"[^A-Za-z]", "", m.group(0)))) for m in words], dtype=np.float64)
    edges = start + np.concatenate(([0.0], np.cumsum(weights) / weights.sum() * (end-start)))
    return np.stack((edges[:-1], edges[1:]), axis=1).astype(np.float32)


def speech_activity_interval(wave: np.ndarray, duration: float) -> tuple[float,float]:
    """Simple energy gate for bounding the speech region used by fallback timing."""
    sr=SAMPLE_RATE; win=400; hop=160
    if len(wave)<=win: return (0.0,duration)
    padded=np.pad(wave,(win//2,win//2),mode="reflect")
    frames=np.lib.stride_tricks.sliding_window_view(padded,win)[::hop]
    rms=np.sqrt(np.mean(frames.astype(np.float64)**2,axis=1))
    peak=float(rms.max()) if len(rms) else 0.0
    if peak<1e-5: return (0.0,duration)
    # Relative threshold adapts to recording gain; the floor prevents codec noise
    # from being mistaken for speech. Gaps under ~200 ms remain inside the span.
    active=rms>max(peak*0.08,1e-4)
    ids=np.flatnonzero(active)
    if not len(ids): return (0.0,duration)
    lo=max(0.0,float(ids[0]*hop/sr)-0.10)
    hi=min(duration,float((ids[-1]*hop+win)/sr)+0.10)
    if hi-lo<0.25 or (duration>0 and (hi-lo)/duration>0.995): return (0.0,duration)
    return lo,hi


def fill_missing_intervals(times: np.ndarray, text: str, duration: float) -> np.ndarray:
    if not len(times): return times.astype(np.float32)
    good = np.isfinite(times[:, 0]) & np.isfinite(times[:, 1]) & (times[:, 1] > times[:, 0])
    fallback = proportional_times(text, duration)
    out = times.copy()
    out[~good] = fallback[~good]
    # Establish monotonic word order; keep measured CTC intervals and repair
    # collisions using their midpoint, then attach unaligned words locally.
    mids = (out[:, 0] + out[:, 1]) / 2
    order = np.argsort(mids, kind="stable")
    if not np.array_equal(order, np.arange(len(out))):
        out = fallback.copy()
        good[:] = False
    for i in range(len(out)):
        if not good[i]:
            left = next((j for j in range(i - 1, -1, -1) if good[j]), None)
            right = next((j for j in range(i + 1, len(out)) if good[j]), None)
            if left is not None and right is not None and times[right, 0] >= times[left, 1]:
                lo, hi = times[left, 1], times[right, 0]
                group = [k for k in range(left + 1, right) if not good[k]]
                weights = [max(1, len(re.sub(r"[^A-Za-z]", "", list(WORD_RE.finditer(text))[k].group(0)))) for k in group]
                total = sum(weights) or 1
                pos = group.index(i)
                before = sum(weights[:pos]) / total
                after = sum(weights[:pos + 1]) / total
                out[i] = [lo + (hi - lo) * before, lo + (hi - lo) * after]
            # otherwise the proportional fallback already supplies a bounded interval
    out[:, 0] = np.maximum.accumulate(np.clip(out[:, 0], 0, duration))
    out[:, 1] = np.clip(np.maximum(out[:, 1], out[:, 0] + 0.005), 0, duration)
    # Empty/very short edge intervals are reset to an ordered proportional partition.
    if np.any(out[:, 1] <= out[:, 0]):
        out = fallback
    return out.astype(np.float32)


def decode_audio(video: Path) -> np.ndarray:
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(video),
           "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if p.returncode != 0 or len(p.stdout) < SAMPLE_RATE // 2 * 4:
        raise RuntimeError(f"ffmpeg audio decode failed: {p.stderr.decode(errors='replace')[-500:]}")
    wave = np.frombuffer(p.stdout, dtype="<f4").copy()
    if not np.all(np.isfinite(wave)):
        wave = np.nan_to_num(wave)
    return wave


def mel_mfcc_prosody(wave: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = torch.from_numpy(wave).float()
    nfft, hop, sr, nmel, ncoef = 400, 160, SAMPLE_RATE, 40, 13
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=nfft, win_length=nfft,
        hop_length=hop, f_min=20, f_max=7600, n_mels=nmel, power=2.0,
        center=True, pad_mode="reflect")(x)
    logmel = torch.log(mel.clamp_min(1e-8))
    n = torch.arange(nmel).float()
    k = torch.arange(ncoef).float().unsqueeze(1)
    dct = torch.cos(math.pi / nmel * (n + 0.5) * k)
    dct[0] *= math.sqrt(1 / nmel)
    dct[1:] *= math.sqrt(2 / nmel)
    mfcc = dct @ logmel
    delta = torchaudio.functional.compute_deltas(mfcc.unsqueeze(0), win_length=5).squeeze(0)

    # 25 ms centered frames, 10 ms hop. autocorrelation peak is a transparent
    # F0 proxy; it is not a clinically calibrated voice-quality measurement.
    pad = nfft // 2
    padded = F.pad(x.view(1, 1, -1), (pad, pad), mode="reflect").view(-1)
    frames = padded.unfold(0, nfft, hop)
    frame_count = min(logmel.shape[1], mfcc.shape[1], delta.shape[1], frames.shape[0])
    frames = frames[:frame_count]
    rms = frames.square().mean(-1).sqrt()
    zcr = ((frames[:, 1:] >= 0) != (frames[:, :-1] >= 0)).float().mean(-1)
    win = torch.hann_window(nfft)
    spec = torch.fft.rfft(frames * win, dim=-1).abs()
    freq = torch.linspace(0, sr / 2, spec.shape[-1])
    centroid = (spec * freq).sum(-1) / spec.sum(-1).clamp_min(1e-8)
    arr = frames.cpu().numpy()
    # Vectorized autocorrelation with zero-padding to 1024 samples.
    fft = np.fft.rfft(arr, n=1024, axis=1)
    ac = np.fft.irfft(fft * np.conjugate(fft), n=1024, axis=1)
    ac /= np.maximum(ac[:, :1], 1e-8)
    lag_lo, lag_hi = int(sr / 400), int(sr / 65)
    section = ac[:, lag_lo:lag_hi + 1]
    lag = section.argmax(axis=1) + lag_lo
    strength = section.max(axis=1)
    f0 = sr / np.maximum(lag, 1)
    voiced = (strength >= 0.30).astype(np.float32)
    f0 = f0.astype(np.float32) * voiced
    pros = np.stack((rms.cpu().numpy(), zcr.cpu().numpy(), f0, voiced,
                     centroid.cpu().numpy()), axis=1).astype(np.float32)
    # Three local first derivatives: RMS, F0, and spectral centroid.
    chosen = pros[:, [0, 2, 4]]
    dpros = np.zeros_like(chosen)
    if frame_count > 1:
        dpros[1:-1] = (chosen[2:] - chosen[:-2]) * 0.5
        dpros[0] = chosen[1] - chosen[0]; dpros[-1] = chosen[-1] - chosen[-2]
    feat = np.concatenate((logmel[:, :frame_count].T.cpu().numpy(),
                           mfcc[:, :frame_count].T.cpu().numpy(),
                           delta[:, :frame_count].T.cpu().numpy(), pros, dpros), axis=1)
    if feat.shape[1] != 74:
        raise RuntimeError(f"internal audio feature size={feat.shape[1]}, expected 74")
    centers = np.arange(frame_count, dtype=np.float32) * hop / sr
    return feat.astype(np.float32), centers


def lbp_uniform_hist(gray: np.ndarray) -> np.ndarray:
    # 8-neighbour uniform LBP (10 bins) in each of four spatial quadrants.
    g = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    center = g[1:-1, 1:-1]
    neigh = [g[0:-2, 0:-2], g[0:-2, 1:-1], g[0:-2, 2:], g[1:-1, 2:],
             g[2:, 2:], g[2:, 1:-1], g[2:, 0:-2], g[1:-1, 0:-2]]
    code = np.zeros(center.shape, dtype=np.uint8)
    bits = []
    for i, q in enumerate(neigh):
        b = (q >= center).astype(np.uint8); bits.append(b); code |= b << i
    transitions = np.zeros(center.shape, dtype=np.uint8)
    for i in range(8): transitions += bits[i] != bits[(i + 1) % 8]
    pop = np.zeros(center.shape, dtype=np.uint8)
    for b in bits: pop += b
    labels = np.where(transitions <= 2, pop, 9)
    hist = []
    h, w = labels.shape
    for ys, xs in ((slice(0, h//2), slice(0, w//2)), (slice(0,h//2), slice(w//2,w)),
                   (slice(h//2,h), slice(0,w//2)), (slice(h//2,h), slice(w//2,w))):
        counts = np.bincount(labels[ys, xs].ravel(), minlength=10).astype(np.float32)
        hist.append(counts / max(1, counts.sum()))
    # Nine coarse face-region luminance values retain interpretable appearance cues.
    means = np.asarray([g[y*64//3:(y+1)*64//3, x*64//3:(x+1)*64//3].mean()/255.
                        for y in range(3) for x in range(3)], dtype=np.float32)
    return np.concatenate((*hist, means)).astype(np.float32)  # 4*10 + 9 = 49


def read_video_frames(video: Path, fps_sample: float, face_landmarker):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise RuntimeError("OpenCV could not open video")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = nframes / source_fps if source_fps > 0 and nframes > 0 else 0.0
    if duration <= 0: raise RuntimeError("invalid fps/frame count")
    times = np.arange(0, duration, 1 / fps_sample, dtype=np.float32)
    crops, handcrafted, blendshapes, used_face, valid_blendshape, bboxes = [], [], [], [], [], []
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    blendshape_names: list[str] | None = None
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, frame = cap.read()
        if not ok:
            crops.append(np.zeros((224,224,3), np.uint8)); handcrafted.append(np.zeros(49,np.float32))
            blendshapes.append(np.zeros(BLENDSHAPE_DIM,np.float32))
            used_face.append(False); valid_blendshape.append(False); bboxes.append([0,0,0,0]); continue
        h, w = frame.shape[:2]
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result=face_landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
        if result.face_landmarks:
            landmarks=result.face_landmarks[0]
            xs=np.asarray([p.x for p in landmarks],np.float32)*w
            ys=np.asarray([p.y for p in landmarks],np.float32)*h
            x0,y0,x1,y1=float(xs.min()),float(ys.min()),float(xs.max()),float(ys.max())
            pad_x,pad_y=.18*(x1-x0),.18*(y1-y0)
            x0,y0=max(0,int(x0-pad_x)),max(0,int(y0-pad_y))
            x1,y1=min(w,int(x1+pad_x)),min(h,int(y1+pad_y))
            if x1<=x0 or y1<=y0:
                crop=frame; face=False; blend_ok=False; bbox=[0,0,0,0]; bs=np.zeros(BLENDSHAPE_DIM,np.float32)
            else:
                crop=frame[y0:y1,x0:x1]; face=True; blend_ok=True; bbox=[x0,y0,x1-x0,y1-y0]
                categories=result.face_blendshapes[0] if result.face_blendshapes else []
                bs=np.asarray([c.score for c in categories],np.float32)
                if bs.shape!=(BLENDSHAPE_DIM,):
                    raise RuntimeError(f"MediaPipe returned {len(bs)} blendshapes; expected {BLENDSHAPE_DIM}")
                names=[c.category_name for c in categories]
                if blendshape_names is None: blendshape_names=names
                elif names!=blendshape_names: raise RuntimeError("MediaPipe blendshape order changed within a clip")
        else:
            # Haar is a conservative crop fallback only; blendshapes remain invalid.
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            detections=haar.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(48,48)) if not haar.empty() else ()
            if len(detections):
                x0,y0,bw,bh=max(detections,key=lambda b:int(b[2])*int(b[3]))
                px,py=int(.18*bw),int(.18*bh); x0,y0=max(0,int(x0-px)),max(0,int(y0-py))
                x1,y1=min(w,int(x0+bw+2*px)),min(h,int(y0+bh+2*py))
                crop=frame[y0:y1,x0:x1]; bbox=[x0,y0,x1-x0,y1-y0]; face=True
            else:
                side=min(h,w); x0=max(0,(w-side)//2); y0=max(0,(h-side)//4)
                crop=frame[y0:min(h,y0+side),x0:x0+side]
                bbox=[int(x0),int(y0),int(side),int(side)]; face=False
            blend_ok=False; bs=np.zeros(BLENDSHAPE_DIM,np.float32)
        gray_crop=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        crops.append(cv2.resize(crop,(224,224),interpolation=cv2.INTER_AREA))
        handcrafted.append(lbp_uniform_hist(gray_crop)); blendshapes.append(bs)
        used_face.append(face); valid_blendshape.append(blend_ok); bboxes.append(bbox)
    cap.release()
    return (duration,source_fps,times,crops,np.asarray(handcrafted,np.float32),
            np.asarray(blendshapes,np.float32),np.asarray(used_face,bool),np.asarray(valid_blendshape,bool),
            np.asarray(bboxes,np.int32),blendshape_names or [])


def pool_by_word(feat: np.ndarray, frame_times: np.ndarray, intervals: np.ndarray) -> tuple[np.ndarray, list[int]]:
    if len(intervals) == 0: return np.zeros((0, feat.shape[-1]),np.float32), []
    if len(feat) == 0: return np.zeros((len(intervals), 0),np.float32), [0]*len(intervals)
    out=np.zeros((len(intervals),feat.shape[1]),np.float32); counts=[]
    for i,(a,b) in enumerate(intervals):
        # Strict interval pooling avoids leaking neighboring-word frames.
        # Short intervals with no sampled frame use the nearest midpoint frame
        # and are recorded with a zero in counts for downstream QC.
        sel=(frame_times >= a) & (frame_times < b)
        ids=np.flatnonzero(sel)
        if not len(ids): ids=np.asarray([int(np.argmin(np.abs(frame_times-(a+b)/2)))]); counts.append(0)
        else: counts.append(len(ids))
        out[i]=feat[ids].mean(axis=0)
    return out,counts


def pool_audio_rep(feat: np.ndarray, frame_period: float, intervals: np.ndarray) -> tuple[np.ndarray,list[int]]:
    ts=np.arange(len(feat),dtype=np.float32)*frame_period
    return pool_by_word(feat,ts,intervals)


def word_table(text: str, times: np.ndarray, ctc_conf: float, method: str,
               txt_counts: list[int], aud_counts: list[int], vis_counts: list[int],
               frame_counts: list[int], face_present: list[bool], bboxes: np.ndarray,
               text_intervals: list[tuple[int,int]]) -> list[dict[str,Any]]:
    ms=list(WORD_RE.finditer(text)); out=[]
    for i,m in enumerate(ms):
        bbox=bboxes[i].tolist() if i<len(bboxes) else [0,0,0,0]
        ca,cb=text_intervals[i] if i<len(text_intervals) else (m.start(),m.end())
        out.append({"word_index":i,"word":m.group(0),"char_start":ca,"char_end":cb,
          "start_sec":round(float(times[i,0]),4),"end_sec":round(float(times[i,1]),4),
          "ctc_alignment":method,"ctc_log_confidence":round(ctc_conf,5) if np.isfinite(ctc_conf) else "",
          "bert_subtokens":txt_counts[i] if i<len(txt_counts) else 0,
          "audio_frames":aud_counts[i] if i<len(aud_counts) else 0,
          "video_frames":frame_counts[i] if i<len(frame_counts) else 0,
          "vision_appearance_frames":vis_counts[i] if i<len(vis_counts) else 0,
          "face_detected":bool(face_present[i]) if i<len(face_present) else False,
          "bbox_x":bbox[0],"bbox_y":bbox[1],"bbox_w":bbox[2],"bbox_h":bbox[3]})
    return out


def atomic_json(path: Path, data: Any):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str,Any]]):
    if not rows: return
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def extract_one(record: dict[str,Any], video: Path, out_dir: Path, enc: Encoders,
                face_landmarker, visual_fps: float) -> tuple[dict[str,Any],list[dict[str,Any]]]:
    start=time.time(); text=record["text"]; matches=list(WORD_RE.finditer(text)); words=[m.group(0) for m in matches]
    if not words: raise ValueError("transcript contains no word tokens")
    wave=decode_audio(video)
    (duration,source_fps,frame_times,crops,vision49,blendshape_frames,
     face_flags,blendshape_flags,bboxes,blendshape_names)=read_video_frames(video,visual_fps,face_landmarker)
    wav_duration=len(wave)/SAMPLE_RATE
    # Preserve original video time as master timeline; ffmpeg audio is aligned
    # from t=0. Duration mismatch is recorded for the audit trail.
    duration=min(duration,wav_duration)
    (wav2vec,word_times,rep_period,align_method,ctc_conf,ctc_word_conf,
     ctc_word_conf_valid,audio_active_bounds,audio_rms,audio_peak,audio_present)=enc.audio(wave,text,duration)
    if len(word_times)!=len(words): raise RuntimeError("CTC word alignment count mismatch")
    text_features,subtoken_counts=enc.text_words(text,words)
    audio74,audio_times=mel_mfcc_prosody(wave)
    if not audio_present: audio74.fill(0)
    audio_word,audio_counts=pool_by_word(audio74,audio_times,word_times)
    wav_word,wav_counts=pool_audio_rep(wav2vec,float(rep_period[0]),word_times)
    resnet=enc.vision(crops)
    vision_word,vision_counts=pool_by_word(vision49,frame_times,word_times)
    blendshape_word,blendshape_counts=pool_by_word(blendshape_frames,frame_times,word_times)
    resnet_word,_=pool_by_word(resnet,frame_times,word_times)
    face_float=face_flags.astype(np.float32).reshape(-1,1)
    face_word,frame_counts=pool_by_word(face_float,frame_times,word_times)
    face_present=face_word[:,0] > .5
    bs_word, _=pool_by_word(blendshape_flags.astype(np.float32).reshape(-1,1),frame_times,word_times)
    blendshape_present=bs_word[:,0] > .5
    # Do not expose a partial blendshape average when the per-word validity mask is false.
    blendshape_word[~blendshape_present]=0
    sid=f"{record['video_id']}__{record['clip_id']}"
    n_words=len(words)
    alignment_reliable=align_method=="wav2vec2_ctc_forced_alignment"
    out_dir.mkdir(parents=True,exist_ok=True)
    npz=out_dir/f"{sid}.npz"; tmp=npz.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f,
            sample_id=np.asarray(sid),video_id=np.asarray(record['video_id']),clip_id=np.asarray(record['clip_id']),
            transcript=np.asarray(text),label=np.asarray(record['label']),annotation=np.asarray(record['annotation']),
            audio_present=np.asarray(audio_present),audio_rms=np.asarray(audio_rms,np.float32),audio_peak=np.asarray(audio_peak,np.float32),
            audio_active_bounds=np.asarray(audio_active_bounds,np.float32),
            words=np.asarray(words,dtype="U"),word_times=word_times.astype(np.float32),
            ctc_char_log_confidence=ctc_word_conf,ctc_char_log_confidence_valid=ctc_word_conf_valid,
            sequence_position=np.arange(n_words,dtype=np.int32),valid_length=np.asarray(n_words,np.int32),
            padding_mask=np.zeros(n_words,dtype=bool),valid_mask=np.ones(n_words,dtype=bool),
            text_valid=np.ones(n_words,dtype=bool),audio_valid=np.full(n_words,audio_present,dtype=bool),
            vision_valid=np.ones(n_words,dtype=bool),face_blendshape_valid=blendshape_present,
            alignment_reliable=np.full(n_words,alignment_reliable,dtype=bool),
            vision_nearest_frame_fallback=np.asarray(vision_counts,dtype=np.int32)==0,
            text=text_features,audio=audio_word,vision=vision_word,
            audio_wav2vec=wav_word,vision_resnet18=resnet_word,face_blendshapes=blendshape_word,
            face_blendshape_names=np.asarray(blendshape_names,dtype="U"),
            face_blendshape_frame_counts=np.asarray(blendshape_counts,dtype=np.int16),
            audio_frame_times=audio_times,video_frame_times=frame_times,
            face_detected=face_present,face_blendshape_detected=blendshape_present,face_bboxes=bboxes[:len(frame_times)])
    tmp.replace(npz)
    word_rows=word_table(text,word_times,ctc_conf,align_method,subtoken_counts,
       audio_counts,vision_counts,frame_counts,face_present,
       np.tile(np.array([0,0,0,0],np.int32),(len(words),1)),
       [(m.start(),m.end()) for m in matches])
    # Add a per-word visualization evidence location: the sampled frame nearest
    # the aligned word midpoint. Frame-level crop boxes are stored separately.
    for i,row in enumerate(word_rows):
        row["ctc_char_log_confidence"]=round(float(ctc_word_conf[i]),5) if ctc_word_conf_valid[i] else ""
        row["face_blendshape_frames"]=blendshape_counts[i]
        row["face_blendshape_valid"]=bool(blendshape_present[i])
        row["vision_nearest_frame_fallback"]=bool(vision_counts[i]==0)
        row["alignment_reliable"]=alignment_reliable
        mid=(word_times[i,0]+word_times[i,1])/2
        fi=int(np.argmin(np.abs(frame_times-mid))) if len(frame_times) else -1
        row["nearest_frame_sec"]=round(float(frame_times[fi]),4) if fi>=0 else ""
        if fi>=0:
            bb=bboxes[fi].tolist(); row.update({"bbox_x":bb[0],"bbox_y":bb[1],"bbox_w":bb[2],"bbox_h":bb[3],"face_detected":bool(face_flags[fi])})
    per_word_path=out_dir.parent/"word_alignment"/f"{sid}.csv"
    per_word_path.parent.mkdir(parents=True,exist_ok=True); write_csv(per_word_path,word_rows)
    row={"sample_id":sid,"video_id":record["video_id"],"clip_id":record["clip_id"],
      "video_file":video.relative_to(video.parents[1]).as_posix(),"duration_video_sec":round(duration,4),"duration_audio_sec":round(wav_duration,4),
      "source_fps":round(source_fps,4),"visual_sample_fps":visual_fps,"word_count":len(words),
      "text_dim":text_features.shape[1],"audio_dim":audio_word.shape[1],"vision_dim":vision_word.shape[1],
      "audio_wav2vec_dim":wav_word.shape[1],"vision_resnet18_dim":resnet_word.shape[1],
      "face_blendshape_dim":blendshape_word.shape[1],"valid_length":n_words,"padding":"none",
      "ctc_alignment":align_method,"ctc_log_confidence":round(ctc_conf,5) if np.isfinite(ctc_conf) else "",
      "audio_present":audio_present,"audio_rms":round(audio_rms,8),"audio_peak":round(audio_peak,8),
      "audio_active_start_sec":round(audio_active_bounds[0],4),"audio_active_end_sec":round(audio_active_bounds[1],4),
      "face_detection_rate":round(float(face_present.mean()),4) if len(face_present) else 0,
      "face_blendshape_detection_rate":round(float(blendshape_present.mean()),4) if len(blendshape_present) else 0,
      "haar_crop_fallback_rate":round(float(np.mean(face_present & ~blendshape_present)),4) if len(face_present) else 0,
      "vision_nearest_frame_fallback_rate":round(float(np.mean(np.asarray(vision_counts)==0)),4) if vision_counts else 0,
      "label":record["label"],"annotation":record["annotation"],
      "feature_file":npz.relative_to(out_dir.parent).as_posix(),
      "word_alignment_file":per_word_path.relative_to(out_dir.parent).as_posix(),
      "status":"ok","elapsed_sec":round(time.time()-start,2)}
    return row,word_rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root",type=Path,default=Path("E题数据/E题数据"),help="competition E题数据 root")
    ap.add_argument("--out",type=Path,default=Path("outputs/question1"))
    ap.add_argument("--cache-dir",type=Path,default=None)
    ap.add_argument("--face-landmarker-model",type=Path,default=None,help="MediaPipe .task file; downloaded automatically if omitted")
    ap.add_argument("--visual-fps",type=float,default=10.0)
    ap.add_argument("--limit",type=int,default=0,help="smoke-run only the first N records; 0=all")
    ap.add_argument("--sample-id",action="append",default=[],help="process only a selected video_id__clip_id; may be repeated")
    ap.add_argument("--resume",action="store_true",help="skip feature files already present")
    ap.add_argument("--offline",action="store_true",help="use only locally cached pretrained weights")
    ap.add_argument("--seed",type=int,default=2026)
    args=ap.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.offline: os.environ["HF_HUB_OFFLINE"]="1"
    if args.visual_fps<=0 or args.visual_fps>10: raise ValueError("visual-fps must be in (0,10]")
    data_root=args.data_root.expanduser().resolve(); out=args.out.expanduser().resolve()
    xlsx,video_root=find_inputs(data_root); records=load_labels(xlsx)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER.info("device=%s",device)
    if device.type=="cuda": LOGGER.info("GPU=%s",torch.cuda.get_device_name(device))
    out.mkdir(parents=True,exist_ok=True)
    log_file=out/"run.log"
    fh=logging.FileHandler(log_file,encoding="utf-8"); fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s")); LOGGER.addHandler(fh)
    face_model_path=ensure_face_landmarker(args.face_landmarker_model.expanduser() if args.face_landmarker_model else Path.home()/".cache/mosei-q1/face_landmarker.task")
    atomic_json(out/"model_and_environment.json",{
      "text_model":TEXT_MODEL,"speech_model":SPEECH_MODEL,
      "visual_model":f"MediaPipe Face Landmarker {mp.__version__} blendshapes (52-D) + torchvision ResNet18 ImageNet-1K (512-D)",
      "face_landmarker_model":"MediaPipe Face Landmarker float16 task model",
      "face_landmarker_model_url":FACE_LANDMARKER_URL,
      "face_landmarker_model_sha256":sha256_file(face_model_path),
      "text_feature":"mean of BERT WordPiece hidden states overlapping each transcript word; 768-D",
      "audio_feature":"mean of per-word 74-D descriptors: 40 log-mel + 13 MFCC + 13 MFCC delta + RMS/ZCR/F0/voicing/centroid + three prosody deltas",
      "audio_context_feature":"mean wav2vec2 final-layer hidden state over forced-aligned word frames; 768-D",
      "vision_feature":"per-word mean of 52 MediaPipe blendshape coefficients plus 49-D LBP/luminance and 512-D ImageNet ResNet18 crop embeddings",
      "vision_temporal_pooling":"strict word interval; nearest midpoint frame only when no sampled frame lands inside the interval",
      "vision_context_feature":"mean ImageNet ResNet-18 penultimate-layer activation on face crop; 512-D",
      "alignment":"wav2vec2 CTC Viterbi forced alignment to provided English transcript; proportional fallback recorded if alignment cannot be built",
      "ctc_fallback_rule":f"use proportional word-length intervals over energy-bounded speech region when mean CTC character log probability < {CTC_MIN_MEAN_LOGPROB}; silent audio is marked missing and uses proportional intervals over clip",
      "visual_sampling_fps":args.visual_fps,"face_crop_fallback":"OpenCV Haar face box used only when MediaPipe landmarks are absent; blendshape validity remains false",
      "sample_rate_hz":SAMPLE_RATE,
      "torch":torch.__version__,"torchaudio":torchaudio.__version__,"transformers":__import__('transformers').__version__,
      "opencv":cv2.__version__,"mediapipe":mp.__version__,"python":sys.version,"device":str(device),"data_root_label":args.data_root.name,
      "label_file_name":xlsx.name,"label_sha256":sha256_file(xlsx),"resume":args.resume,"seed":args.seed})
    # HF mirror is optional; use standard endpoint env variable when present.
    enc=Encoders(device,args.cache_dir)
    face_model=mp.tasks.vision.FaceLandmarker.create_from_options(
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(face_model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,output_face_blendshapes=True,
            min_face_detection_confidence=0.5,min_face_presence_confidence=0.5))
    if args.sample_id:
        wanted=set(args.sample_id); run_records=[r for r in records if f"{r['video_id']}__{r['clip_id']}" in wanted]
        missing=wanted-{f"{r['video_id']}__{r['clip_id']}" for r in run_records}
        if missing: raise ValueError(f"unknown sample-id(s): {sorted(missing)}")
    else:
        run_records=records[:args.limit] if args.limit>0 else records
    manifest=[]; failures=[]; total_words=0; total_start=time.time()
    for i,r in enumerate(run_records,1):
        sid=f"{r['video_id']}__{r['clip_id']}"; expected=out/"features"/f"{sid}.npz"
        if args.resume and expected.exists():
            try:
                z=np.load(expected,allow_pickle=False); n=len(z["words"]); z.close()
                manifest.append({"sample_id":sid,"video_id":r["video_id"],"clip_id":r["clip_id"],"feature_file":expected.relative_to(out).as_posix(),"word_count":n,"status":"skipped_existing"})
                total_words+=n; continue
            except Exception as e: LOGGER.warning("resume file %s invalid; recomputing: %s",expected,e)
        video=resolve_video(video_root,r["video_id"],r["clip_id"])
        if video is None:
            failures.append({"sample_id":sid,"video_id":r["video_id"],"clip_id":r["clip_id"],"error":"raw video file not found"})
            continue
        try:
            row,_=extract_one(r,video,out/"features",enc,face_model,args.visual_fps)
            manifest.append(row); total_words+=row["word_count"]
            LOGGER.info("[%d/%d] %s words=%d alignment=%s elapsed=%.1fs",i,len(run_records),sid,row["word_count"],row["ctc_alignment"],row["elapsed_sec"])
        except Exception as e:
            LOGGER.exception("sample failed: %s",sid)
            failures.append({"sample_id":sid,"video_id":r["video_id"],"clip_id":r["clip_id"],"video_file":video.relative_to(video.parents[1]).as_posix(),"error":repr(e)})
    face_model.close()
    write_csv(out/"sample_manifest.csv",manifest)
    write_csv(out/"failures.csv",failures)
    summary={"expected_samples":len(records),"attempted_samples":len(run_records),
      "successful_or_existing":len(manifest),"failed_or_missing":len(failures),"total_word_rows":total_words,
      "all_100_covered":len(records)==100 and len(manifest)==100 and not failures,
      "elapsed_seconds":round(time.time()-total_start,2),"output_dir":args.out.name,
      "models":{"text":TEXT_MODEL,"audio":SPEECH_MODEL,"vision":"MediaPipe Face Landmarker + torchvision ResNet18_Weights.DEFAULT"},
      "failures":failures}
    atomic_json(out/"summary.json",summary)
    LOGGER.info("finished: %s",json.dumps(summary,ensure_ascii=False))
    if failures: sys.exit(2)


if __name__=="__main__":
    main()
