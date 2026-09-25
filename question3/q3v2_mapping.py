"""Create evidence-map review sheet; optional offline CTC word timing, never guess A/V row times."""
import argparse
import re
import shutil
import subprocess
from pathlib import Path
import numpy as np
from q3v2_common import ROOT, MODS, special_data, csv_write, dump, sha


def forced_path(logp, labels, blank):
    """CTC Viterbi state alignment with blanks and repeated-character constraints."""
    states = np.full(2 * len(labels) + 1, blank, dtype=np.int64); states[1::2] = labels
    tmax, count = len(logp), len(states)
    if tmax < len(labels): raise ValueError('Too few acoustic frames for transcript')
    score = np.full(count, -np.inf); score[:2] = logp[0, states[:2]]
    back = np.zeros((tmax, count), dtype=np.int8)
    skip = np.zeros(count, bool); skip[2:] = (states[2:] != blank) & (states[2:] != states[:-2])
    for t in range(1, tmax):
        previous = np.stack([score, np.r_[-np.inf, score[:-1]], np.r_[-np.inf, -np.inf, score[:-2]]])
        previous[2, ~skip] = -np.inf
        choice = previous.argmax(0); back[t] = choice
        score = previous[choice, np.arange(count)] + logp[t, states]
    end = count - 1 if score[-1] > score[-2] else count - 2
    if not np.isfinite(score[end]): raise ValueError('No feasible CTC alignment')
    path = np.zeros(tmax, np.int64); path[-1] = end
    for t in range(tmax-1, 0, -1): path[t-1] = path[t] - back[t, path[t]]
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--bert', type=Path, default=ROOT / 'work/pretrained_models/bert')
    p.add_argument('--out', type=Path, default=ROOT / 'outputs/question3/时间映射待核验')
    p.add_argument('--speech-model', type=Path, help='Optional offline wav2vec2-base-960h directory')
    p.add_argument('--device', default='cpu')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.bert, local_files_only=True, use_fast=True)
    data, _ = special_data(a.data_root, tokenizer)
    processor = speech = None
    if a.speech_model:
        import torch
        from transformers import AutoProcessor, AutoModelForCTC
        if not shutil.which('ffmpeg'): raise RuntimeError('ffmpeg is required for raw audio decoding')
        processor = AutoProcessor.from_pretrained(a.speech_model, local_files_only=True)
        speech = AutoModelForCTC.from_pretrained(a.speech_model, local_files_only=True).to(a.device).eval()
    mapping = []; words_out = []; failures = []
    for i, sid in enumerate(data['ids']):
        word_rows = []
        if speech is not None:
            try:
                raw_audio = subprocess.run(['ffmpeg', '-v', 'error', '-i', data['videos'][i], '-f', 'f32le', '-ac', '1', '-ar', '16000', 'pipe:1'], capture_output=True, check=True).stdout
                audio = np.frombuffer(raw_audio, dtype='<f4').copy()
                if len(audio) < 400 or float(np.std(audio)) < 1e-6: raise ValueError('Empty/silent audio')
                inp = processor(audio, sampling_rate=16000, return_tensors='pt')
                with torch.inference_mode():
                    logp = speech(**{k: v.to(a.device) for k, v in inp.items()}).logits.log_softmax(-1)[0].cpu().numpy()
                matches = list(re.finditer(r"[A-Za-z]+(?:['’][A-Za-z]+)?", data['raw'][i]))
                vocab = processor.tokenizer.get_vocab(); ids = []; ranges = []
                for match in matches:
                    if ids: ids.append(vocab['|'])
                    lo = len(ids); chars = match.group().replace('’', "'").upper()
                    if any(c not in vocab for c in chars): raise ValueError('Unsupported CTC character')
                    ids.extend(vocab[c] for c in chars); ranges.append((lo, len(ids)))
                if not ids: raise ValueError('No supported transcript words')
                path = forced_path(logp, ids, speech.config.pad_token_id)
                # Conv geometry: centers are based on receptive field / strides, not duration/word count.
                stride = 1; receptive = 1
                for kernel, step in zip(speech.config.conv_kernel, speech.config.conv_stride):
                    receptive += (kernel-1)*stride; stride *= step
                for match, (lo, hi) in zip(matches, ranges):
                    frames = np.flatnonzero((path >= 2*lo+1) & (path <= 2*(hi-1)+1))
                    if not len(frames): raise ValueError('Empty word alignment')
                    confidences = [float(logp[t, ids[(path[t]-1)//2]]) for t in frames if path[t] % 2]
                    confidence = float(np.mean(confidences)) if confidences else -100.
                    start = max(0., (frames[0]*stride + receptive/2 - stride/2) / 16000)
                    end = min(len(audio)/16000, (frames[-1]*stride + receptive/2 + stride/2) / 16000)
                    word_rows.append({'sample_id': sid, 'word': match.group(), 'char_start': match.start(), 'char_end': match.end(),
                                      'start_seconds': start, 'end_seconds': end, 'mean_log_probability': confidence,
                                      'status': 'approximate' if confidence >= -3 else 'unresolved',
                                      'note': 'automatic CTC; relative to decoded audio start; media A/V offset and boundaries need review'})
                words_out += word_rows
            except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
                failures.append({'sample_id': sid, 'error': str(exc)})
        for m, name in enumerate(MODS):
            for t in np.flatnonzero(data['mask'][i, :, m]):
                row = {'sample_id': sid, 'modality': name, 'feature_index': int(t), 'start_seconds': '', 'end_seconds': '',
                       'status': 'unresolved', 'source': '', 'mapping_rule': '', 'reviewer': ''}
                if m == 0 and data['text_mapping'][i]:
                    lo, hi = data['offsets'][i, t]
                    match = [w for w in word_rows if w['char_start'] < hi and w['char_end'] > lo and w['status'] == 'approximate']
                    if match:
                        row.update(start_seconds=min(w['start_seconds'] for w in match), end_seconds=max(w['end_seconds'] for w in match),
                                   status='approximate', source='automatic_ctc_audio_relative', mapping_rule='exact_token_offsets_to_ctc_word')
                mapping.append(row)
    csv_write(a.out / 'evidence_map_review.csv', mapping)
    csv_write(a.out / 'word_times.csv', words_out)
    dump(a.out / 'mapping_status.json', {'automatic_word_rows': len(words_out), 'failures': failures,
         'av_feature_row_times': 'unresolved; official row-to-word rule must be verified independently',
         'audio_video_offset': 'must be checked before promoting audio-relative CTC times to verified video times',
         'speech_config_sha256': sha(a.speech_model / 'config.json') if a.speech_model else None})
    print(a.out / 'evidence_map_review.csv')


if __name__ == '__main__': main()
