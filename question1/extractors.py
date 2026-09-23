"""Frozen pretrained extractors. Ground-truth emotion labels never enter models."""
from pathlib import Path
import numpy as np
from alignment import ctc_viterbi, transcript_words
from media import sampled_frames

MODEL_IDS = {
    'aligner': 'facebook/wav2vec2-base-960h',
    'text': 'j-hartmann/emotion-english-distilroberta-base',
    'vision': 'trpakov/vit-face-expression',
}


class Extractors:
    def __init__(self, snapshots, device):
        import torch
        import cv2
        import opensmile
        from transformers import (AutoProcessor, AutoModelForCTC, AutoTokenizer,
                                  AutoModelForSequenceClassification,
                                  AutoImageProcessor, AutoModelForImageClassification)
        self.torch, self.device = torch, device
        self.ap = AutoProcessor.from_pretrained(snapshots['aligner'], local_files_only=True)
        self.am = AutoModelForCTC.from_pretrained(snapshots['aligner'], local_files_only=True).to(device).eval()
        self.tp = AutoTokenizer.from_pretrained(snapshots['text'], local_files_only=True, use_fast=True)
        self.tm = AutoModelForSequenceClassification.from_pretrained(snapshots['text'], local_files_only=True).to(device).eval()
        self.vp = AutoImageProcessor.from_pretrained(snapshots['vision'], local_files_only=True, use_fast=False)
        self.vm = AutoModelForImageClassification.from_pretrained(snapshots['vision'], local_files_only=True).to(device).eval()
        self.smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                     feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
        self.face_cascade_path = str(Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml')
        self.face = cv2.CascadeClassifier(self.face_cascade_path)
        if self.face.empty():
            raise RuntimeError('OpenCV face detector unavailable')
        self.text_dim = self.tm.config.hidden_size + self.tm.config.num_labels
        self.vision_dim = self.vm.config.hidden_size + self.vm.config.num_labels
        self.schema = {
            'text_hidden_size': self.tm.config.hidden_size,
            'text_emotions': self.tm.config.id2label,
            'vision_hidden_size': self.vm.config.hidden_size,
            'vision_emotions': self.vm.config.id2label,
            'text_feature_layout': ['contextual_hidden'] * self.tm.config.hidden_size
                                  + ['emotion_probability'] * self.tm.config.num_labels,
            'vision_feature_layout': ['face_cls_hidden'] * self.vm.config.hidden_size
                                    + ['emotion_probability'] * self.vm.config.num_labels,
            'vision_detector': {'type': 'opencv_haar_frontalface_default', 'scale_factor': 1.1,
                                'min_neighbors': 5, 'min_size': [32, 32],
                                'cascade_file': self.face_cascade_path},
            'audio_lld': list(self.smile.feature_names),
            'audio_aligned': [f'{stat}:{name}' for stat in ('mean', 'std') for name in self.smile.feature_names],
        }

    def text(self, signal, observed, raw_text, duration, min_confidence):
        torch = self.torch
        words, tokens, owners = transcript_words(raw_text, self.ap.tokenizer.get_vocab())
        values = np.zeros((len(words), self.text_dim), np.float32)
        if not words:
            return words, values, np.empty((0, 2)), np.empty(0, bool), [], 'empty_transcript'
        alignment_error = None
        if not observed.any():
            alignment_error = 'no_audio_observed'
        else:
            try:
                inputs = self.ap(signal, sampling_rate=16000, return_tensors='pt')
                with torch.inference_mode():
                    logits = self.am(inputs.input_values.to(self.device)).logits[0]
                    logp = logits.log_softmax(-1).cpu().numpy()
                spans = ctc_viterbi(logp, tokens, self.am.config.pad_token_id)
                # Wav2Vec2 conv output positions: receptive field centers on original samples.
                stride, receptive = 1, 1
                for kernel, step in zip(self.am.config.conv_kernel, self.am.config.conv_stride):
                    receptive += (kernel - 1) * stride
                    stride *= step
                frame_boundaries = np.clip((np.arange(len(logp) + 1) * stride
                                            + (receptive - stride) / 2) / 16000., 0, duration)
                owners = np.asarray(owners)
                for j, w in enumerate(words):
                    ix = np.flatnonzero(owners == j)
                    first, last = spans[ix[0]][0], spans[ix[-1]][1]
                    confidence = float(np.exp(np.mean([np.log(max(spans[k][2], 1e-30)) for k in ix])))
                    start, end = float(frame_boundaries[first]), float(frame_boundaries[last])
                    observed_fraction = float(observed[int(start*16000):max(int(end*16000),int(start*16000)+1)].mean())
                    valid = end > start and confidence >= min_confidence and observed_fraction >= .99
                    w.update(start=start, end=end, confidence=confidence, valid=bool(valid),
                             observed_fraction=observed_fraction, ctc_first_frame=first,
                             ctc_last_frame_exclusive=last)
            except Exception as exc:
                alignment_error = f'{type(exc).__name__}: {exc}'
        if alignment_error:
            for w in words:
                w.update(start=0., end=0., confidence=0., valid=False, reason=alignment_error)
        # Chunk by tokenizer offsets, never silently truncate a long transcript.
        encoded = self.tp(raw_text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = encoded['input_ids'], encoded['offset_mapping']
        sums, mass = np.zeros_like(values), np.zeros(len(words), np.float32)
        chunks = []
        for base in range(0, len(ids), 384):
            part = ids[base:base+384]
            # Fast RoBERTa cannot create a special mask via prepare_for_model.
            # Build the positional mask with sentinel IDs; only real IDs enter the model.
            data = self.tp.prepare_for_model(part, return_tensors='pt')
            special = torch.tensor([x != -1 for x in self.tp.build_inputs_with_special_tokens([-1]*len(part))])
            data = {k: v.unsqueeze(0).to(self.device) for k, v in data.items()}
            with torch.inference_mode():
                out = self.tm(**data, output_hidden_states=True)
                hidden = out.hidden_states[-1][0][~special.to(self.device)].cpu().numpy()
                prob = out.logits.softmax(-1)[0].cpu().numpy()
            chunks.append(dict(token_start=base, token_end=base+len(part),
                               char_start=offsets[base][0], char_end=offsets[base+len(part)-1][1],
                               emotion_probabilities=prob.tolist()))
            for local, (a, b) in enumerate(offsets[base:base+384]):
                feat = np.concatenate((hidden[local], prob))
                for j, w in enumerate(words):
                    weight = max(0, min(b, w['char_end']) - max(a, w['char_start']))
                    if weight:
                        sums[j] += weight * feat
                        mass[j] += weight
        valid = np.array([w['valid'] for w in words]) & (mass > 0)
        values = sums / np.maximum(mass[:, None], 1)
        for j, w in enumerate(words):
            w['valid'] = bool(valid[j])
        intervals = np.array([[w['start'], w['end']] for w in words])
        return words, values, intervals, valid, chunks, alignment_error

    def audio(self, signal, observed):
        if not observed.any():
            return np.empty((0, len(self.smile.feature_names)), np.float32), np.empty((0, 2)), np.empty(0, bool)
        data = self.smile.process_signal(signal, 16000)
        intervals = np.column_stack((data.index.get_level_values('start').total_seconds(),
                                      data.index.get_level_values('end').total_seconds()))
        intervals = np.clip(intervals, 0., len(signal)/16000.)
        values = data.to_numpy(dtype=np.float32)
        valid = np.isfinite(values).all(axis=1)
        for j, (a, b) in enumerate(intervals):
            mask = observed[max(0, int(a*16000)):min(len(observed), int(np.ceil(b*16000)))]
            valid[j] &= bool(len(mask) and mask.all())
        values[~valid] = 0
        return values, intervals, valid

    def vision(self, path, info, folder, fps):
        import cv2
        from PIL import Image
        folder.mkdir(exist_ok=True)
        records, values = [], []
        for index, pts, t, rgb in sampled_frames(path, info, fps):
            faces = self.face.detectMultiScale(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                                               scaleFactor=1.1, minNeighbors=5, minSize=(32, 32))
            record = dict(frame_index=index, pts=pts, start=t,
                          end=min(t+1/fps, info['duration']), face_count=len(faces),
                          valid=False, bbox=None, image=f'frames/{index:06d}.jpg')
            Image.fromarray(rgb).save(folder / f'{index:06d}.jpg', quality=85)
            value = np.zeros(self.vision_dim, np.float32)
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: int(f[2])*int(f[3]))
                record['bbox'] = [int(x), int(y), int(w), int(h)]
                inputs = self.vp(images=Image.fromarray(rgb[y:y+h, x:x+w]), return_tensors='pt')
                with self.torch.inference_mode():
                    out = self.vm(**{k:v.to(self.device) for k,v in inputs.items()}, output_hidden_states=True)
                    value = self.torch.cat((out.hidden_states[-1][0, 0], out.logits.softmax(-1)[0])).cpu().numpy()
                record['valid'] = True
            records.append(record)
            values.append(value)
        intervals = np.array([[r['start'], r['end']] for r in records]).reshape(-1, 2)
        # End at the next observation if sampling jitter makes adjacent supports overlap.
        if len(intervals) > 1:
            intervals[:-1, 1] = np.minimum(intervals[:-1, 1], intervals[1:, 0])
            for j, row in enumerate(records):
                row['end'] = float(intervals[j, 1])
        return (records, np.asarray(values, np.float32).reshape(-1, self.vision_dim),
                intervals, np.array([r['valid'] for r in records], bool))
