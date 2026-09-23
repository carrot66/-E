"""Use decoded presentation timestamps, including stream start offsets."""
from pathlib import Path
import numpy as np


def probe(path):
    import av
    with av.open(str(path)) as c:
        streams = []
        for s in c.streams:
            if s.type not in ('video', 'audio'):
                continue
            streams.append(dict(type=s.type, index=s.index,
                                start=None if s.start_time is None else float(s.start_time * s.time_base),
                                duration=None if s.duration is None else float(s.duration * s.time_base)))
        stream_starts = [s['start'] for s in streams if s['start'] is not None]
        origin = min(([float(c.start_time / av.time_base)] if c.start_time is not None else [])
                     + stream_starts, default=0.)
        container_duration = (float(c.duration / av.time_base)
                              if c.duration is not None else 0.)
        stream_duration = max(
            ((s['start'] if s['start'] is not None else origin) - origin + (s['duration'] or 0)
             for s in streams), default=0.)
        duration = max(container_duration, stream_duration)
    if duration <= 0 or not np.isfinite(duration):
        raise ValueError(f'Invalid media duration: {path}')
    return dict(origin=origin, duration=duration, streams=streams)


def audio_decode(path, info, destination, sr=16000):
    import av
    import soundfile as sf
    n = int(np.ceil(info['duration'] * sr))
    signal, observed = np.zeros(n, np.float32), np.zeros(n, bool)
    with av.open(str(path)) as c:
        if c.streams.audio:
            resampler = av.AudioResampler(format='fltp', layout='mono', rate=sr)
            def put(frame):
                if frame.pts is None:
                    raise ValueError('Audio frame lacks PTS; cannot audit synchronization')
                start = int(round((float(frame.pts * frame.time_base) - info['origin']) * sr))
                x = frame.to_ndarray().reshape(-1)
                lo, hi = max(0, start), min(n, start + len(x))
                if hi > lo:
                    signal[lo:hi] = x[lo-start:hi-start]
                    observed[lo:hi] = True
            for frame in c.decode(audio=0):
                for out in resampler.resample(frame):
                    put(out)
            for out in resampler.resample(None):
                put(out)
    sf.write(str(destination), signal, sr, subtype='FLOAT')
    return signal, observed


def sampled_frames(path, info, fps):
    import av
    target = 0.
    with av.open(str(path)) as c:
        if not c.streams.video:
            return
        for index, frame in enumerate(c.decode(video=0)):
            if frame.pts is None:
                raise ValueError('Video frame lacks PTS; cannot audit synchronization')
            t = float(frame.pts * frame.time_base) - info['origin']
            if t < 0 or t >= info['duration'] or t + 1e-8 < target:
                continue
            yield index, int(frame.pts), t, frame.to_ndarray(format='rgb24')
            target = (np.floor(t * fps + 1e-8) + 1) / fps
