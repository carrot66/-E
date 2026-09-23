"""Pure NumPy alignment kernels; no model downloads needed for tests."""
import re
import numpy as np


def time_grid(duration, step):
    if not np.isfinite(duration) or duration <= 0 or step <= 0:
        raise ValueError('duration and step must be positive')
    starts = np.arange(int(np.ceil(duration / step)), dtype=np.float64) * step
    return np.column_stack((starts, np.minimum(starts + step, duration)))


def overlap_pool(intervals, values, grid, valid=None, std=False):
    """Duration-weighted moments. Missing is zero with a separate boolean mask."""
    intervals = np.asarray(intervals, dtype=np.float64).reshape(-1, 2)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or len(values) != len(intervals):
        raise ValueError('interval/value shape mismatch')
    valid = np.ones(len(values), bool) if valid is None else np.asarray(valid, bool)
    valid = valid & np.isfinite(values).all(axis=1)
    w = np.maximum(0, np.minimum(grid[:, None, 1], intervals[None, :, 1])
                   - np.maximum(grid[:, None, 0], intervals[None, :, 0]))
    w[:, ~valid] = 0
    mass = w.sum(axis=1)
    norm = w / np.maximum(mass[:, None], 1e-15)
    clean = np.where(valid[:, None], values, 0)
    mean = norm @ clean
    if std:
        variance = np.maximum(norm @ (clean.astype(np.float64) ** 2) - mean ** 2, 0)
        mean = np.concatenate((mean, np.sqrt(variance)), axis=1)
    return mean.astype(np.float32), mass > 0, w


def interval_coverage(intervals, grid, valid=None):
    """Union coverage in each grid cell; unlike raw weights it never double-counts overlaps."""
    intervals = np.asarray(intervals, dtype=np.float64).reshape(-1, 2)
    grid = np.asarray(grid, dtype=np.float64).reshape(-1, 2)
    if len(intervals) and not np.isfinite(intervals).all():
        raise ValueError('intervals must be finite')
    if len(grid) and not np.isfinite(grid).all():
        raise ValueError('grid must be finite')
    valid = np.ones(len(intervals), bool) if valid is None else np.asarray(valid, bool)
    if len(valid) != len(intervals):
        raise ValueError('interval/valid shape mismatch')
    out = np.zeros(len(grid), np.float32)
    for k, (left, right) in enumerate(grid):
        width = right - left
        if width <= 0:
            continue
        pieces = []
        for (a, b), ok in zip(intervals, valid):
            if not ok:
                continue
            a, b = max(left, a), min(right, b)
            if b > a:
                pieces.append((a, b))
        if not pieces:
            continue
        pieces.sort()
        covered = 0.
        a, b = pieces[0]
        for c, d in pieces[1:]:
            if c > b:
                covered += b - a
                a, b = c, d
            else:
                b = max(b, d)
        out[k] = min(1., max(0., (covered + b - a) / width))
    return out


def overlap_scalar(intervals, scores, grid, valid=None):
    """Weighted scalar quality score using the same support rule as feature pooling."""
    scores = np.asarray(scores, dtype=np.float32).reshape(-1, 1)
    pooled, mask, weights = overlap_pool(intervals, scores, grid, valid, std=False)
    return pooled[:, 0], mask, weights


def ctc_viterbi(logp, tokens, blank):
    """Exact CTC Viterbi with mandatory blanks between identical adjacent tokens.

    Returns one (first_frame, exclusive_last_frame, geometric_probability)
    per target token. Keeps the complete transcript, or raises on no path.
    """
    logp = np.asarray(logp, np.float32)
    tokens = np.asarray(tokens, np.int64)
    if not len(tokens):
        return []
    states = np.full(2 * len(tokens) + 1, blank, np.int64)
    states[1::2] = tokens
    T, S = len(logp), len(states)
    if T < len(tokens) + np.sum(tokens[1:] == tokens[:-1]):
        raise ValueError('CTC: insufficient frames for full transcript')
    previous = np.full(S, -np.inf, np.float32)
    previous[0] = logp[0, blank]
    previous[1] = logp[0, tokens[0]]
    trace = np.zeros((T, S), np.uint8)
    skip = np.zeros(S, bool)
    skip[2:] = (states[2:] != blank) & (states[2:] != states[:-2])
    for t in range(1, T):
        p1 = np.r_[-np.inf, previous[:-1]]
        p2 = np.r_[-np.inf, -np.inf, previous[:-2]]
        p2[~skip] = -np.inf
        choices = np.stack((previous, p1, p2))
        trace[t] = choices.argmax(axis=0)
        previous = choices.max(axis=0) + logp[t, states]
    s = S - 1 if previous[-1] >= previous[-2] else S - 2
    if not np.isfinite(previous[s]):
        raise ValueError('CTC: no valid full-transcript path')
    path = np.zeros(T, np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        if t:
            s -= int(trace[t, s])
    spans = []
    for j, token in enumerate(tokens):
        frames = np.flatnonzero(path == 2 * j + 1)
        if not len(frames):
            raise ValueError('CTC omitted a target token')
        spans.append((int(frames[0]), int(frames[-1] + 1),
                      float(np.exp(logp[frames, token].mean()))))
    return spans


def transcript_words(text, vocab):
    """Keep original offsets; normalization only affects the CTC target."""
    from num2words import num2words
    rows, tokens, owners = [], [], []
    for match in re.finditer(r"[A-Za-z]+(?:['’][A-Za-z]+)*|\d+(?:\.\d+)?", text):
        raw = match.group()
        normalized = num2words(raw) if raw[0].isdigit() else raw
        normalized = normalized.upper().replace('’', "'")
        normalized = re.sub(r"[^A-Z']+", '|', normalized).strip('|')
        if any(ch not in vocab for ch in normalized):
            raise ValueError(f'Unsupported CTC characters: {raw}')
        if tokens:
            tokens.append(vocab['|'])
            owners.append(-1)
        owner = len(rows)
        rows.append(dict(word=raw, char_start=match.start(), char_end=match.end(),
                         normalized=normalized))
        for ch in normalized:
            tokens.append(vocab[ch])
            owners.append(owner)
    return rows, tokens, owners
