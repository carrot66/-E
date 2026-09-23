"""Recompute pooling and verify coverage, temporal masks, padding and provenance."""
import argparse
import json
from pathlib import Path
import numpy as np
from alignment import interval_coverage, overlap_pool, overlap_scalar, time_grid


def verify(out):
    from run import sha256, write_json
    # Never leave a previous successful report in place if a new audit fails.
    write_json(out/'verification.json', dict(passed=False, status='verification_started'))
    inventory = json.loads((out/'input_audit.json').read_text(encoding='utf-8'))
    config = json.loads((out/'run.json').read_text(encoding='utf-8'))
    rows = inventory['samples']
    summaries = []
    with np.load(out/'aligned_features.npz', allow_pickle=False) as all_data:
        assert all_data['ids'].tolist() == [r['id'] for r in rows], 'ID/order mismatch'
        for i, row in enumerate(rows):
            folder = out/'samples'/row['id']
            m = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
            assert m['fingerprint'] == config['fingerprint']
            assert m['source_sha256'] == row['source_sha256']
            assert m['text'] == row['text'] and m['label'] == row['label']
            assert m['annotation'] == row['annotation'] and m['id'] == row['id']
            assert m['row_index'] == row['row_index']
            assert set(m['modality_sources']) == {'text', 'audio', 'vision'}
            assert m['feature_sha256'] == sha256(folder/'features.npz')
            assert m['audio_sha256'] == sha256(folder/'audio.wav')
            assert all((folder/r['image']).is_file() for r in m['frames'])
            with np.load(folder/'features.npz', allow_pickle=False) as f:
                n = m['length']
                assert f['text_global'].shape == (f['text'].shape[1],)
                assert bool(f['text_global_mask']) == bool(len(f['text_native']))
                if len(f['text_native']):
                    np.testing.assert_allclose(f['text_global'], f['text_native'].mean(axis=0), atol=1e-6)
                else:
                    assert not bool(f['text_global_mask'])
                np.testing.assert_allclose(f['text_global'], all_data['text_global'][i], atol=1e-6)
                assert bool(f['text_global_mask']) == bool(all_data['text_global_mask'][i])
                assert n == len(f['time']) == all_data['lengths'][i]
                np.testing.assert_allclose(f['label'], row['label'])
                np.testing.assert_allclose(all_data['labels'][i], row['label'])
                assert f['sequence_mask'].all() and len(f['sequence_mask']) == n
                np.testing.assert_allclose(f['time'], time_grid(m['media']['duration'], config['settings']['step']))
                assert all_data['sequence_mask'][i, :n].all()
                assert not all_data['sequence_mask'][i, n:].any()
                assert (all_data['time'][i, n:] == -1).all()
                np.testing.assert_array_equal(f['time'], all_data['time'][i, :n])
                for name in ('text','audio','vision'):
                    intervals = f[name+'_intervals']
                    assert np.isfinite(intervals).all()
                    assert (intervals[:, 0] >= 0).all()
                    assert (intervals[:, 1] >= intervals[:, 0]).all()
                    assert (intervals[:, 1] <= m['media']['duration'] + 1e-5).all()
                    x, mask, weights = overlap_pool(f[name+'_intervals'], f[name+'_native'],
                                                     f['time'], f[name+'_native_mask'], std=name=='audio')
                    np.testing.assert_allclose(x, f[name], rtol=1e-5, atol=1e-5)
                    np.testing.assert_array_equal(mask, f[name+'_mask'])
                    np.testing.assert_allclose(weights, f[name+'_weights'], atol=1e-7)
                    np.testing.assert_allclose(
                        interval_coverage(f[name+'_intervals'], f['time'], f[name+'_native_mask']),
                        f[name+'_coverage'], atol=1e-6)
                    quality, _, _ = overlap_scalar(f[name+'_intervals'], f[name+'_native_quality'],
                                                    f['time'], f[name+'_native_mask'])
                    np.testing.assert_allclose(quality, f[name+'_quality'], atol=1e-6)
                    assert ((f[name+'_coverage'] > 0) == f[name+'_mask']).all()
                    assert ((f[name+'_quality'] >= 0) & (f[name+'_quality'] <= 1) &
                            np.isfinite(f[name+'_quality'])).all()
                    assert np.isfinite(f[name]).all()
                    assert (f[name][~mask] == 0).all()
                    np.testing.assert_array_equal(f[name], all_data[name][i, :n])
                    np.testing.assert_array_equal(mask, all_data[name+'_mask'][i, :n])
                    np.testing.assert_allclose(f[name+'_coverage'], all_data[name+'_coverage'][i, :n])
                    np.testing.assert_allclose(f[name+'_quality'], all_data[name+'_quality'][i, :n])
                    assert (all_data[name][i, n:] == 0).all()
                    assert not all_data[name+'_mask'][i, n:].any()
                    assert (all_data[name+'_coverage'][i, n:] == 0).all()
                    assert (all_data[name+'_quality'][i, n:] == 0).all()
                    for k in range(n):
                        assert meta_indices(m,k,name) == np.flatnonzero(weights[k]>0).tolist()
                    assert m['valid_counts'][name] == int(mask.sum())
            summaries.append(dict(id=row['id'], length=n, **m['valid_counts'],
                                  low_confidence_words=sum(not w['valid'] for w in m['words']),
                                  no_face_frames=sum(not r['valid'] for r in m['frames']),
                                  multiple_face_frames=sum(r['face_count']>1 for r in m['frames'])))
    report = dict(passed=True, expected=len(rows), verified=len(summaries), samples=summaries,
                  note='Structural validation does not establish recognition accuracy. Review low-confidence words and face crops.')
    write_json(out/'verification.json', report)
    print(f'PASS: {len(summaries)}/{len(rows)} samples; pooling, masks, padding and hashes verified.')
    return report


def meta_indices(metadata, position, modality):
    row = metadata['timeline'][position]
    assert row['position'] == position
    return row[modality+'_source_indices']


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'outputs')
    verify(p.parse_args().output)
