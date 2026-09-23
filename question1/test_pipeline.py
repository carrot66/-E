"""Integration regression: missing modalities, variable lengths, hashes and report.

Uses synthetic native features, not pretrained model outputs.
Run: python -m unittest -v test_alignment test_pipeline
"""
import json
import unittest
import uuid
from pathlib import Path
import numpy as np
from alignment import overlap_pool, time_grid
from run import aggregate, sha256, write_json
from verify import verify
from make_report import make_report


class PipelineTests(unittest.TestCase):
    def test_pack_audit_and_detect_corrupt_padding(self):
        # Use an ordinary workspace directory, compatible with restricted Windows environments.
        out = Path(__file__).resolve().parent/'outputs_test'/uuid.uuid4().hex
        out.mkdir(parents=True)
        rows = []
        for i, duration in enumerate((.7, 1.4)):
            row = dict(row_index=i, id=f'test$_${i}', source_sha256=f'fake-source-{i}',
                       text='Good', label=float(i), annotation='Neutral' if i==0 else 'Positive')
            rows.append(row)
            folder = out/'samples'/row['id']
            folder.mkdir(parents=True)
            grid = time_grid(duration, .5)
            arrays = dict(time=grid, sequence_mask=np.ones(len(grid), bool), label=np.array(float(i)),
                          text_global=np.array([2., 4.], np.float32), text_global_mask=np.array(True, bool))
            timeline = [dict(position=k,start=float(a),end=float(b)) for k,(a,b) in enumerate(grid)]
            counts = {}
            for name in ('text','audio','vision'):
                intervals = np.array([[0.,duration]])
                native = np.array([[2.,4.]], np.float32)
                valid = np.array([not (i==0 and name=='vision')])
                x, mask, w = overlap_pool(intervals,native,grid,valid,std=name=='audio')
                from alignment import interval_coverage, overlap_scalar
                qsource = valid.astype(np.float32)
                quality, _, _ = overlap_scalar(intervals, qsource, grid, valid)
                arrays.update({name:x,name+'_mask':mask,name+'_native':native,
                               name+'_intervals':intervals,name+'_native_mask':valid,
                               name+'_native_quality':qsource,name+'_weights':w,
                               name+'_coverage':interval_coverage(intervals,grid,valid),
                               name+'_quality':quality})
                counts[name] = int(mask.sum())
                for k in range(len(grid)):
                    timeline[k][name+'_source_indices'] = np.flatnonzero(w[k]>0).tolist()
            np.savez_compressed(folder/'features.npz',**arrays)
            (folder/'audio.wav').write_bytes(b'synthetic placeholder; not an actual recording')
            (folder/'frame.jpg').write_bytes(b'synthetic image placeholder')
            write_json(folder/'metadata.json',dict(**row, fingerprint='test',length=len(grid),
                       text_alignment_error=None,
                       modality_sources={'text':{},'audio':{},'vision':{}},
                       media=dict(duration=duration),frames=[dict(image='frame.jpg',start=0.,end=duration,
                       frame_index=0, bbox=[0,0,10,10],valid=i!=0,face_count=int(i!=0))],words=[dict(word='Good',start=0.,end=duration,
                       confidence=.9,valid=True)],timeline=timeline,valid_counts=counts,
                       feature_sha256=sha256(folder/'features.npz'),audio_sha256=sha256(folder/'audio.wav')))
        write_json(out/'input_audit.json',dict(samples=rows))
        write_json(out/'run.json',dict(fingerprint='test',settings=dict(step=.5)))
        (out/'manifest.jsonl').write_text(''.join(json.dumps(dict(**r,status='complete',
                metadata=f"samples/{r['id']}/metadata.json",features=f"samples/{r['id']}/features.npz"))+'\n'
                for r in rows),encoding='utf-8')
        aggregate(out, rows)
        self.assertEqual(verify(out)['verified'],2)
        make_report(out)
        self.assertIn('test$_$0',(out/'alignment_report.html').read_text(encoding='utf-8'))
        with np.load(out/'aligned_features.npz') as f:
            data = {k:f[k].copy() for k in f.files}
        data['text'][0,-1] = 99  # last row is padding only for the short sample
        np.savez_compressed(out/'aligned_features.npz',**data)
        with self.assertRaises(AssertionError):
            verify(out)


if __name__ == '__main__':
    unittest.main()
