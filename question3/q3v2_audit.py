"""Independent file-based checks; does not equate unresolved AV timing to completion."""
import json
from pathlib import Path
import numpy as np
from q3v2_common import MODS, csv_read, dump, sha, scores
from q3v2_explain import exact_shapley


def audit_outputs(out):
    out = Path(out)
    manifest = json.loads((out / '模型参数/manifest.json').read_text(encoding='utf-8'))
    if manifest.get('components'):
        components = manifest['components']
        assert components and np.isclose(sum(c['weight'] for c in components), 1)
        for c in components:
            assert c['weight'] > 0 and sha(out / c['path']) == c['sha256']
    else:
        assert sha(out / '模型参数/best.pt') == manifest['sha256']
    checks = {'checkpoint_hash': True, 'smoke': manifest['smoke']}
    data_audit = json.loads((out / '配置与审计/数据审计.json').read_text(encoding='utf-8'))
    expected_valid = int(data_audit['valid'])
    if not manifest['smoke']: assert expected_valid == 728
    for directory, expected in [('附件4预测', 20), ('验证集解释', expected_valid)]:
        rows = csv_read(out / directory / '全量预测与解释.csv'); coalitions = csv_read(out / directory / '8子集输出.csv')
        assert len(rows) == expected and len({r['sample_id'] for r in rows}) == expected
        assert len(coalitions) == expected * 8
        for r in rows:
            pp = np.array([float(r[k]) for k in ('p_negative', 'p_neutral', 'p_positive')])
            assert np.isfinite(pp).all() and (pp >= 0).all() and np.isclose(pp.sum(), 1, atol=1e-6)
            assert -3 <= float(r['intensity']) <= 3
            cc = sorted([c for c in coalitions if c['sample_id'] == r['sample_id']], key=lambda c: int(c['coalition_bitmask']))
            assert [int(c['coalition_bitmask']) for c in cc] == list(range(8))
            values = np.array([[float(c['class_margin']), float(c['intensity'])] for c in cc])
            calculated = exact_shapley(values)
            saved = np.array([[float(r[m + '_shapley_class']), float(r[m + '_shapley_intensity'])] for m in MODS])
            assert np.allclose(calculated, saved, atol=1e-6)
            assert np.allclose(saved.sum(0), values[7]-values[0], atol=1e-5)
            shares = np.array([float(r[m + '_share']) for m in MODS])
            assert np.isclose(shares.sum(), 0 if r['main_modality'] == 'none' else 1, atol=1e-6)
        evidence = csv_read(out / directory / '关键证据.csv')
        lookup = {r['sample_id']: r for r in rows}
        for e in evidence:
            if e['modality'] == 'text' and e['text_mapping_status'] == 'verified':
                assert lookup[e['sample_id']]['raw_text'][int(e['char_start']):int(e['char_end_exclusive'])] == e['text']
            if e['time_mapping_status'] == 'unresolved': assert e['start_seconds'] == e['end_seconds'] == ''
        checks[directory] = {'count': expected, 'shapley_recomputed': True, 'text_spans_checked': True}
    valid = csv_read(out / '验证集评价/全量预测.csv')
    assert len(valid) == expected_valid
    recomputed = scores(np.array([int(r['true_class']) for r in valid]),
        np.array([float(r['true_intensity']) for r in valid]),
        np.array([[float(r[k]) for k in ('p_negative', 'p_neutral', 'p_positive')] for r in valid]),
        np.array([float(r['intensity']) for r in valid]))
    published = json.loads((out / '验证集评价/评价.json').read_text(encoding='utf-8'))
    for key in ('Accuracy', 'Macro_F1', 'Weighted_F1', 'MAE', 'Pearson'):
        if recomputed[key] is not None: assert np.isclose(recomputed[key], published[key], atol=1e-6), key
    explained = {r['sample_id']: r for r in csv_read(out / '验证集解释/全量预测与解释.csv')}
    for r in valid:
        e = explained[r['sample_id']]
        for k in ('intensity', 'p_negative', 'p_neutral', 'p_positive'):
            assert np.isclose(float(r[k]), float(e[k]), atol=1e-5), (r['sample_id'], k)
    assert len(list((out / '解释卡').glob('*.html'))) == 20
    checks.update({'inference_matches_validation': True, 'metrics_recomputed': True, 'cards': 20, 'numerical_passed': True,
                   'scope': 'numerical audit; precise AV timing and manual semantic error causes require separate review'})
    dump(out / 'quality_audit.json', checks)
    status_path = out / '交付状态.json'
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding='utf-8'))
    else:
        status = {
            'model_version': manifest.get('version'),
            'smoke': bool(manifest.get('smoke', False)),
            'scope': 'official validation and attachment-4 explanation audit',
        }
    status['numerical_audit'] = 'passed'; dump(status_path, status)
    print(json.dumps(checks, ensure_ascii=False, indent=2))
