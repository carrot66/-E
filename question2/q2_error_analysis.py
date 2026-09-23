"""按选定模型的验证集逐样本输出，生成可追溯的错误分析。"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', default=str(ROOT / 'outputs/问题2_优化实验结果'))
    args = parser.parse_args()
    directory = Path(args.result_dir)
    manifest = json.loads((directory / '问题2_集成选模结论.json').read_text(encoding='utf-8'))
    selected = manifest['selected']
    with (directory / f'问题2_验证集_{selected}_全量预测.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 728:
        raise ValueError(f'验证集行数异常：{len(rows)}')
    result = {'model': selected, 'sample_count': len(rows)}
    for cls in ['Negative', 'Neutral', 'Positive']:
        part = [x for x in rows if x['真实极性'] == cls]
        result[cls] = {
            'n': len(part),
            'classification_error_rate': sum(x['分类是否错误'] == 'True' for x in part) / len(part),
            'MAE': statistics.mean(float(x['绝对误差']) for x in part),
        }
    for key, name in [('文本不可用比例', 'text'), ('语音不可用比例', 'audio'), ('视觉不可用比例', 'vision')]:
        groups = [('原始有效位置均可观测', [x for x in rows if float(x[key]) == 0]),
                  ('存在原始不可观测位置', [x for x in rows if float(x[key]) > 0])]
        result[name + '_native_availability'] = []
        for label, part in groups:
            if part:
                result[name + '_native_availability'].append({
                    'group': label, 'n': len(part),
                    'mean_missing': statistics.mean(float(x[key]) for x in part),
                    'classification_error_rate': sum(x['分类是否错误'] == 'True' for x in part) / len(part),
                    'MAE': statistics.mean(float(x['绝对误差']) for x in part),
                })
    largest = sorted(rows, key=lambda x: float(x['绝对误差']), reverse=True)[:10]
    result['largest_absolute_errors'] = [{
        'id': x['样本编号'], 'true': x['真实极性'], 'pred': x['预测极性'],
        'true_intensity': float(x['真实强度']), 'predicted_intensity': float(x['预测强度']),
        'absolute_error': float(x['绝对误差']), 'text': x['文本'][:180],
    } for x in largest]
    output = directory / '问题2_验证集错误归因.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
