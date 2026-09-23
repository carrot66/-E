"""Generate a portable HTML evidence report from real extraction results."""
import argparse
import html
import json
from pathlib import Path
from urllib.parse import quote
import numpy as np


def make_report(out):
    h = html.escape
    rows = [json.loads(line) for line in (out/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    sections = []
    for row in rows:
        if row['status'] != 'complete':
            continue
        meta = json.loads((out/row['metadata']).read_text(encoding='utf-8'))
        with np.load(out/row['features'], allow_pickle=False) as f:
            score = sum(f[m+'_mask'].astype(int) for m in ('text','audio','vision'))
            index = int(np.argmax(score))
            a, b = f['time'][index]
            timeline = []
            for k, (start,end) in enumerate(f['time']):
                words = ' '.join(meta['words'][j]['word'] for j in meta['timeline'][k]['text_source_indices'])
                timeline.append('<tr><td>%d</td><td>%.3f–%.3f</td><td>%s</td><td>%s</td><td>%s</td></tr>' % (
                    k,start,end,h(words),
                    h(' / '.join(f'{m}:{int(f[m+"_mask"][k])}' for m in ('text','audio','vision'))),
                    h(' / '.join(f'{m}:c={f[m+"_coverage"][k]:.2f},q={f[m+"_quality"][k]:.2f}'
                                 for m in ('text','audio','vision')))))
            native_words = ''.join('<tr><td>%s</td><td>%.3f–%.3f</td><td>%.3f</td><td>%s</td></tr>' % (
                h(w['word']),w['start'],w['end'],w['confidence'],str(w['valid'])) for w in meta['words'])
        base = f"samples/{row['id']}/"
        evidence = meta['timeline'][index]
        images = []
        for j in evidence['vision_source_indices'][:3]:
            fr = meta['frames'][j]
            images.append(f'<figure><img src="{quote(base+fr["image"])}"><figcaption>'
                          f'frame={fr["frame_index"]}, t={fr["start"]:.3f}s; bbox={h(str(fr["bbox"]))}'
                          '</figcaption></figure>')
        # Include a sampled frame even if no detected face contributed to this window.
        if not images and meta['frames']:
            fr = min(meta['frames'], key=lambda r: abs(r['start']-(a+b)/2))
            images.append(f'<figure><img src="{quote(base+fr["image"])}"><figcaption>'
                          f'邻近帧 t={fr["start"]:.3f}s（可能未参与本窗口）</figcaption></figure>')
        sections.append(f'<section><h2>{h(row["id"])}</h2><p>{h(row["text"])}</p>'
                        f'<p>有效窗口数：{h(str(meta["valid_counts"]))}；总长度：{meta["length"]}</p>'
                        f'<p>代表窗口 {index}：{a:.3f}–{b:.3f}s（优先选择三模态均可用的首个窗口）</p>'
                        f'<audio controls preload="none" src="{quote(base+"audio.wav")}#t={a:.3f},{b:.3f}"></audio>'
                        '<div class="frames">'+''.join(images)+'</div>'
                        '<details><summary>全部窗口、模态掩码、覆盖率和质量</summary><table><tr><th>位置</th><th>秒</th><th>文本</th><th>mask</th><th>coverage / quality</th></tr>'
                        +''.join(timeline)+'</table></details>'
                        '<details><summary>逐词时间与CTC置信度</summary><table><tr><th>词</th><th>秒</th><th>置信度</th><th>有效</th></tr>'
                        +native_words+'</table></details></section>')
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>问题1对齐核验</title>'
            '<style>body{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#172435}'
            'section{border-top:1px solid #ccd5df;margin-top:28px}table{border-collapse:collapse;width:100%}'
            'td,th{border:1px solid #ddd;padding:7px;text-align:left}.frames{display:flex;flex-wrap:wrap}'
            'figure{margin:12px}img{width:280px;max-width:100%}details{margin:15px 0}</style>'
            '<h1>问题1：原始素材与时序特征对应核验</h1><p>窗口mask表示存在有效观测，不表示预测正确。'
            '音频片段播放依赖浏览器对媒体时间片段的支持；可按时间轴手动定位。框坐标记录在图下及metadata中。</p>'
            +''.join(sections)+'</html>')
    target = out/'alignment_report.html'
    target.write_text(page, encoding='utf-8')
    print(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'outputs')
    make_report(parser.parse_args().output)
