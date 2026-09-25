"""Extract media ONLY for externally verified video-time evidence intervals."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path
from q3v2_common import csv_read, csv_write, dump, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True, help='Existing Q3 run directory')
    a = p.parse_args()
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise RuntimeError('ffmpeg and ffprobe must be on PATH')
    evidence = csv_read(a.out / '附件4预测/关键证据.csv')
    videos = {r['sample_id']: r for r in csv_read(a.out / '附件4预测/视频映射.csv')}
    media = a.out / '原始证据片段'; media.mkdir(exist_ok=True)
    rows = []; skipped = 0; durations = {}
    for j, r in enumerate(evidence):
        if r['time_mapping_status'] != 'verified': skipped += 1; continue
        sid = r['sample_id']; video = (a.data_root / videos[sid]['video_relative']).resolve()
        if not video.is_relative_to(a.data_root.resolve()) or not video.is_file(): raise ValueError('Invalid video path')
        if sid not in durations:
            result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(video)], capture_output=True, text=True, check=True)
            durations[sid] = float(json.loads(result.stdout)['format']['duration'])
        start, end = float(r['start_seconds']), float(r['end_seconds'])
        if not 0 <= start < end <= durations[sid] + .02: raise ValueError(f'Out-of-range interval: {sid}')
        prefix = f'{sid}_{r["modality"]}_{j:04d}'
        # ffmpeg timestamps refer to the verified video timeline; no duration/row interpolation.
        for kind in ('video', 'frame'):
            output = media / (prefix + ('.mp4' if kind == 'video' else '.jpg'))
            command = ['ffmpeg', '-v', 'error', '-y', '-i', str(video), '-ss', str(start if kind == 'video' else (start+end)/2)]
            if kind == 'video': command += ['-t', str(end-start), '-c:v', 'libx264', '-crf', '28', '-c:a', 'aac', '-movflags', '+faststart']
            else: command += ['-frames:v', '1', '-q:v', '3']
            subprocess.run(command + [str(output)], capture_output=True, check=True)
            rows.append({'sample_id': sid, 'modality': r['modality'], 'kind': kind,
                         'start_seconds': start, 'end_seconds': end, 'source_video_sha256': sha(video),
                         'file': str(output.relative_to(a.out)), 'sha256': sha(output),
                         'frame_rule': 'midpoint of verified interval; not maximum-expression selection'})
    csv_write(media / 'manifest.csv', rows)
    dump(media / 'status.json', {'exported_files': len(rows), 'unverified_windows_skipped': skipped,
                               'note': 'verified input timing is required; exports do not prove the mapping rule'})
    print(f'Exported {len(rows)} files; skipped {skipped} unverified intervals.')


if __name__ == '__main__': main()
