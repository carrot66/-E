#!/usr/bin/env python3
"""Sanitize paths and audit Question 1 feature outputs for completeness."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import q1_feature_extract as q1

def read_csv(p):
    if not p.exists() or p.stat().st_size == 0: return []
    with p.open(newline="",encoding="utf-8-sig") as f: return list(csv.DictReader(f))

def write_csv(p, rows, fields):
    with p.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",type=Path,default=Path("E题数据/E题数据"))
    ap.add_argument("--out",type=Path,default=Path("outputs/question1_final"))
    args=ap.parse_args(); out=args.out.resolve(); root=args.data_root.resolve()
    label_path,video_root=q1.find_inputs(root); labels=q1.load_labels(label_path)
    # Remove account and server-specific absolute paths from portable files.
    manifest_path=out/"sample_manifest.csv"; manifest=read_csv(manifest_path)
    for row in manifest:
        for key in ("feature_file","word_alignment_file"):
            raw=row.get(key,"")
            if raw:
                p=Path(raw)
                try: row[key]=p.resolve().relative_to(out).as_posix() if p.is_absolute() else p.as_posix()
                except ValueError: row[key]=Path(raw).name
        raw=row.get("video_file","")
        if raw:
            marker="MOSEI数据集部分原始视频-100条/"
            if marker in raw: row["video_file"]=raw.split(marker,1)[1].replace("\\","/")
            elif Path(raw).is_absolute():
                p=Path(raw)
                row["video_file"]=p.relative_to(video_root).as_posix() if p.is_relative_to(video_root) else p.name
    if manifest: write_csv(manifest_path,manifest,list(manifest[0]))
    failure_path=out/"failures.csv"; failures=read_csv(failure_path)
    for row in failures:
        raw=row.get("video_file","")
        if raw and Path(raw).is_absolute():
            p=Path(raw)
            row["video_file"]=p.relative_to(video_root).as_posix() if p.is_relative_to(video_root) else p.name
    fields=list(failures[0]) if failures else ["sample_id","video_id","clip_id","video_file","error"]
    write_csv(failure_path,failures,fields)
    env_path=out/"model_and_environment.json"
    env=json.loads(env_path.read_text(encoding="utf-8"))
    env.pop("data_root",None); env.pop("label_file",None)
    env["data_root_label"]=args.data_root.as_posix(); env["label_file_name"]=label_path.name
    env_path.write_text(json.dumps(env,ensure_ascii=False,indent=2),encoding="utf-8")
    summary_path=out/"summary.json"; summary=json.loads(summary_path.read_text(encoding="utf-8"))
    summary["output_dir"]=args.out.as_posix(); summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    # Scrub previous-run absolute home/project locations from the progress log.
    log_path=out/"run.log"
    if log_path.exists():
        project=Path(__file__).resolve().parent
        raw=log_path.read_text(encoding="utf-8",errors="replace")
        raw=raw.replace(str(project),".").replace(str(out),args.out.as_posix())
        log_path.write_text(raw,encoding="utf-8")

    expected={f"{r['video_id']}__{r['clip_id']}" for r in labels}
    got={r.get("sample_id","") for r in manifest}
    errors=[]; total_words=0; ctc_fallback=0; durations=[]; face_rates=[]; confidences=[]
    if len(labels)!=100: errors.append(f"expected label count 100, got {len(labels)}")
    if got!=expected: errors.append(f"sample id mismatch: missing={sorted(expected-got)[:5]}, extra={sorted(got-expected)[:5]}")
    for row in manifest:
        sid=row["sample_id"]; p=out/row.get("feature_file","")
        if not p.is_file(): errors.append(f"missing feature file: {sid}"); continue
        d=np.load(p,allow_pickle=False)
        words=d["words"]; times=d["word_times"]; n=len(words); total_words+=n
        for key,dim in (("text",768),("audio",74),("vision",49),("audio_wav2vec",768),("vision_resnet18",512)):
            x=d[key]
            if x.shape!=(n,dim): errors.append(f"{sid}: {key} shape {x.shape}, expected {(n,dim)}")
            if not np.isfinite(x).all(): errors.append(f"{sid}: nonfinite {key}")
        if times.shape!=(n,2) or np.any(times[:,1]<=times[:,0]) or np.any(np.diff(times[:,0]) < -1e-5):
            errors.append(f"{sid}: invalid word time intervals")
        duration=float(row.get("duration_video_sec") or 0)
        if n and duration and float(times[:,1].max())>duration+0.03: errors.append(f"{sid}: timestamp exceeds video duration")
        method=str(row.get("ctc_alignment",""))
        if "fallback" in method: ctc_fallback+=1
        conf=row.get("ctc_log_confidence","")
        try: confidences.append(float(conf))
        except (ValueError,TypeError): pass
        if len(d["face_detected"]): face_rates.append(float(d["face_detected"].mean()))
        if n: durations.extend((times[:,1]-times[:,0]).tolist())
        d.close()
    # No generated CSV/JSON/log should contain the machine-specific project root.
    project=str(Path(__file__).resolve().parent)
    private_refs=[]
    for p in out.rglob("*"):
        if p.suffix.lower() in {".csv",".json",".log"} and p.is_file():
            if project in p.read_text(encoding="utf-8",errors="replace"): private_refs.append(p.name)
    if private_refs: errors.append(f"absolute project path remains in: {private_refs}")
    audit={"expected_samples":len(labels),"manifest_samples":len(manifest),"feature_files":len(list((out/"features").glob("*.npz"))),
      "total_word_rows":total_words,"ctc_fallback_samples":ctc_fallback,
      "ctc_fallback_rate":ctc_fallback/max(1,len(manifest)),
      "mean_face_detection_word_rate":float(np.mean(face_rates)) if face_rates else 0,
      "ctc_confidence_mean":float(np.mean(confidences)) if confidences else None,
      "word_duration_min_sec":float(np.min(durations)) if durations else None,
      "word_duration_median_sec":float(np.median(durations)) if durations else None,
      "word_duration_max_sec":float(np.max(durations)) if durations else None,
      "path_anonymization_passed":not private_refs,"errors":errors,"passed":not errors}
    (out/"quality_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(audit,ensure_ascii=False,indent=2))
    if errors: raise SystemExit(2)

if __name__=="__main__": main()
