"""只读审计完整版 CMU-MOSEI 与竞赛附件2/3的关系。

本脚本绝不把附件3的特征匹配结果与标签连接，也不提供其标签。
输出仅用于排除训练泄漏、核对格式，不用于选择模型。
完整 pkl 解压后约 4.7 GB，请在内存充足的机器上运行。
"""
from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import pickle
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from q2_group_cv_audit import make_group_folds, sha256_id_order


ROOT = Path(__file__).resolve().parents[1]


def get_video(sample_id: str) -> str:
    if "$_$" not in sample_id:
        raise ValueError(f"异常样本ID：{sample_id}")
    return sample_id.rsplit("$_$", 1)[0]


def load_original(path: Path) -> dict:
    with path.open("rb") as stream:
        data = pickle.load(stream)
    lookup = {}
    groups = {}
    for split in ("train", "valid", "test"):
        sample = data[split]
        ids = [str(value) for value in sample["id"]]
        groups[split] = {get_video(sid) for sid in ids}
        for sid, value, raw in zip(ids, sample["regression_labels"], sample["raw_text"]):
            if sid in lookup:
                raise ValueError(f"原数据重复样本：{sid}")
            lookup[sid] = (split, float(value) if split != "test" else None, str(raw).strip())
    del data
    gc.collect()
    return {"lookup": lookup, "video_groups": groups}


def load_annex3(directory: Path) -> list[tuple[str, dict]]:
    files = sorted(directory.glob("附件3_*.pkl"))
    if len(files) != 30:
        raise ValueError(f"附件3对齐版文件数不是30：{len(files)}")
    queries = []
    for path in files:
        with path.open("rb") as stream:
            container = pickle.load(stream)
        if set(container) != {"test"}:
            raise ValueError(f"{path.name} 外层键异常")
        sample = container["test"]
        if set(sample) != {"text_bert", "audio", "vision"}:
            raise ValueError(f"{path.name} 特征键异常")
        for key, trailing in (("text_bert", (3, 50)), ("audio", (50, 74)), ("vision", (50, 35))):
            if np.asarray(sample[key]).shape != (1, *trailing):
                raise ValueError(f"{path.name} {key} 维度异常")
        queries.append((path.stem, sample))
    return queries


def read_label_csv(archive: zipfile.ZipFile) -> tuple[list[dict], dict]:
    name = next(name for name in archive.namelist() if name.endswith("/label.csv"))
    with archive.open(name) as binary:
        rows = list(csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")))
    if len(rows) != len({row["video_id"] + "$_$" + row["clip_id"] for row in rows}):
        raise ValueError("完整版 label.csv 存在重复样本ID")
    expected = {"Negative": 0, "Neutral": 1, "Positive": 2}
    for row in rows:
        if row["mode"] not in {"train", "valid", "test"}:
            raise ValueError(f"未知划分：{row['mode']}")
        if row["mode"] == "test":
            # ZIP has labels for the source of Annex3. Discard immediately;
            # they must never enter analysis, tuning, or output statistics.
            row["label"] = None
            row["annotation"] = None
        else:
            value = float(row["label"])
            sentiment = int(np.sign(value)) + 1
            if not np.isfinite(value) or not -3 <= value <= 3 or expected.get(row["annotation"]) != sentiment:
                raise ValueError(f"完整版标签符号异常：{row['video_id']}$_${row['clip_id']}")
    label = {row["video_id"] + "$_$" + row["clip_id"]: row for row in rows}
    return rows, label


def inspect_csv(rows: list[dict], label: dict, original: dict) -> dict:
    old = original["lookup"]
    shared = set(old) & set(label)
    if len(shared) != len(old):
        raise ValueError(f"附件2中{len(old)-len(shared)}条未出现在完整版")
    mismatch = {"mode": 0, "regression_label": 0, "raw_text": 0}
    for sid in shared:
        split, value, raw = old[sid]
        row = label[sid]
        mismatch["mode"] += split != row["mode"]
        if split != "test":
            mismatch["regression_label"] += abs(value - float(row["label"])) > 1e-6
        mismatch["raw_text"] += raw != row["text"].strip()
    if any(mismatch.values()):
        raise ValueError(f"附件2与完整版存在不一致：{mismatch}")
    groups = {split: {row["video_id"] for row in rows if row["mode"] == split}
              for split in ("train", "valid", "test")}
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if groups[a] & groups[b]:
            raise ValueError(f"完整版 {a}/{b} 存在视频组交叉")
    all_old_videos = set.union(*original["video_groups"].values())
    added = [row for row in rows if row["video_id"] + "$_$" + row["clip_id"] not in old]
    new_train = [row for row in added if row["mode"] == "train"]
    return {
        "samples": len(rows),
        "videos": len(set.union(*groups.values())),
        "split_counts": dict(Counter(row["mode"] for row in rows)),
        "split_video_counts": {split: len(groups[split]) for split in groups},
        "train_class_counts": dict(Counter(row["annotation"] for row in rows if row["mode"] == "train")),
        "valid_class_counts": dict(Counter(row["annotation"] for row in rows if row["mode"] == "valid")),
        "test_label_policy": "完整版test的标签字段在读取CSV后立即置空；不做标签统计或核对",
        "original_samples_all_present": len(shared),
        "original_train_valid_mode_label_text_mismatch": mismatch,
        "additional_samples": len(added),
        "additional_by_split": dict(Counter(row["mode"] for row in added)),
        "additional_train_from_existing_train_videos": sum(row["video_id"] in original["video_groups"]["train"] for row in new_train),
        "additional_train_from_entirely_new_videos": sum(row["video_id"] not in all_old_videos for row in new_train),
        "additional_train_overlapping_original_valid_test_videos": sum(row["video_id"] in
            (original["video_groups"]["valid"] | original["video_groups"]["test"]) for row in new_train),
        "optional_label_T_A_V_nonempty": {
            key: sum(bool(row[key]) for row in rows) for key in ("label_T", "label_A", "label_V")
        },
    }


def match_annex3(full: dict, queries: list[tuple[str, dict]], competition_test_ids: set[str]) -> list[dict]:
    results = []
    for name, item in queries:
        query = np.asarray(item["text_bert"])[0]
        q_tokens = query[0]
        # Exclude only the intentionally unknown positions. CLS/SEP/PAD still
        # constrain sequence length; no sentiment label participates in matching.
        visible = q_tokens != 100
        query_audio = np.asarray(item["audio"])[0]
        query_vision = np.asarray(item["vision"])[0]
        audio_observed = np.any(query_audio != 0, axis=-1)
        vision_observed = np.any(query_vision != 0, axis=-1)
        token_candidates = 0
        exact_matches = []
        for split in ("train", "valid", "test"):
            sample = full[split]
            all_tokens = np.asarray(sample["text_bert"])[:, 0, :]
            candidate_indices = np.flatnonzero(np.all(all_tokens[:, visible] == q_tokens[visible], axis=1))
            token_candidates += len(candidate_indices)
            for i in candidate_indices:
                source_audio = np.asarray(sample["audio"])[i]
                source_vision = np.asarray(sample["vision"])[i]
                if (np.array_equal(source_audio[audio_observed], query_audio[audio_observed])
                        and np.array_equal(source_vision[vision_observed], query_vision[vision_observed])):
                    sid = str(sample["id"][i])
                    exact_matches.append((split, sid, get_video(sid)))
        results.append({
            "annex3_sample": name,
            "visible_token_candidates": int(token_candidates),
            "observed_AV_exact_matches": len(exact_matches),
            "matched_split": exact_matches[0][0] if len(exact_matches) == 1 else None,
            # Never serialize source IDs of valid/test matches: label.csv contains
            # their labels, and a saved join key could invite accidental leakage.
            "matched_video_id_for_training_exclusion": (
                exact_matches[0][2] if len(exact_matches) == 1 and exact_matches[0][0] == "train" else None
            ),
            "overlaps_competition_attachment2_test": bool(
                len(exact_matches) == 1 and exact_matches[0][1] in competition_test_ids
            ),
            "warning": "multiple_feature_matches" if len(exact_matches) > 1 else "no_exact_feature_match" if not exact_matches else "",
        })
    return results


def inspect_feature_quality(sample: dict) -> dict:
    """仅扫描输入特征，不接触测试划分标签。"""
    bert = np.asarray(sample["text_bert"])
    n = len(bert)
    if not np.isin(bert[:, 1, :], [0, 1]).all():
        raise ValueError("text_bert attention mask 含非0/1值")
    content = np.zeros((n, 50), dtype=bool)
    for i, ids in enumerate(bert[:, 0, :]):
        cls = np.flatnonzero(ids == 101)
        sep = np.flatnonzero(ids == 102)
        if len(cls) == 0 or len(sep) == 0 or cls[0] >= sep[-1]:
            raise ValueError(f"第{i}条缺少有效CLS/SEP")
        content[i, cls[0] + 1:sep[-1]] = True
    audio = np.asarray(sample["audio"])
    vision = np.asarray(sample["vision"])
    text = np.asarray(sample["text"])
    for key, array in (("text", text), ("audio", audio), ("vision", vision)):
        if not np.isfinite(array).all():
            raise ValueError(f"{key} 存在NaN/Inf")
    obs_a = content & np.any(audio != 0, axis=-1)
    obs_v = content & np.any(vision != 0, axis=-1)
    unk = content & (bert[:, 0, :] == 100)
    return {
        "content_positions": int(content.sum()),
        "unknown_id100_content_positions": int(unk.sum()),
        "unknown_id100_attended_positions": int((unk & (bert[:, 1, :] == 1)).sum()),
        "unknown_id100_with_audio_zero": int((unk & ~obs_a).sum()),
        "unknown_id100_with_vision_zero": int((unk & ~obs_v).sum()),
        "unknown_id100_with_both_av_zero": int((unk & ~obs_a & ~obs_v).sum()),
        "attention_zero_content_positions": int((content & (bert[:, 1, :] == 0)).sum()),
        "audio_zero_content_positions": int((content.sum() - obs_a.sum())),
        "vision_zero_content_positions": int((content.sum() - obs_v.sum())),
        "whole_audio_unavailable_samples": int((obs_a.sum(axis=1) == 0).sum()),
        "whole_vision_unavailable_samples": int((obs_v.sum(axis=1) == 0).sum()),
        "attention_mask_at_50_samples": int((bert[:, 1, :].sum(axis=1) == 50).sum()),
        "nonfinite_input_values": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-zip", type=Path, required=True)
    parser.add_argument("--competition-aligned", type=Path, required=True)
    parser.add_argument("--annex3", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "问题2_完整版MOSEI安全整合审计")
    args = parser.parse_args()
    original = load_original(args.competition_aligned.resolve(strict=True))
    queries = load_annex3(args.annex3.resolve(strict=True))
    print("已读取附件2标识及附件3无标签输入；即将加载完整版大文件", flush=True)
    with zipfile.ZipFile(args.full_zip.resolve(strict=True)) as archive:
        rows, label = read_label_csv(archive)
        csv_report = inspect_csv(rows, label, original)
        pkl_name = next(name for name in archive.namelist() if name.endswith("/aligned_50.pkl"))
        with archive.open(pkl_name) as binary:
            full = pickle.load(binary)
    print("完整版 pkl 已加载，开始核对维度与无标签特征重合", flush=True)
    if set(full) != {"train", "valid", "test"}:
        raise ValueError(f"完整版 pkl 划分键异常：{sorted(full)}")
    feature_report = {}
    feature_quality = {}
    for split in ("train", "valid", "test"):
        sample = full[split]
        n = len(sample["id"])
        if n != csv_report["split_counts"][split]:
            raise ValueError(f"完整版 {split} pkl 与 CSV 样本数不一致")
        feature_report[split] = {}
        for key, trailing in (("text_bert", (3, 50)), ("text", (50, 768)),
                              ("audio", (50, 74)), ("vision", (50, 35))):
            value = np.asarray(sample[key])
            feature_report[split][key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            if value.shape != (n, *trailing):
                raise ValueError(f"完整版 {split} {key} 维度异常：{value.shape}")
        ids = [str(sid) for sid in sample["id"]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"完整版 {split} 存在重复ID")
        if split != "test":
            y = np.asarray(sample["classification_labels"])
            r = np.asarray(sample["regression_labels"])
            if not np.isfinite(y).all() or not np.isfinite(r).all() or not np.array_equal(y, np.sign(r) + 1):
                raise ValueError(f"完整版 {split} 标签异常")
        else:
            r = [None] * n
        for sid, value, raw in zip(ids, r, sample["raw_text"]):
            row = label.get(sid)
            if (row is None or row["mode"] != split or str(raw).strip() != row["text"].strip()
                    or (split != "test" and abs(float(value) - float(row["label"])) > 1e-6)):
                raise ValueError(f"完整版 {split} pkl 与 CSV 不一致：{sid}")
        feature_quality[split] = inspect_feature_quality(sample)
    competition_test_ids = {sid for sid, (split, _value, _text) in original["lookup"].items()
                            if split == "test"}
    match = match_annex3(full, queries, competition_test_ids)
    train_videos = {get_video(str(sid)) for sid in full["train"]["id"]}
    annex_videos = {row["matched_video_id_for_training_exclusion"] for row in match
                    if row["matched_video_id_for_training_exclusion"]}
    excluded_train_videos = sorted(train_videos & annex_videos)
    remaining_train_indices = [i for i, sid in enumerate(full["train"]["id"])
                               if get_video(str(sid)) not in excluded_train_videos]
    safe_ids = [str(full["train"]["id"][i]) for i in remaining_train_indices]
    safe_y = np.asarray(full["train"]["classification_labels"])[remaining_train_indices].astype(np.int64)
    fold_of_row, fold_summary = make_group_folds(safe_ids, safe_y, folds=5, seed=20260923, starts=32)
    fold_video_sets = [set(get_video(safe_ids[i]) for i in np.flatnonzero(fold_of_row == fold))
                       for fold in range(5)]
    assert all(not (fold_video_sets[i] & fold_video_sets[j])
               for i in range(5) for j in range(i + 1, 5))
    report = {
        "policy": "附件3只作输入重合检查，绝不读取其匹配来源的标签。匹配到完整训练集的视频必须全组排除。",
        "zip_csv": csv_report,
        "full_feature_shapes": feature_report,
        "full_input_feature_quality": feature_quality,
        "annex3_feature_matches": match,
        "annex3_unique_matches": sum(row["observed_AV_exact_matches"] == 1 for row in match),
        "annex3_match_split_counts": dict(Counter(row["matched_split"] or "unmatched_or_ambiguous" for row in match)),
        "annex3_overlap_with_competition_test_samples": sum(row["overlaps_competition_attachment2_test"] for row in match),
        "competition_test_samples_after_excluding_annex3_overlap": (
            len(competition_test_ids) - sum(row["overlaps_competition_attachment2_test"] for row in match)
        ),
        "train_videos_excluded_due_to_annex3_overlap": excluded_train_videos,
        "safe_train_sample_count_after_annex3_exclusion": len(remaining_train_indices),
        "safe_train_video_count_after_annex3_exclusion": len(train_videos) - len(excluded_train_videos),
        "safe_train_grouped_5fold_summary": fold_summary,
        "safe_train_5fold_video_overlap_count": 0,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "问题2_完整版MOSEI与竞赛数据安全整合审计.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "问题2_完整版训练安全索引.json").write_text(
        json.dumps({"source_split": "train", "excluded_video_ids": excluded_train_videos,
                    "remaining_train_indices": remaining_train_indices}, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output / "问题2_完整版训练集视频分组五折清单.json").write_text(
        json.dumps({
            "source_split": "train", "safe_train_sample_id_order_sha256": sha256_id_order(safe_ids),
            "seed": 20260923, "folds": 5, "starts": 32,
            "rule": "对排除附件3重合视频后的完整版train，以video_id为组进行类别均衡五折划分；每折须独立拟合归一化",
            "fold_summary": fold_summary,
            "validation_indices_within_safe_train_by_fold": {
                str(fold): np.flatnonzero(fold_of_row == fold).tolist() for fold in range(5)
            },
            "validation_sample_ids_by_fold": {
                str(fold): [safe_ids[i] for i in np.flatnonzero(fold_of_row == fold)] for fold in range(5)
            },
        }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("zip_csv", "full_feature_shapes", "annex3_unique_matches",
                                              "full_input_feature_quality",
                                              "annex3_match_split_counts", "annex3_overlap_with_competition_test_samples",
                                              "competition_test_samples_after_excluding_annex3_overlap",
                                              "train_videos_excluded_due_to_annex3_overlap",
                                              "safe_train_sample_count_after_annex3_exclusion",
                                              "safe_train_grouped_5fold_summary")},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
