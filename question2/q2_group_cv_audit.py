"""附件2训练集的数据审计和视频分组交叉验证划分。

只读取 train 的标签；valid/test 仅读取 ID 来检查样本和视频是否跨划分。
附件3完全不参与。本脚本仅依赖 Python 标准库和 NumPy。

示例：
python q2_group_cv_audit.py --aligned "E题数据/附件2-数据集特征文件/aligned_50.pkl"

输出的 JSON/CSV 放在被 .gitignore 排除的 outputs/ 中，不包含原始特征。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "问题2_训练集分组交叉验证"


def video_id(sample_id: object) -> str:
    value = str(sample_id)
    if "$_$" not in value:
        raise ValueError(f"样本 ID 缺少视频分隔符：{value!r}")
    video, segment = value.rsplit("$_$", 1)
    if not video or not segment:
        raise ValueError(f"无法解析视频与片段编号：{value!r}")
    return video


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_id_order(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def inspect_train(train: dict) -> tuple[list[str], np.ndarray, dict]:
    ids = [str(value) for value in train["id"]]
    n = len(ids)
    if len(set(ids)) != n:
        raise ValueError("train 存在重复样本 ID")
    groups = [video_id(sid) for sid in ids]
    y_raw = np.asarray(train["classification_labels"])
    r = np.asarray(train["regression_labels"])
    if y_raw.shape != (n,) or r.shape != (n,):
        raise ValueError("训练标签长度与样本 ID 数量不一致")
    if not np.isfinite(y_raw).all() or not np.isfinite(r).all():
        raise ValueError("训练标签存在 NaN/Inf")
    if not np.equal(y_raw, y_raw.astype(np.int64)).all():
        raise ValueError("分类标签存在非整数")
    y = y_raw.astype(np.int64)
    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError("分类标签超出 0/1/2 范围")
    if not np.array_equal(y, np.sign(r).astype(np.int64) + 1):
        raise ValueError("分类标签和回归标签的符号不一致")
    if np.any(np.abs(r) > 3):
        raise ValueError("回归标签超出 [-3, 3]")

    shape = {}
    for key, trailing in (("text_bert", (3, 50)), ("text", (50, 768)),
                          ("audio", (50, 74)), ("vision", (50, 35))):
        value = np.asarray(train[key])
        if value.shape != (n, *trailing):
            raise ValueError(f"{key} 维度错误：{value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{key} 存在 NaN/Inf")
        shape[key] = list(value.shape)

    bert = np.asarray(train["text_bert"])
    if not np.isin(bert[:, 1], [0, 1]).all():
        raise ValueError("text_bert 注意力掩码不是 0/1")
    content = np.zeros((n, 50), dtype=bool)
    for i in range(n):
        cls = np.flatnonzero(bert[i, 0] == 101)
        sep = np.flatnonzero(bert[i, 0] == 102)
        if len(cls) == 0 or len(sep) == 0 or cls[0] >= sep[-1]:
            raise ValueError(f"{ids[i]} 的 CLS/SEP 位置异常")
        content[i, cls[0] + 1:sep[-1]] = True
    if not content.any(axis=1).all():
        raise ValueError("存在空文本内容窗口")
    missing = {}
    for key in ("audio", "vision"):
        observed = content & np.any(np.asarray(train[key]) != 0, axis=-1)
        missing[key] = {
            "whole_sample_unavailable": int((observed.sum(axis=1) == 0).sum()),
            "unavailable_content_positions": int((content.sum() - observed.sum())),
            "content_positions": int(content.sum()),
        }
    text_observed = content & (bert[:, 1] > 0) & (bert[:, 0] != 0)
    missing["text"] = {
        "whole_sample_unavailable": int((text_observed.sum(axis=1) == 0).sum()),
        "unavailable_content_positions": int(content.sum() - text_observed.sum()),
        "content_positions": int(content.sum()),
    }
    group_rows: dict[str, list[int]] = defaultdict(list)
    for i, group in enumerate(groups):
        group_rows[group].append(i)
    group_size = Counter(groups)
    normalized_texts = [str(value).strip().casefold() for value in train["raw_text"]]
    text_counts = Counter(normalized_texts)
    labels_by_text: dict[str, set[int]] = defaultdict(set)
    for item, label in zip(normalized_texts, y):
        labels_by_text[item].add(int(label))
    report = {
        "samples": n,
        "video_groups": len(group_rows),
        "class_counts_negative_neutral_positive": np.bincount(y, minlength=3).tolist(),
        "sample_id_order_sha256": sha256_id_order(ids),
        "feature_shapes": shape,
        "modality_availability": missing,
        "attention_mask_at_50": int((bert[:, 1].sum(axis=1) == 50).sum()),
        "content_unknown_token_id100": int((content & (bert[:, 0] == 100)).sum()),
        "largest_video_group_size": max(group_size.values()),
        "groups_with_multiple_classes": int(sum(len(set(y[rows])) > 1 for rows in group_rows.values())),
        "empty_raw_text": int(sum(not str(t).strip() for t in train["raw_text"])),
        "duplicate_normalized_raw_text_keys": int(sum(count > 1 for count in text_counts.values())),
        "duplicate_normalized_raw_text_extra_rows": int(sum(count - 1 for count in text_counts.values() if count > 1)),
        "duplicate_text_keys_with_different_train_labels": int(sum(len(labels_by_text[item]) > 1 for item in text_counts)),
        "duplicate_text_rows_with_different_train_labels": int(sum(text_counts[item] for item in text_counts if len(labels_by_text[item]) > 1)),
    }
    return ids, y, report


def inspect_split_ids(data: dict) -> dict:
    split_ids = {}
    split_groups = {}
    split_texts = {}
    report = {}
    for split in ("train", "valid", "test"):
        ids = [str(value) for value in data[split]["id"]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{split} 内有重复样本 ID")
        split_ids[split] = set(ids)
        split_groups[split] = {video_id(sid) for sid in ids}
        split_texts[split] = {str(value).strip().casefold() for value in data[split]["raw_text"]}
        report[split] = {"samples": len(ids), "video_groups": len(split_groups[split])}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        common_ids = split_ids[left] & split_ids[right]
        common_groups = split_groups[left] & split_groups[right]
        report[f"{left}_{right}_overlap"] = {
            "sample_ids": len(common_ids), "video_groups": len(common_groups),
            "exact_normalized_raw_text_keys_warning_only": len(split_texts[left] & split_texts[right]),
        }
        if common_ids or common_groups:
            raise ValueError(f"{left}/{right} 样本或视频分组泄漏："
                             f"{len(common_ids)} 个样本，{len(common_groups)} 个视频")
    return report


def contiguous_runs(mask: np.ndarray) -> list[list[int]]:
    positions = np.flatnonzero(mask)
    if len(positions) == 0:
        return []
    cuts = np.flatnonzero(np.diff(positions) > 1) + 1
    return [[int(part[0]), int(part[-1])] for part in np.split(positions, cuts)]


def inspect_annex3(directory: Path) -> dict:
    """仅检查无标签附件3的输入格式与缺失形态，不参与模型选择。"""
    files = sorted(directory.glob("附件3_*.pkl"))
    if len(files) != 30:
        raise ValueError(f"附件3对齐文件应为30个，实际为{len(files)}个")
    rows = []
    totals = Counter()
    for file in files:
        with file.open("rb") as stream:
            item = pickle.load(stream)
        if set(item) != {"test"}:
            raise ValueError(f"{file.name} 的外层键异常：{sorted(item)}")
        sample = item["test"]
        if set(sample) != {"text_bert", "audio", "vision"}:
            raise ValueError(f"{file.name} 含意外键：{sorted(sample)}")
        bert = np.asarray(sample["text_bert"])
        audio = np.asarray(sample["audio"])
        vision = np.asarray(sample["vision"])
        if bert.shape != (1, 3, 50) or audio.shape != (1, 50, 74) or vision.shape != (1, 50, 35):
            raise ValueError(f"{file.name} 输入维度异常：{bert.shape}, {audio.shape}, {vision.shape}")
        if not np.isfinite(bert).all() or not np.isfinite(audio).all() or not np.isfinite(vision).all():
            raise ValueError(f"{file.name} 存在非有限输入")
        token = bert[0, 0]
        attention = bert[0, 1]
        cls = np.flatnonzero(token == 101)
        sep = np.flatnonzero(token == 102)
        if len(cls) == 0 or len(sep) == 0 or cls[0] >= sep[-1]:
            raise ValueError(f"{file.name} 无法定位内容窗口")
        content = np.zeros(50, dtype=bool)
        content[cls[0] + 1:sep[-1]] = True
        unknown = content & (token == 100)
        zero_audio = content & ~np.any(audio[0] != 0, axis=-1)
        zero_vision = content & ~np.any(vision[0] != 0, axis=-1)
        token_mask_missing = content & (attention == 0)
        token_missing_revised = token_mask_missing | unknown
        if np.any(unknown & (attention == 0)):
            raise ValueError(f"{file.name} 出现 ID100 且 attention=0，需重新核实缺失定义")
        if np.any(unknown & ~zero_audio) or np.any(unknown & ~zero_vision):
            raise ValueError(f"{file.name} 出现 ID100 但 A/V 非零，不能简单作为共同缺失")
        row = {
            "sample": file.stem,
            "content_positions": int(content.sum()),
            "unknown_id100_positions": int(unknown.sum()),
            "unknown_attended_positions": int((unknown & (attention == 1)).sum()),
            "unknown_runs_inclusive_0_based": contiguous_runs(unknown),
            "unknown_run_lengths": [b - a + 1 for a, b in contiguous_runs(unknown)],
            "unknown_positions_0_based": np.flatnonzero(unknown).tolist(),
            "text_missing_attention_only": int(token_mask_missing.sum()),
            "text_missing_attention_or_unknown": int(token_missing_revised.sum()),
            "audio_zero_content_positions": int(zero_audio.sum()),
            "vision_zero_content_positions": int(zero_vision.sum()),
            "all_three_missing_revised_positions": int((token_missing_revised & zero_audio & zero_vision).sum()),
            "audio_zero_outside_unknown": int((zero_audio & ~unknown).sum()),
            "vision_zero_outside_unknown": int((zero_vision & ~unknown).sum()),
        }
        rows.append(row)
        for key in row:
            if key not in ("sample", "unknown_runs_inclusive_0_based", "unknown_run_lengths",
                           "unknown_positions_0_based"):
                totals[key] += row[key]
    totals["samples_with_unknown"] = sum(row["unknown_id100_positions"] > 0 for row in rows)
    totals["unknown_runs"] = sum(len(row["unknown_runs_inclusive_0_based"]) for row in rows)
    totals["longest_unknown_run"] = max((max(row["unknown_run_lengths"], default=0) for row in rows), default=0)
    return {"summary": dict(totals), "samples": rows,
            "interpretation": "ID100=[UNK] 且 attention=1；其位置A/V同时为零。该占位符不构成有效文本证据，应在BERT编码前视为缺失并屏蔽。附件3无标签，不可用于挑选分类模型。"}


def make_group_folds(ids: list[str], y: np.ndarray, folds: int, seed: int, starts: int) -> tuple[np.ndarray, dict]:
    group_ids = sorted({video_id(sid) for sid in ids})
    group_index = {group: i for i, group in enumerate(group_ids)}
    g_of_row = np.fromiter((group_index[video_id(sid)] for sid in ids), dtype=np.int64, count=len(ids))
    group_class = np.zeros((len(group_ids), 3), dtype=np.int64)
    np.add.at(group_class, (g_of_row, y), 1)
    total = group_class.sum(axis=0).astype(np.float64)
    target_class = total / folds
    target_size = len(ids) / folds
    target_groups = len(group_ids) / folds
    best_key = None
    best_assignment = None
    for trial in range(starts):
        rng = np.random.default_rng(seed + trial * 10007)
        # Large and class-specific groups first; random jitter breaks ties reproducibly.
        priority = np.max(group_class / total, axis=1) + (group_class.sum(axis=1) / len(ids))
        order = np.argsort(-(priority + rng.uniform(0, 0.002, len(group_ids))), kind="stable")
        counts = np.zeros((folds, 3), dtype=np.float64)
        n_groups = np.zeros(folds, dtype=np.int64)
        assignment = np.full(len(group_ids), -1, dtype=np.int64)
        for group in order:
            values = group_class[group]
            new_counts = counts + values
            class_cost = np.mean(((new_counts - target_class) / target_class) ** 2, axis=1)
            size_cost = ((new_counts.sum(axis=1) - target_size) / target_size) ** 2
            group_cost = ((n_groups + 1 - target_groups) / target_groups) ** 2
            old_class_cost = np.mean(((counts - target_class) / target_class) ** 2, axis=1)
            old_size_cost = ((counts.sum(axis=1) - target_size) / target_size) ** 2
            old_group_cost = ((n_groups - target_groups) / target_groups) ** 2
            candidate = ((class_cost - old_class_cost)
                         + 0.25 * (size_cost - old_size_cost)
                         + 0.01 * (group_cost - old_group_cost))
            # No random tie choice is needed: trial-specific order already varies.
            fold = int(np.argmin(candidate))
            counts[fold] += values
            n_groups[fold] += 1
            assignment[group] = fold
        final_class_cost = np.max(np.abs((counts - target_class) / target_class))
        final_size_cost = np.max(np.abs((counts.sum(axis=1) - target_size) / target_size))
        final_cost = (final_class_cost, final_size_cost, float(np.mean(((counts - target_class) / target_class) ** 2)))
        if best_key is None or final_cost < best_key:
            best_key, best_assignment = final_cost, assignment.copy()
    row_fold = best_assignment[g_of_row]
    if np.any(row_fold < 0):
        raise AssertionError("存在未分配的视频组")
    summary = {}
    for fold in range(folds):
        selection = row_fold == fold
        if not selection.any() or np.bincount(y[selection], minlength=3).min() == 0:
            raise AssertionError(f"fold {fold} 样本为空或缺少某一类")
        summary[str(fold)] = {
            "validation_samples": int(selection.sum()),
            "training_samples": int((~selection).sum()),
            "validation_video_groups": int(np.sum(best_assignment == fold)),
            "validation_class_counts_negative_neutral_positive": np.bincount(y[selection], minlength=3).tolist(),
        }
    all_indices = np.concatenate([np.flatnonzero(row_fold == f) for f in range(folds)])
    assert np.array_equal(np.sort(all_indices), np.arange(len(ids)))
    return row_fold, summary


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned", type=Path, required=True, help="附件2 aligned_50.pkl 的路径")
    parser.add_argument("--annex3", type=Path, help="可选：附件3对齐版文件夹，仅审计无标签输入")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--starts", type=int, default=32)
    args = parser.parse_args()
    if args.folds < 2 or args.starts < 1:
        parser.error("--folds 至少为 2，--starts 至少为 1")
    source = args.aligned.resolve(strict=True)
    with source.open("rb") as stream:
        data = pickle.load(stream)
    if set(data) != {"train", "valid", "test"}:
        raise ValueError(f"附件2的划分键异常：{sorted(data)}")
    ids, y, train_report = inspect_train(data["train"])
    splits = inspect_split_ids(data)
    row_fold, fold_report = make_group_folds(ids, y, args.folds, args.seed, args.starts)
    groups_by_fold = [set(video_id(ids[i]) for i in np.flatnonzero(row_fold == fold)) for fold in range(args.folds)]
    for i in range(args.folds):
        for j in range(i + 1, args.folds):
            if groups_by_fold[i] & groups_by_fold[j]:
                raise AssertionError(f"fold {i}/{j} 存在视频交叉")
    args.output.mkdir(parents=True, exist_ok=True)
    source_sha256 = sha256_file(source)
    manifest = {
        "source_filename": source.name,
        "source_sha256": source_sha256,
        "train_sample_id_order_sha256": train_report["sample_id_order_sha256"],
        "seed": args.seed,
        "folds": args.folds,
        "starts": args.starts,
        "group_rule": "取样本 ID 最后一个 $_$ 前的字符作为视频 ID；同视频的全部片段进入同一验证折",
        "scope": "仅附件2 train；valid/test 的标签未读取，附件3未读取",
        "selection_rule": "train 内分组交叉验证用于方法比较；既有 valid 供最终定型诊断；test 不参与开发选择",
        "fold_summary": fold_report,
        "validation_indices_by_fold": {
            str(fold): np.flatnonzero(row_fold == fold).tolist() for fold in range(args.folds)
        },
        "validation_sample_ids_by_fold": {
            str(fold): [ids[i] for i in np.flatnonzero(row_fold == fold)] for fold in range(args.folds)
        },
    }
    audit = {
        "source_filename": source.name,
        "source_sha256": source_sha256,
        "train": train_report,
        "split_id_leakage": splits,
        "fold_summary": fold_report,
        "fold_group_overlap_count": 0,
        "text_duplicate_interpretation": "完全相同的短转写可因语境和音视频不同而有不同情感标签；文本重合是复核线索，不能据此删除或修改标签。",
        "evaluation_protocol": [
            "拟合归一化、类别权重、任何特征选择均须在每折的训练子集重新计算；验证折只能评价和选择。",
            "样本以视频 ID 分组，所有同视频片段不可跨训练/验证折。",
            "比较模型时固定这五折、缺失掩码随机种子和评价指标；报告均值、标准差及逐折结果。",
            "定型后用附件2原 train 训练；valid 做最终阈值或早停诊断。附件2 test 已被历史实验查看，后续不应称为完全盲测。",
            "附件3无标签，只提交全量预测；不能用附件3预测分布反向挑选模型。",
        ],
    }
    if args.annex3 is not None:
        audit["annex3_unlabelled_input_shift"] = inspect_annex3(args.annex3.resolve(strict=True))
    write_json(args.output / "问题2_训练集视频分组五折清单.json", manifest)
    write_json(args.output / "问题2_数据泄漏与质量审计.json", audit)
    if args.annex3 is not None:
        write_json(args.output / "问题2_附件3无标签缺失形态.json", audit["annex3_unlabelled_input_shift"])
    with (args.output / "问题2_训练集样本折号.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["样本编号", "视频编号", "验证折号"])
        for sid, fold in zip(ids, row_fold):
            writer.writerow([sid, video_id(sid), int(fold)])
    print(json.dumps({"train": train_report, "split_id_leakage": splits,
                      "fold_summary": fold_report,
                      "annex3_unlabelled_input_shift": audit.get("annex3_unlabelled_input_shift", {}).get("summary"),
                      "output": str(args.output)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
