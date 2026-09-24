"""Audit errors of the selected Q2 model using labeled validation predictions only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CLASSES = ("Negative", "Neutral", "Positive")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(predictions: Path, out: Path) -> None:
    with predictions.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1871 or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Expected 1871 unique validation predictions")
    confusion = [[0] * 3 for _ in range(3)]
    errors = []
    intensity_bands = {"0–0.5": [], "0.5–1.5": [], "1.5–3": []}
    for row in rows:
        true, pred = row["true_class"], row["predicted_class"]
        if true not in CLASSES or pred not in CLASSES:
            raise ValueError(f"Unexpected class: {true}, {pred}")
        confusion[CLASSES.index(true)][CLASSES.index(pred)] += 1
        confidence = max(float(row[f"{label.lower()}_probability"]) for label in CLASSES)
        absolute_error = float(row["absolute_error"])
        true_strength = abs(float(row["true_intensity"]))
        band = "0–0.5" if true_strength < .5 else "0.5–1.5" if true_strength < 1.5 else "1.5–3"
        intensity_bands[band].append(absolute_error)
        if true != pred:
            audit_row = {
                "样本编号": row["sample_id"], "真实类别": true,
                "预测类别": pred, "最高类别概率": confidence,
                "真实强度": float(row["true_intensity"]),
                "预测强度": float(row["predicted_intensity"]),
                "强度绝对误差": absolute_error,
            }
            errors.append(audit_row)
    metrics = {}
    for j, label in enumerate(CLASSES):
        tp = confusion[j][j]
        support = sum(confusion[j])
        predicted = sum(confusion[i][j] for i in range(3))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {"support": support, "precision": precision, "recall": recall, "F1": f1,
                          "misclassified": support - tp}
    accuracy = sum(confusion[i][i] for i in range(3)) / len(rows)
    macro_f1 = sum(metrics[c]["F1"] for c in CLASSES) / 3
    high_conf_errors = sorted((r for r in errors if r["最高类别概率"] >= .8),
                              key=lambda r: r["最高类别概率"], reverse=True)
    large_strength_errors = sorted(errors, key=lambda r: r["强度绝对误差"], reverse=True)[:30]
    result = {
        "protocol": "Post-hoc audit of the selected model's labeled, video-disjoint validation split; no attachment-3 labels",
        "validation_samples": len(rows), "correct": len(rows) - len(errors), "wrong": len(errors),
        "Accuracy": accuracy, "Macro_F1": macro_f1,
        "confusion_rows_true_columns_predicted": confusion,
        "per_class": metrics,
        "error_directions": {f"{CLASSES[i]}→{CLASSES[j]}": confusion[i][j]
                             for i in range(3) for j in range(3) if i != j},
        "high_confidence_error_threshold": .8,
        "high_confidence_error_count": len(high_conf_errors),
        "true_intensity_absolute_bands": {name: {"n": len(values),
            "MAE": sum(values) / len(values) if values else None}
            for name, values in intensity_bands.items()},
        "note": "The highest class probability is an uncalibrated model score, not an empirical probability of correctness.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "问题2_验证集错误归因.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(out / "问题2_高置信误判样本.csv", high_conf_errors)
    write_csv(out / "问题2_大强度误差样本.csv", large_strength_errors)
    print(json.dumps({"Accuracy": accuracy, "Macro_F1": macro_f1,
                      "wrong": len(errors), "high_conf_errors": len(high_conf_errors),
                      "intensity_bands": result["true_intensity_absolute_bands"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.predictions, args.out)
