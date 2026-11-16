"""Detector metrics. y = 1 for attacks, s = detector score; flag as attack if s >= thr."""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

Array = np.ndarray


def _roc(y: Array, s: Array) -> tuple[Array, Array, Array]:
    # drop_intermediate=False keeps every threshold, so "first point reaching the target TPR"
    # is exact.
    fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)
    return fpr, tpr, thr


def threshold_at_tpr(y: Array, s: Array, target_tpr: float) -> float:
    """Highest threshold that still catches at least `target_tpr` of the attacks."""
    _, tpr, thr = _roc(y, s)
    i = int(np.argmax(tpr >= target_tpr))
    return float(thr[i])


def fpr_at_tpr(y: Array, s: Array, target_tpr: float) -> float:
    """Lowest false-positive rate among thresholds that catch at least `target_tpr` of attacks."""
    fpr, tpr, _ = _roc(y, s)
    return float(fpr[int(np.argmax(tpr >= target_tpr))])


def auroc(y: Array, s: Array) -> float:
    return float(roc_auc_score(y, s))


def average_precision(y: Array, s: Array) -> float:
    return float(average_precision_score(y, s))


def at_threshold(y: Array, s: Array, thr: float) -> dict[str, float]:
    pred = s >= thr
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr}


def per_group_rates(
    y: Array, s: Array, groups: Sequence[str], thr: float
) -> list[dict[str, str | int | float]]:
    """Per group: catch rate for attack groups, false-positive rate for benign groups."""
    groups_arr = np.asarray(groups)
    pred = s >= thr
    rows: list[dict[str, str | int | float]] = []
    for label, metric in ((1, "catch_rate"), (0, "fpr")):
        for g in sorted(set(groups_arr[y == label].tolist())):
            mask = (y == label) & (groups_arr == g)
            rows.append(
                {
                    "group": g,
                    "label": label,
                    "metric": metric,
                    "n": int(mask.sum()),
                    "value": float(pred[mask].mean()),
                }
            )
    return rows


def latency_summary(latencies_ms: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(latencies_ms, dtype=float)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
    }
