"""`gl report`: turn run folders into summary CSVs and figures (counts and metrics only).

Each detector gets one row per operating point (threshold):
- `tpr95_on_dev`: the score that catches `target_tpr` of **dev** attacks, so a test report
  later reuses it unchanged;
- any fixed points from `extra_operating_points` in eval.yaml (e.g. D0 "any rule fires").
Threshold-free metrics (AUROC, AP, FPR @ target TPR) are the same on every row of a detector.

On test, every metric is also computed per slice: all / seen / held-out (eval/slices.py).
Outputs: summary_<split>.csv, per_group_<split>.csv, comparisons_<split>.csv,
fpr_at_tpr_<split>.csv, headline_<split>.csv (+ summary.csv for test) and figures.
"""

import csv
import itertools
import json
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from guardlens.eval.config import EvalConfig
from guardlens.eval.curves import plot_all
from guardlens.eval.metrics import (
    at_threshold,
    auroc,
    average_precision,
    fpr_at_tpr,
    latency_summary,
    per_group_rates,
    threshold_at_tpr,
)
from guardlens.eval.runner import Prediction, load_predictions, run_dir_for
from guardlens.eval.slices import SLICES, slices
from guardlens.eval.stats import bootstrap_ci, paired_bootstrap


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def label_score_arrays(preds: list[Prediction]) -> tuple[np.ndarray, np.ndarray]:
    return np.array([p.label for p in preds]), np.array([p.score for p in preds])


def operating_points(
    name: str, dev_preds: list[Prediction], config: EvalConfig
) -> dict[str, float]:
    """Operating point name -> threshold for one detector."""
    y_dev, s_dev = label_score_arrays(dev_preds)
    points = {
        f"tpr{int(config.target_tpr * 100)}_on_dev": threshold_at_tpr(
            y_dev, s_dev, config.target_tpr
        )
    }
    points.update(config.extra_operating_points.get(name, {}))
    return points


def summarize(
    name: str,
    preds: list[Prediction],
    operating_point: str,
    threshold: float,
    config: EvalConfig,
    latency_ms: list[float],
    cost_per_1k_usd: float = 0.0,
) -> dict[str, Any]:
    y, s = label_score_arrays(preds)
    b = config.bootstrap
    t = config.target_tpr

    def ci(metric: Any) -> Any:
        return bootstrap_ci(y, s, metric, b.resamples, b.seed)

    au = ci(auroc)
    ap = ci(average_precision)
    fpr_t = ci(lambda yy, ss: fpr_at_tpr(yy, ss, t))
    f1 = ci(lambda yy, ss: at_threshold(yy, ss, threshold)["f1"])
    m = at_threshold(y, s, threshold)
    lat = latency_summary(latency_ms) if latency_ms else {"p50_ms": np.nan, "p95_ms": np.nan}
    return {
        "detector": name,
        "n": len(y),
        "n_attack": int(y.sum()),
        "auroc": au.value,
        "auroc_low": au.low,
        "auroc_high": au.high,
        "avg_precision": ap.value,
        "avg_precision_low": ap.low,
        "avg_precision_high": ap.high,
        f"fpr_at_tpr{int(t * 100)}": fpr_t.value,
        f"fpr_at_tpr{int(t * 100)}_low": fpr_t.low,
        f"fpr_at_tpr{int(t * 100)}_high": fpr_t.high,
        "operating_point": operating_point,
        "threshold": threshold,
        "precision": m["precision"],
        "recall": m["recall"],
        "fpr": m["fpr"],
        "f1": f1.value,
        "f1_low": f1.low,
        "f1_high": f1.high,
        "latency_p50_ms": lat["p50_ms"],
        "latency_p95_ms": lat["p95_ms"],
        "cost_per_1k_usd": cost_per_1k_usd,  # 0 for local detectors
    }


def group_rows(
    name: str, preds: list[Prediction], operating_point: str, threshold: float
) -> list[dict[str, Any]]:
    y, s = label_score_arrays(preds)
    rows: list[dict[str, Any]] = []
    for kind, groups in (
        ("family_or_category", [p.family or p.category for p in preds]),
        ("source", [p.source for p in preds]),
    ):
        for r in per_group_rates(y, s, groups, threshold):
            k = round(float(r["value"]) * int(r["n"]))
            low, high = wilson(k, int(r["n"]))
            rows.append(
                {
                    "detector": name,
                    "operating_point": operating_point,
                    "grouping": kind,
                    **r,
                    "low": low,
                    "high": high,
                }
            )
    return rows


def comparison_rows(runs: dict[str, list[Prediction]], config: EvalConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    t, b = config.target_tpr, config.bootstrap
    metrics = {
        "auroc": auroc,
        f"fpr_at_tpr{int(t * 100)}": lambda yy, ss: fpr_at_tpr(yy, ss, t),
    }
    for a, c in itertools.combinations(sorted(runs), 2):
        by_id = {p.id: p for p in runs[c]}
        common = [p for p in runs[a] if p.id in by_id]
        y = np.array([p.label for p in common])
        s_a = np.array([p.score for p in common])
        s_c = np.array([by_id[p.id].score for p in common])
        for metric_name, fn in metrics.items():
            r = paired_bootstrap(y, s_a, s_c, fn, b.resamples, b.seed)
            rows.append(
                {
                    "a": a,
                    "b": c,
                    "metric": metric_name,
                    "n": len(common),
                    "diff_a_minus_b": r.diff.value,
                    "low": r.diff.low,
                    "high": r.diff.high,
                    "p_value": r.p_value,
                }
            )
    return rows


def _fpr_at(y: np.ndarray, s: np.ndarray, level: float) -> float:
    return fpr_at_tpr(y, s, level)


def fpr_at_tpr_rows(
    name: str, slice_name: str, preds: list[Prediction], config: EvalConfig
) -> list[dict[str, Any]]:
    """FPR at several catch rates, read off this slice's own ROC curve (threshold-free)."""
    y, s = label_score_arrays(preds)
    b = config.bootstrap
    rows = []
    for level in config.fpr_at_tpr_levels:
        ci = bootstrap_ci(y, s, partial(_fpr_at, level=level), b.resamples, b.seed)
        rows.append(
            {
                "detector": name,
                "slice": slice_name,
                "catch_rate": level,
                "fpr": ci.value,
                "low": ci.low,
                "high": ci.high,
            }
        )
    return rows


def headline_rows(summaries: list[dict[str, Any]], config: EvalConfig) -> list[dict[str, Any]]:
    """One row per detector at its main operating point, slices side by side."""
    default = f"tpr{int(config.target_tpr * 100)}_on_dev"
    rows: list[dict[str, Any]] = []
    for name in dict.fromkeys(r["detector"] for r in summaries):
        point = config.main_operating_point.get(name, default)
        mine = {
            r["slice"]: r
            for r in summaries
            if r["detector"] == name and r["operating_point"] == point
        }
        if not mine:
            continue
        first = mine["all"]
        row: dict[str, Any] = {
            "detector": name,
            "operating_point": point,
            "threshold": first["threshold"],
        }
        for slice_name in SLICES:
            if slice_name not in mine:
                continue
            r = mine[slice_name]
            for key in (
                "auroc",
                "auroc_low",
                "auroc_high",
                "recall",
                "fpr",
                "f1",
                "f1_low",
                "f1_high",
            ):
                row[f"{key}_{slice_name}"] = r[key]
        row.update(
            latency_p50_ms=first["latency_p50_ms"],
            latency_p95_ms=first["latency_p95_ms"],
            cost_per_1k_usd=first["cost_per_1k_usd"],
        )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    detectors: list[str],
    split: str,
    config: EvalConfig,
    results_dir: Path,
    held_out_sources: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    runs: dict[str, list[Prediction]] = {}
    summaries: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    fpr_rows: list[dict[str, Any]] = []
    for name in detectors:
        run_dir = run_dir_for(results_dir, name, split, config.data_version)
        dev_dir = run_dir_for(results_dir, name, "dev", config.data_version)
        if not (run_dir / "predictions.jsonl").exists():
            print(f"[report] skipping {name}: no run in {run_dir}")
            continue
        preds = load_predictions(run_dir)
        lat_path = run_dir / "latency.json"
        latency = json.loads(lat_path.read_text("utf-8")) if lat_path.exists() else []
        # Paid detectors write cost.json (mean $ per API call x 1000); local ones don't.
        cost_path = run_dir / "cost.json"
        cost = json.loads(cost_path.read_text("utf-8")) if cost_path.exists() else {}
        cost_per_1k = float(cost.get("cost_per_1k_usd", 0.0))
        runs[name] = preds
        points = operating_points(name, load_predictions(dev_dir), config)
        for slice_name, sliced in slices(preds, held_out_sources).items():
            for point, threshold in points.items():
                row = summarize(name, sliced, point, threshold, config, latency, cost_per_1k)
                summaries.append({"slice": slice_name, **row})
            fpr_rows.extend(fpr_at_tpr_rows(name, slice_name, sliced, config))
        for point, threshold in points.items():
            groups.extend(group_rows(name, preds, point, threshold))

    if not runs:
        raise FileNotFoundError(f"no runs found for split {split!r}; run `gl eval` first")
    _write_csv(results_dir / f"summary_{split}.csv", summaries)
    _write_csv(results_dir / f"per_group_{split}.csv", groups)
    _write_csv(results_dir / f"fpr_at_tpr_{split}.csv", fpr_rows)
    headline = headline_rows(summaries, config)
    _write_csv(results_dir / f"headline_{split}.csv", headline)
    if split == "test":
        _write_csv(results_dir / "summary.csv", headline)  # the M5 deliverable

    sliced_runs = {n: slices(p, held_out_sources) for n, p in runs.items()}
    comparisons: list[dict[str, Any]] = []
    for slice_name in SLICES:
        in_slice = {n: s[slice_name] for n, s in sliced_runs.items() if slice_name in s}
        comparisons.extend({"slice": slice_name, **r} for r in comparison_rows(in_slice, config))
    if comparisons:
        _write_csv(results_dir / f"comparisons_{split}.csv", comparisons)

    plot_all(
        {n: label_score_arrays(p) for n, p in runs.items()},
        results_dir / "figures",
        split,
        config.target_tpr,
    )
    held = {n: s["held_out"] for n, s in sliced_runs.items() if "held_out" in s}
    if held:
        plot_all(
            {n: label_score_arrays(p) for n, p in held.items()},
            results_dir / "figures",
            f"{split}_heldout",
            config.target_tpr,
        )
    return summaries
