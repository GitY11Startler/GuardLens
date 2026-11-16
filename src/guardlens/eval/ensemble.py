"""Fixed-rule ensembles, built from the component detectors' saved predictions.

Nothing is fitted, so an ensemble's dev numbers are as honest as its components'.
- mean:    score = average of the components' scores.
- cascade: components [first, second]. When `first` is confident (score outside the band
           [lo, hi]) its score stands; inside the band the `second` detector decides, and its
           score is mapped into the band (lo + (hi - lo) * second), so the ranking stays
           consistent: a text sent to `second` never outranks one `first` was sure about.

The result is written like any other run (`results/runs/<name>_<split>_<version>/`), so the
report treats ensembles as detectors. Latency per text is rebuilt from the components'
latency samples (same seeded texts for every detector): mean = sum of both; cascade = first,
plus second only for texts inside the band. Cost per 1k: mean = sum; cascade = first +
(share of texts in the band) x second.
"""

import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from guardlens.data.schema import Example
from guardlens.detectors.base import NO_CATEGORY
from guardlens.eval.config import EnsembleConfig, LatencyConfig
from guardlens.eval.runner import (
    Prediction,
    RunConfig,
    assert_split_allowed,
    load_predictions,
    run_dir_for,
)
from guardlens.run_info import eval_tag_at_head, git_commit


def in_band(score: float, band: tuple[float, float]) -> bool:
    return band[0] <= score <= band[1]


def combine(cfg: EnsembleConfig, components: dict[str, list[Prediction]]) -> list[Prediction]:
    """Ensemble predictions, in the order of the first component."""
    names = cfg.components
    by_id = {n: {p.id: p for p in components[n]} for n in names}
    ids = [p.id for p in components[names[0]]]
    for n in names[1:]:
        if set(by_id[n]) != set(ids):
            raise ValueError(f"{n} was not scored on the same examples as {names[0]}")

    out = []
    for i in ids:
        parts = [by_id[n][i] for n in names]
        if cfg.kind == "mean":
            score = sum(p.score for p in parts) / len(parts)
            decider = parts[0]
            reason = f"mean of {', '.join(names)}"
        else:
            first, second = parts
            lo, hi = cfg.band
            if in_band(first.score, cfg.band):
                score = lo + (hi - lo) * second.score
                decider = second
                reason = f"{names[0]} unsure ({first.score:.2f}) -> {names[1]}"
            else:
                score = first.score
                decider = first
                reason = f"{names[0]} confident ({first.score:.2f})"
        out.append(
            parts[0].model_copy(
                update={
                    "score": score,
                    "pred_category": decider.pred_category if score >= 0.5 else NO_CATEGORY,
                    "reason": reason,
                    "batch_ms_per_example": sum(p.batch_ms_per_example for p in parts),
                }
            )
        )
    return out


def ensemble_latency(
    cfg: EnsembleConfig,
    latencies: dict[str, list[float]],
    sampled_ids: list[str],
    first_scores: dict[str, float],
) -> list[float]:
    names = cfg.components
    if any(len(latencies[n]) != len(sampled_ids) for n in names):
        return []  # a component has no latency sample: leave it empty rather than guess
    if cfg.kind == "mean":
        return [sum(latencies[n][k] for n in names) for k in range(len(sampled_ids))]
    first, second = names
    return [
        latencies[first][k] + (latencies[second][k] if in_band(first_scores[i], cfg.band) else 0.0)
        for k, i in enumerate(sampled_ids)
    ]


def ensemble_cost(
    cfg: EnsembleConfig, costs: dict[str, float], first_preds: list[Prediction]
) -> dict[str, float]:
    names = cfg.components
    if cfg.kind == "mean":
        return {"cost_per_1k_usd": sum(costs[n] for n in names)}
    share = sum(in_band(p.score, cfg.band) for p in first_preds) / len(first_preds)
    return {
        "cost_per_1k_usd": costs[names[0]] + share * costs[names[1]],
        "share_sent_to_second": share,
    }


def _read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text("utf-8")) if path.exists() else default


def build_ensemble(
    name: str,
    cfg: EnsembleConfig,
    split: str,
    data_version: str,
    results_dir: Path,
    examples: list[Example],
    latency_cfg: LatencyConfig,
) -> Path:
    assert_split_allowed(split)
    dirs = {n: run_dir_for(results_dir, n, split, data_version) for n in cfg.components}
    for n, d in dirs.items():
        if not (d / "predictions.jsonl").exists():
            raise FileNotFoundError(f"{name} needs a {n} run on {split}: run `gl eval` first")
    components = {n: load_predictions(d) for n, d in dirs.items()}
    preds = combine(cfg, components)

    run_dir = run_dir_for(results_dir, name, split, data_version)
    run_dir.mkdir(parents=True, exist_ok=True)
    component_configs = {
        n: _read_json(d / "run_config.json", {})["detector"] for n, d in dirs.items()
    }
    config = RunConfig(
        detector={"name": name, **cfg.model_dump(), "component_configs": component_configs},
        split=split,
        data_version=data_version,
        data_manifest_sha256=_read_json(dirs[cfg.components[0]] / "run_config.json", {})[
            "data_manifest_sha256"
        ],
        git_commit=git_commit(),
        eval_tag=eval_tag_at_head(),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    # Deterministic from the component runs, so it is simply rebuilt every time.
    (run_dir / "run_config.json").write_text(config.model_dump_json(indent=2), "utf-8")
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(p.model_dump_json() + "\n")

    sample = random.Random(latency_cfg.seed).sample(
        examples, min(latency_cfg.sample, len(examples))
    )  # the same texts measure_latency timed for every component
    latencies = {n: _read_json(d / "latency.json", []) for n, d in dirs.items()}
    first_scores = {p.id: p.score for p in components[cfg.components[0]]}
    latency = ensemble_latency(cfg, latencies, [e.id for e in sample], first_scores)
    (run_dir / "latency.json").write_text(json.dumps(latency), "utf-8")

    costs = {
        n: float(_read_json(d / "cost.json", {}).get("cost_per_1k_usd", 0.0))
        for n, d in dirs.items()
    }
    cost = ensemble_cost(cfg, costs, components[cfg.components[0]])
    (run_dir / "cost.json").write_text(json.dumps(cost, indent=2), "utf-8")
    return run_dir
