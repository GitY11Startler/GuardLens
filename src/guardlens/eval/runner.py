"""Run one detector over one split.

Writes to `results/runs/<detector>_<split>_<data version>/`:
  run_config.json     detector config, data version + manifest hash, git commit, time
  predictions.jsonl   one line per example (ids, labels and scores; no text). Git-ignored.
  latency.json        per-request latency, measured separately without any cache

Resumable: examples already in predictions.jsonl are skipped. A run folder can only be
resumed with the same detector config and data; otherwise the runner refuses.
"""

import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from guardlens.data.schema import Example
from guardlens.detectors.base import Detector
from guardlens.eval.config import LatencyConfig
from guardlens.run_info import eval_tag_at_head, git_commit


class RunConfig(BaseModel):
    detector: dict[str, Any]
    split: str
    data_version: str
    data_manifest_sha256: str
    git_commit: str
    eval_tag: str | None
    created_at: str


class Prediction(BaseModel):
    id: str
    label: int
    source: str
    family: str | None
    category: str  # true category of the example
    score: float
    pred_category: str
    reason: str
    batch_ms_per_example: float


def assert_split_allowed(split: str) -> None:
    """Test runs only from a tagged `eval-v*` commit (CLAUDE.md, evaluation integrity)."""
    if split == "test" and eval_tag_at_head() is None:
        raise PermissionError(
            "The test split can only be evaluated from a commit tagged eval-v*. "
            "Tune on dev; tag a commit when the configuration is frozen."
        )


def run_dir_for(results_dir: Path, detector: str, split: str, data_version: str) -> Path:
    return results_dir / "runs" / f"{detector}_{split}_{data_version}"


def _check_or_write_config(run_dir: Path, config: RunConfig) -> None:
    path = run_dir / "run_config.json"
    if path.exists():
        old = RunConfig.model_validate_json(path.read_text("utf-8"))
        same = (old.detector, old.split, old.data_version, old.data_manifest_sha256) == (
            config.detector,
            config.split,
            config.data_version,
            config.data_manifest_sha256,
        )
        if not same:
            raise ValueError(
                f"{run_dir} was produced with a different detector config or data. "
                "Delete that folder to start a fresh run."
            )
        return
    path.write_text(config.model_dump_json(indent=2), "utf-8")


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def run_detector(
    detector: Detector,
    examples: list[Example],
    run_dir: Path,
    split: str,
    data_version: str,
    data_manifest_sha256: str,
    batch_size: int = 256,
) -> Path:
    assert_split_allowed(split)
    run_dir.mkdir(parents=True, exist_ok=True)
    _check_or_write_config(
        run_dir,
        RunConfig(
            detector=detector.config(),
            split=split,
            data_version=data_version,
            data_manifest_sha256=data_manifest_sha256,
            git_commit=git_commit(),
            eval_tag=eval_tag_at_head(),
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        ),
    )

    pred_path = run_dir / "predictions.jsonl"
    done = _done_ids(pred_path)
    todo = [e for e in examples if e.id not in done]
    print(f"[{detector.name}] {len(done)} already scored, {len(todo)} to go")

    with pred_path.open("a", encoding="utf-8") as f:
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            t0 = time.perf_counter()
            outputs = detector.score_batch([e.text for e in batch], "input")
            ms = (time.perf_counter() - t0) * 1000 / len(batch)
            for e, out in zip(batch, outputs, strict=True):
                pred = Prediction(
                    id=e.id,
                    label=e.label,
                    source=e.source,
                    family=e.family,
                    category=e.category,
                    score=out.score,
                    pred_category=out.category,
                    reason=out.reason,
                    batch_ms_per_example=ms,
                )
                f.write(pred.model_dump_json() + "\n")
            f.flush()  # a crash loses at most one batch
            print(
                f"[{detector.name}] {len(done) + start + len(batch)}/{len(examples)} scored",
                flush=True,
            )
    return pred_path


def measure_latency(
    detector: Detector, examples: list[Example], config: LatencyConfig
) -> list[float]:
    """Per-request latency in ms: one text at a time, the detector's cache bypassed."""
    sample = random.Random(config.seed).sample(examples, min(config.sample, len(examples)))
    texts = [e.text for e in sample]
    with detector.without_cache():
        for t in texts[: config.warmup]:
            detector.score(t)
        latencies = []
        for t in texts:
            t0 = time.perf_counter()
            detector.score(t)
            latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def pilot_sample(examples: list[Example], size: int, seed: int) -> list[Example]:
    """`size` examples spread evenly over the categories (fewer if a category is small)."""
    rng = random.Random(seed)
    by_cat: dict[str, list[Example]] = {}
    for e in examples:
        by_cat.setdefault(e.category, []).append(e)
    per_cat = size // len(by_cat)
    sample: list[Example] = []
    for cat in sorted(by_cat):
        group = by_cat[cat]
        sample.extend(rng.sample(group, min(per_cat, len(group))))
    return sample


def load_predictions(run_dir: Path) -> list[Prediction]:
    with (run_dir / "predictions.jsonl").open(encoding="utf-8") as f:
        return [Prediction.model_validate_json(line) for line in f if line.strip()]
