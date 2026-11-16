"""Bootstrap confidence intervals and paired comparisons (seeded, so results are reproducible)."""

from collections.abc import Callable

import numpy as np
from pydantic import BaseModel

Metric = Callable[[np.ndarray, np.ndarray], float]


class CI(BaseModel):
    value: float  # metric on the full set
    low: float
    high: float


class PairedResult(BaseModel):
    diff: CI  # metric(A) - metric(B)
    p_value: float  # two-sided bootstrap p-value for "no difference"


def _resample_indices(y: np.ndarray, n_resamples: int, seed: int) -> list[np.ndarray]:
    """Resample examples with replacement; skip draws that lose a whole class."""
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    while len(out) < n_resamples:
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == len(np.unique(y)):
            out.append(idx)
    return out


def bootstrap_ci(
    y: np.ndarray,
    s: np.ndarray,
    metric: Metric,
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> CI:
    values = [metric(y[i], s[i]) for i in _resample_indices(y, n_resamples, seed)]
    low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return CI(value=metric(y, s), low=float(low), high=float(high))


def paired_bootstrap(
    y: np.ndarray,
    s_a: np.ndarray,
    s_b: np.ndarray,
    metric: Metric,
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> PairedResult:
    """Same resampled examples for both detectors, so the comparison is paired."""
    diffs = np.array(
        [
            metric(y[i], s_a[i]) - metric(y[i], s_b[i])
            for i in _resample_indices(y, n_resamples, seed)
        ]
    )
    low, high = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    return PairedResult(
        diff=CI(value=metric(y, s_a) - metric(y, s_b), low=float(low), high=float(high)),
        p_value=min(p, 1.0),
    )
