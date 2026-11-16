import numpy as np
import pytest

from guardlens.eval.metrics import (
    at_threshold,
    auroc,
    fpr_at_tpr,
    latency_summary,
    per_group_rates,
    threshold_at_tpr,
)
from guardlens.eval.stats import bootstrap_ci, paired_bootstrap

# 4 attacks, 4 benign. Scores chosen so results can be checked by hand.
Y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
S = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.3, 0.1, 0.0])


def test_auroc_by_hand() -> None:
    # Pairs (attack, benign) ranked correctly: 0.9,0.8,0.7 beat all 4; 0.2 beats 0.1 and 0.0.
    assert auroc(Y, S) == pytest.approx((3 * 4 + 2) / 16)


def test_threshold_and_fpr_at_tpr() -> None:
    # Catching 75% of attacks needs thr = 0.7 -> no benign above 0.7 except none -> FPR 0.
    assert threshold_at_tpr(Y, S, 0.75) == pytest.approx(0.7)
    assert fpr_at_tpr(Y, S, 0.75) == 0.0
    # Catching 100% needs thr = 0.2 -> benign 0.6 and 0.3 are flagged -> FPR 2/4.
    assert threshold_at_tpr(Y, S, 1.0) == pytest.approx(0.2)
    assert fpr_at_tpr(Y, S, 1.0) == pytest.approx(0.5)


def test_at_threshold_by_hand() -> None:
    m = at_threshold(Y, S, 0.5)  # flags 0.9, 0.8, 0.7 (attacks) and 0.6 (benign)
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["f1"] == pytest.approx(3 / 4)
    assert m["fpr"] == pytest.approx(1 / 4)


def test_per_group_rates() -> None:
    groups = ["fam_a", "fam_a", "fam_b", "fam_b", "hard", "hard", "plain", "plain"]
    rows = {(r["group"], r["metric"]): r for r in per_group_rates(Y, S, groups, 0.5)}
    assert rows[("fam_a", "catch_rate")]["value"] == 1.0
    assert rows[("fam_b", "catch_rate")]["value"] == 0.5
    assert rows[("hard", "fpr")]["value"] == 0.5
    assert rows[("plain", "fpr")]["value"] == 0.0


def test_latency_summary() -> None:
    lat = latency_summary([1.0] * 95 + [10.0] * 5)
    assert lat["p50_ms"] == 1.0
    assert lat["mean_ms"] == pytest.approx(1.45)


def _toy(seed: int, n: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    good = y + rng.normal(0, 0.5, n)  # separates classes well
    bad = y + rng.normal(0, 3.0, n)  # mostly noise
    return y, good, bad


def test_bootstrap_ci_is_seeded_and_contains_value() -> None:
    y, good, _ = _toy(0)
    a = bootstrap_ci(y, good, auroc, n_resamples=200, seed=1)
    b = bootstrap_ci(y, good, auroc, n_resamples=200, seed=1)
    assert a == b
    assert a.low <= a.value <= a.high


def test_paired_bootstrap_detects_a_real_difference() -> None:
    y, good, bad = _toy(0)
    result = paired_bootstrap(y, good, bad, auroc, n_resamples=300, seed=2)
    assert result.diff.value > 0
    assert result.diff.low > 0
    assert result.p_value < 0.01


def test_paired_bootstrap_no_difference_with_itself() -> None:
    y, good, _ = _toy(0)
    result = paired_bootstrap(y, good, good, auroc, n_resamples=100, seed=2)
    assert result.diff.value == 0
    assert result.p_value == 1.0
