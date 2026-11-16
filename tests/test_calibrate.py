import numpy as np
import pytest

from guardlens.training.calibrate import (
    attack_logit,
    brier,
    calibrated_score,
    ece,
    fit_temperature,
    plot_reliability,
    reliability_bins,
    sigmoid,
)


def test_attack_logit_matches_one_minus_p_benign() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(scale=5.0, size=(50, 4))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(sigmoid(attack_logit(logits)), 1 - probs[:, 0], atol=1e-9)


def test_attack_logit_is_stable_for_huge_logits() -> None:
    m = attack_logit(np.array([[0.0, 1000.0, -5.0, 3.0], [1000.0, 0.0, 0.0, 0.0]]))
    assert np.all(np.isfinite(m))
    assert m[0] == pytest.approx(1000.0)
    assert m[1] == pytest.approx(np.log(3) - 1000.0)


def test_fit_temperature_recovers_known_temperature() -> None:
    # Labels drawn from sigmoid(z); the model reports 3*z (overconfident) -> T ~ 3.
    rng = np.random.default_rng(1)
    z = rng.normal(scale=2.0, size=20_000)
    y = (rng.random(z.size) < sigmoid(z)).astype(float)
    assert fit_temperature(3.0 * z, y) == pytest.approx(3.0, rel=0.05)


def test_temperature_keeps_ranking() -> None:
    m = np.array([-3.0, -0.5, 0.2, 4.0])
    for t in (0.3, 1.0, 7.0):
        assert np.all(np.diff(calibrated_score(m, t)) > 0)


def test_brier_and_ece_by_hand() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0])
    p = np.array([0.1, 0.1, 0.9, 0.7])
    assert brier(y, p) == pytest.approx((0.01 + 0.01 + 0.01 + 0.09) / 4)
    # 15 bins: 0.1 -> bin 1 (2 items, acc 0), 0.7 -> bin 10, 0.9 -> bin 13.
    assert [n for _, _, n in reliability_bins(y, p)] == [2, 1, 1]
    assert ece(y, p) == pytest.approx((2 * 0.1 + 1 * 0.3 + 1 * 0.1) / 4)


def test_perfect_scores_have_zero_ece() -> None:
    y = np.array([0.0, 1.0, 1.0])
    assert ece(y, y.copy()) == pytest.approx(0.0)


def test_plot_reliability_writes_png(tmp_path) -> None:
    rng = np.random.default_rng(2)
    y = (rng.random(200) < 0.4).astype(float)
    p = rng.random(200)
    path = plot_reliability(y, p, np.clip(p * 0.9, 0, 1), tmp_path / "rel.png", "test")
    assert path.exists() and path.stat().st_size > 0
