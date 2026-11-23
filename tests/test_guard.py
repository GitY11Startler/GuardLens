"""Service decision logic with fake detectors: no GPU, network or API key."""

import pytest
from fake_detectors import Fixed

from guardlens.detectors.d2_judge import JudgeError
from guardlens.eval.config import load_eval_config
from guardlens.judge_cost import SpendingCapError
from guardlens.service.config import ServiceConfig, Thresholds, load_service_config
from guardlens.service.guard import Guard
from guardlens.settings import get_settings

CONFIG = ServiceConfig(mode="e2", thresholds=Thresholds(d1=0.7, e2=0.4))
BAND = (0.05, 0.95)


def test_confident_d1_decides_without_the_judge() -> None:
    judge = Fixed("d2", 1.0)
    blocked = Guard(Fixed("d1", 0.99), judge, CONFIG, BAND).check("placeholder")
    allowed = Guard(Fixed("d1", 0.01), judge, CONFIG, BAND).check("placeholder")
    assert (blocked.allow, blocked.decided_by, blocked.mode) == (False, "d1", "e2")
    assert (allowed.allow, allowed.decided_by) == (True, "d1")
    assert blocked.threshold == 0.4 and judge.calls == 0
    assert blocked.latency_ms >= 0


def test_unsure_d1_asks_the_judge() -> None:
    out = Guard(Fixed("d1", 0.5), Fixed("d2", 0.9), CONFIG, BAND).check("placeholder")
    assert out.decided_by == "d2" and out.mode == "e2"
    assert out.score == pytest.approx(0.05 + 0.9 * 0.9)  # mapped into the band
    assert not out.allow and out.category == "fam_x" and out.reason == "d2 says"

    benign = Guard(Fixed("d1", 0.5), Fixed("d2", 0.1), CONFIG, BAND).check("placeholder")
    assert benign.allow and benign.category == "none"


@pytest.mark.parametrize(
    "error", [SpendingCapError("cap"), JudgeError("bad answer")], ids=["cap", "answer"]
)
def test_judge_failure_falls_back_to_d1(error: Exception) -> None:
    out = Guard(Fixed("d1", 0.8), Fixed("d2", 0.0, error), CONFIG, BAND).check("placeholder")
    assert out.mode == "d1" and out.decided_by == "d1"
    assert out.threshold == 0.7 and not out.allow  # D1's own threshold: 0.8 >= 0.7
    assert out.fallback is not None and type(error).__name__ in out.fallback


def test_missing_judge_falls_back_with_reason() -> None:
    guard = Guard(Fixed("d1", 0.6), None, CONFIG, BAND, judge_unavailable_reason="no API key")
    out = guard.check("placeholder")
    assert out.mode == "d1" and out.fallback == "no API key"
    assert out.allow  # 0.6 < D1 threshold 0.7


def test_d1_mode_never_calls_the_judge() -> None:
    judge = Fixed("d2", 1.0)
    config = CONFIG.model_copy(update={"mode": "d1"})
    out = Guard(Fixed("d1", 0.5), judge, config, BAND).check("placeholder")
    assert out.mode == "d1" and out.fallback is None and judge.calls == 0


def test_shipped_configs_agree() -> None:
    """configs/service.yaml loads, and the e2 threshold sits inside the e2 band."""
    settings = get_settings()
    service = load_service_config(settings.configs_dir / "service.yaml")
    band = load_eval_config(settings.configs_dir / "eval.yaml").ensembles["e2"].band
    assert service.mode in ("e2", "d1") and service.max_chars == 100_000
    assert band[0] < service.thresholds.e2 < band[1]
    assert 0 < service.thresholds.d1 < 1
