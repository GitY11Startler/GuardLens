"""Demo logic with fake detectors; the page itself is only built, never launched."""

import pytest
from fake_detectors import Fixed

from guardlens.service.config import Thresholds
from guardlens.service.demo import COLUMNS, compare

T = Thresholds(d0=0.1, d1=0.7, d2=0.5, d3=0.5, e1=0.6, e2=0.4)
BAND = (0.05, 0.95)


def _detectors(d1: float, judge: Fixed | None = None) -> dict:  # type: ignore[type-arg]
    dets = {"d0": Fixed("d0", 0.0), "d1": Fixed("d1", d1), "d3": Fixed("d3", 0.8)}
    if judge is not None:
        dets["d2"] = judge
    return dets


def _by_name(rows: list[list[object]]) -> dict[object, list[object]]:
    return {r[0]: r for r in rows}


def test_all_rows_present_and_verdicts() -> None:
    rows = _by_name(compare("placeholder", _detectors(0.99), T, BAND, use_judge=False))
    assert list(rows) == ["D0", "D1", "D2", "D3", "E1", "E2"]
    assert all(len(r) == len(COLUMNS) for r in rows.values())
    assert rows["D0"][4] == "allow" and rows["D1"][4] == "BLOCK"
    assert rows["E1"][2] == f"{(0.99 + 0.8) / 2:.3f}"
    assert rows["E2"][4] == "BLOCK"  # D1 confident: E2 = D1's score


def test_judge_runs_only_when_ticked() -> None:
    judge = Fixed("d2", 0.9)
    rows = _by_name(compare("placeholder", _detectors(0.5, judge), T, BAND, use_judge=False))
    assert judge.calls == 0
    assert rows["D2"][4] == "not run (tick the box)"
    assert rows["E2"][4] == "D1 unsure: needs the judge"

    rows = _by_name(compare("placeholder", _detectors(0.5, judge), T, BAND, use_judge=True))
    assert judge.calls == 1
    assert rows["E2"][2] == f"{0.05 + 0.9 * 0.9:.3f}" and rows["E2"][4] == "BLOCK"


def test_missing_judge_is_explained() -> None:
    rows = _by_name(compare("placeholder", _detectors(0.5), T, BAND, use_judge=True))
    assert rows["D2"][4] == "unavailable (no API key)"


def test_page_builds_without_launching() -> None:
    pytest.importorskip("gradio")  # optional `demo` group
    from guardlens.service.demo import build_ui

    demo = build_ui(lambda text, use_judge: [])
    assert demo is not None
