"""The decision the service makes for one text (kept apart from the web layer so it is easy
to test with fake detectors).

mode e2 (the M5 cascade): D1 scores the text. Outside the band (D1 confident) D1's score
stands; inside the band the judge D2 decides, its score mapped into the band exactly as in
eval/ensemble.py. Block when score >= the e2 threshold.
If the judge can't be used (not configured, spending cap reached, API or answer error), the
guard falls back to D1 alone with D1's own threshold and reports why in `fallback`.
mode d1: D1 alone, D1's threshold. Free and offline.
"""

import threading
import time
from typing import Literal

import anthropic

from guardlens.detectors.base import NO_CATEGORY, Detector
from guardlens.detectors.d2_judge import JudgeError
from guardlens.eval.ensemble import in_band
from guardlens.judge_cost import SpendingCapError
from guardlens.service.config import ServiceConfig
from guardlens.service.schemas import CheckResponse

JUDGE_ERRORS = (SpendingCapError, JudgeError, anthropic.APIError)


class Guard:
    def __init__(
        self,
        d1: Detector,
        judge: Detector | None,
        config: ServiceConfig,
        band: tuple[float, float],
        judge_unavailable_reason: str | None = None,
    ) -> None:
        self.d1 = d1
        self.judge = judge
        self.config = config
        self.band = band
        self.judge_unavailable_reason = judge_unavailable_reason
        self._lock = threading.Lock()  # one text at a time through the GPU model

    def check(self, text: str) -> CheckResponse:
        t0 = time.perf_counter()
        with self._lock:
            response = self._decide(text)
        response.latency_ms = (time.perf_counter() - t0) * 1000
        return response

    def _decide(self, text: str) -> CheckResponse:
        first = self.d1.score(text)
        if self.config.mode == "d1":
            return self._d1_only(first.score, first.category, first.reason, fallback=None)
        if not in_band(first.score, self.band):  # D1 confident: no judge needed
            return self._answer("e2", "d1", first.score, first.category, first.reason, None)
        if self.judge is None:
            return self._d1_only(
                first.score, first.category, first.reason, self.judge_unavailable_reason
            )
        try:
            second = self.judge.score(text)
        except JUDGE_ERRORS as e:
            return self._d1_only(
                first.score, first.category, first.reason, f"judge failed: {type(e).__name__}"
            )
        lo, hi = self.band
        score = lo + (hi - lo) * second.score
        category = second.category if second.score >= 0.5 else NO_CATEGORY
        return self._answer("e2", "d2", score, category, second.reason, None)

    def _d1_only(
        self, score: float, category: str, reason: str, fallback: str | None
    ) -> CheckResponse:
        return self._answer("d1", "d1", score, category, reason, fallback)

    def _answer(
        self,
        mode: Literal["e2", "d1"],
        decided_by: Literal["d1", "d2"],
        score: float,
        category: str,
        reason: str,
        fallback: str | None,
    ) -> CheckResponse:
        threshold = self.config.thresholds.e2 if mode == "e2" else self.config.thresholds.d1
        return CheckResponse(
            allow=score < threshold,
            score=score,
            threshold=threshold,
            category=category,
            reason=reason,
            decided_by=decided_by,
            mode=mode,
            fallback=fallback,
            latency_ms=0.0,
        )
