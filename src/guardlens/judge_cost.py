"""Token + cost logging for paid API calls, and the project-wide spending cap.

Every API call appends one JSON line to the ledger (`data/cache/judge/spend.jsonl`):
time, model, a label (the judge uses prompt version + effort), tokens and $ cost.
The running total is checked against `spending_cap_usd` (configs/detectors.yaml) *before*
each call.

With several calls in flight at once, each one passes the check before the others have
recorded their cost, so the cap can be overshot by at most (concurrency x one call), i.e.
well under a cent for the judge.
"""

import json
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class SpendingCapError(RuntimeError):
    """Raised before a call that could push total spending over the cap."""


class Prices(BaseModel):
    input_per_mtok: float  # $ per 1M uncached input tokens
    output_per_mtok: float  # $ per 1M output tokens (thinking tokens included)
    cache_write_multiplier: float = 1.25  # x input price (5-minute cache writes)
    cache_read_multiplier: float = 0.1  # x input price


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


def call_cost(usage: Usage, prices: Prices) -> float:
    per_in = prices.input_per_mtok / 1e6
    return (
        usage.input_tokens * per_in
        + usage.cache_creation_input_tokens * per_in * prices.cache_write_multiplier
        + usage.cache_read_input_tokens * per_in * prices.cache_read_multiplier
        + usage.output_tokens * prices.output_per_mtok / 1e6
    )


class SpendLedger:
    def __init__(self, path: Path, cap_usd: float) -> None:
        self.path = path
        self.cap_usd = cap_usd
        self._lock = threading.Lock()
        self._total = 0.0
        if path.exists():
            with path.open(encoding="utf-8") as f:
                self._total = sum(json.loads(line)["usd"] for line in f if line.strip())

    @property
    def total_usd(self) -> float:
        with self._lock:
            return self._total

    def check(self, worst_case_usd: float) -> None:
        with self._lock:
            if self._total + worst_case_usd > self.cap_usd:
                raise SpendingCapError(
                    f"spending cap reached: ${self._total:.4f} spent of ${self.cap_usd:.2f} "
                    f"(next call could cost up to ${worst_case_usd:.4f}). Raise "
                    "spending_cap_usd in configs/detectors.yaml or switch backend."
                )

    def record(self, model: str, label: str, usage: Usage, usd: float) -> None:
        row = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": model,
            "label": label,
            **usage.model_dump(),
            "usd": usd,
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            self._total += usd

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
