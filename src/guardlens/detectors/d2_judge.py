"""D2: LLM-as-judge. A Claude model scores each text against the rubric in d2_prompt.py.

Per text: disk cache -> spending-cap check -> one API call (structured JSON answer, effort
from the config) -> log tokens + $ -> cache the answer. Score = attack_probability / 100.
If the model refuses to classify (stop_reason "refusal"), the text is flagged: score 1.0,
category "judge_refusal" (a guardrail that won't look would block).
"""

import hashlib
import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from guardlens.detectors.base import NO_CATEGORY, Detector, DetectorOutput, Direction
from guardlens.detectors.d2_prompt import (
    ANSWER_SCHEMA,
    CATEGORIES,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    user_message,
)
from guardlens.judge_cost import Prices, SpendLedger, Usage, call_cost

REFUSAL_CATEGORY = "judge_refusal"
CHARS_PER_TOKEN_WORST_CASE = 2.0  # deliberately pessimistic for the pre-call cap check


class JudgeError(RuntimeError):
    """The model answered, but not with a usable classification."""


class JudgeDetector(Detector):
    name = "d2"

    def __init__(
        self,
        client: Any,  # anthropic.Anthropic, or a fake with the same .messages.create in tests
        model: str,
        effort: str,
        max_tokens: int,
        prices: Prices,
        ledger: SpendLedger,
        cache_dir: Path,
        concurrency: int = 4,
    ) -> None:
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.prices = prices
        self.ledger = ledger
        self.cache_dir = cache_dir
        self.concurrency = concurrency
        self.use_cache = True
        self.label = f"{PROMPT_VERSION}|effort={effort}"

    def key(self, text: str) -> str:
        raw = "\0".join((self.model, PROMPT_VERSION, self.effort, str(self.max_tokens), text))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def score(self, text: str, direction: Direction = "input") -> DetectorOutput:
        key = self.key(text)
        path = self._cache_path(key)
        if self.use_cache and path.exists():
            return DetectorOutput.model_validate_json(path.read_text("utf-8"))
        out = self._call(text)
        if self.use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(out.model_dump_json(), "utf-8")
        return out

    def score_batch(self, texts: list[str], direction: Direction = "input") -> list[DetectorOutput]:
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(self.score, texts))  # map keeps the input order

    @contextmanager
    def without_cache(self) -> Iterator[None]:
        self.use_cache = False
        try:
            yield
        finally:
            self.use_cache = True

    def _call(self, text: str) -> DetectorOutput:
        prompt = user_message(text)
        worst_in = (len(SYSTEM_PROMPT) + len(prompt)) / CHARS_PER_TOKEN_WORST_CASE
        self.ledger.check(
            worst_in * self.prices.input_per_mtok / 1e6
            + self.max_tokens * self.prices.output_per_mtok / 1e6
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
            },
        )
        u = response.usage
        usage = Usage(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", None) or 0,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", None) or 0,
        )
        self.ledger.record(self.model, self.label, usage, call_cost(usage, self.prices))
        return parse_response(response)

    def cost_summary(self) -> dict[str, float]:
        """Mean cost and tokens per API call for this model + prompt + effort, from the ledger
        (cached answers cost nothing, so this is the price of judging new text)."""
        rows = [
            r for r in self.ledger.rows() if r["model"] == self.model and r["label"] == self.label
        ]
        n = max(len(rows), 1)
        total = sum(r["usd"] for r in rows)
        return {
            "api_calls": len(rows),
            "total_usd": total,
            "cost_per_1k_usd": 1000 * total / n,
            "mean_input_tokens": sum(r["input_tokens"] for r in rows) / n,
            "mean_output_tokens": sum(r["output_tokens"] for r in rows) / n,
        }

    def config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "effort": self.effort,
            "max_tokens": self.max_tokens,
            "prompt_version": PROMPT_VERSION,
        }


def parse_response(response: Any) -> DetectorOutput:
    if response.stop_reason == "refusal":
        return DetectorOutput(
            score=1.0,
            category=REFUSAL_CATEGORY,
            reason="judge declined to classify this text; flagged",
        )
    if response.stop_reason == "max_tokens":
        raise JudgeError("answer cut off at max_tokens; raise d2.max_tokens")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise JudgeError(f"no text block in the answer (stop_reason {response.stop_reason})")
    try:
        answer = json.loads(text)
        probability = int(answer["attack_probability"])
        category = str(answer["category"])
        reason = str(answer["reason"])
    except (ValueError, KeyError, TypeError) as e:
        raise JudgeError(f"answer is not the expected JSON: {e}") from e
    if category not in CATEGORIES:
        raise JudgeError(f"unknown category {category!r}")
    return DetectorOutput(
        score=min(max(probability, 0), 100) / 100,
        category=NO_CATEGORY if category == "benign" else category,
        reason=reason[:300],
    )
