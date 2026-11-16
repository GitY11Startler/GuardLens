"""Shared detector interface: every detector is scored by the same runner the same way."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

from pydantic import BaseModel, Field

Direction = Literal["input", "output"]
NO_CATEGORY = "none"


class DetectorOutput(BaseModel):
    score: float = Field(ge=0.0, le=1.0)  # higher = more likely an attack
    category: str  # predicted attack category, or "none"
    reason: str  # short human-readable explanation (never quotes training data)


class Detector(ABC):
    name: str

    @abstractmethod
    def score(self, text: str, direction: Direction = "input") -> DetectorOutput: ...

    def score_batch(self, texts: list[str], direction: Direction = "input") -> list[DetectorOutput]:
        """Override when a detector is faster in batches (e.g. embeddings on the GPU)."""
        return [self.score(t, direction) for t in texts]

    @contextmanager
    def without_cache(self) -> Iterator[None]:
        """Turn off any result/embedding cache inside the block (used to measure latency).
        Override in detectors that cache."""
        yield

    def config(self) -> dict[str, Any]:
        """Settings that define this detector; saved with every run."""
        return {"name": self.name}
