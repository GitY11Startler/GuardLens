"""D3: embedding k-nearest-neighbours over the training set.

Both attacks and benign training examples are indexed, so benign hard negatives pull the
score down for look-alike benign inputs. Score = similarity-weighted share of attacks among
the k nearest training examples (cosine similarity; exact search with numpy).
"""

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from guardlens.data.schema import ATTACK, Example
from guardlens.detectors.base import NO_CATEGORY, Detector, DetectorOutput, Direction
from guardlens.embeddings import CachedEmbedder, Embedder

CHUNK = 512  # queries per similarity matrix chunk (512 x 24k floats ~ 50 MB)


class KNNDetector(Detector):
    name = "d3"

    def __init__(self, embedder: Embedder, k: int = 10) -> None:
        self.embedder = embedder
        self.k = k
        self._train: np.ndarray | None = None
        self._labels = np.zeros(0, dtype=np.int8)
        self._families: list[str | None] = []

    def fit(self, train: list[Example]) -> "KNNDetector":
        self._train = self.embedder.embed([e.text for e in train])
        self._labels = np.array([e.label for e in train], dtype=np.int8)
        self._families = [e.family for e in train]
        return self

    def score(self, text: str, direction: Direction = "input") -> DetectorOutput:
        return self.score_batch([text], direction)[0]

    @contextmanager
    def without_cache(self) -> Iterator[None]:
        cached = self.embedder
        if isinstance(cached, CachedEmbedder):
            self.embedder = cached.inner
        try:
            yield
        finally:
            self.embedder = cached

    def score_batch(self, texts: list[str], direction: Direction = "input") -> list[DetectorOutput]:
        if self._train is None:
            raise RuntimeError("call fit(train_examples) before scoring")
        queries = self.embedder.embed(texts)
        k = min(self.k, len(self._labels))
        outputs: list[DetectorOutput] = []
        for start in range(0, len(texts), CHUNK):
            sims = queries[start : start + CHUNK] @ self._train.T
            top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            for row, idx in enumerate(top):
                outputs.append(self._vote(sims[row, idx], idx))
        return outputs

    def _vote(self, sims: np.ndarray, idx: np.ndarray) -> DetectorOutput:
        weights = np.clip(sims, 0.0, None)
        is_attack = self._labels[idx] == ATTACK
        total = float(weights.sum())
        score = float(weights[is_attack].sum()) / total if total > 0 else 0.0
        families = Counter(self._families[i] for i in idx[is_attack])
        category = families.most_common(1)[0][0] if families else NO_CATEGORY
        return DetectorOutput(
            score=min(max(score, 0.0), 1.0),
            category=category or NO_CATEGORY,
            reason=f"{int(is_attack.sum())}/{len(idx)} nearest train examples are attacks "
            f"(max similarity {float(sims.max()):.2f})",
        )

    def config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "embedder": self.embedder.name,
            "prefix": getattr(self.embedder, "prefix", ""),
            "k": self.k,
            "n_train": len(self._labels),
        }
