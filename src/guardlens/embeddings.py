"""Sentence embeddings with an on-disk cache.

Cache key = sha256(cache version + model + prefix + text), so changing any of them gives
new keys instead of silently reusing stale vectors. One `.npz` file per model.
"""

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np

CACHE_VERSION = "emb-v1"


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """L2-normalized float32 vectors, shape (len(texts), dim)."""
        ...


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        prefix: str = "",
        batch_size: int = 64,
        max_seq_length: int = 512,
        device: str | None = None,
    ) -> None:
        self.name = model_name
        self.prefix = prefix  # e5 models expect "query: " in front of the text
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.device = device
        self._model = None

    def _load(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name, device=self.device)
            self._model.max_seq_length = self.max_seq_length
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._load().encode(
            [self.prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1000,
        )
        return np.asarray(vectors, dtype=np.float32)


class CachedEmbedder:
    """Wraps an embedder; only texts not seen before are sent to the model."""

    def __init__(self, inner: Embedder, cache_dir: Path) -> None:
        self.inner = inner
        self.name = inner.name
        self.prefix: str = getattr(inner, "prefix", "")
        slug = inner.name.replace("/", "__")
        self.path = cache_dir / f"{slug}.npz"
        self._index: dict[str, int] = {}
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        if self.path.exists():
            data = np.load(self.path)
            self._vectors = data["vectors"]
            self._index = {k: i for i, k in enumerate(data["keys"].tolist())}

    def key(self, text: str) -> str:
        raw = "\0".join((CACHE_VERSION, self.name, self.prefix, text))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def embed(self, texts: list[str]) -> np.ndarray:
        keys = [self.key(t) for t in texts]
        missing: dict[str, str] = {}
        for k, t in zip(keys, texts, strict=True):
            if k not in self._index:
                missing.setdefault(k, t)
        if missing:
            new = self.inner.embed(list(missing.values()))
            start = len(self._index)
            for offset, k in enumerate(missing):
                self._index[k] = start + offset
            self._vectors = new if self._vectors.size == 0 else np.vstack([self._vectors, new])
            self._save()
        return self._vectors[[self._index[k] for k in keys]]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(sorted(self._index, key=self._index.__getitem__))
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")
        np.savez(tmp, keys=keys, vectors=self._vectors)
        tmp.replace(self.path)
