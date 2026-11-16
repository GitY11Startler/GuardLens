import hashlib

import numpy as np


class FakeEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words get similar vectors.

    No model download, no GPU. Counts how many texts it was asked to embed.
    """

    name = "fake-bow"
    prefix = ""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.calls = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += len(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                out[row, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)
