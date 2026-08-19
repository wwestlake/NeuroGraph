"""A deterministic, no-download embedding backend for tests - so the test
suite never needs network access or a real sentence-transformers model."""

from typing import Dict, List

import numpy as np

from neurograph.embeddings import EmbeddingBackend


class FakeEmbedder(EmbeddingBackend):
    """Maps each exact input string to a pre-registered vector. Any text
    not explicitly registered gets a stable, distinct pseudo-random
    vector derived from its hash, so unregistered inputs still behave
    consistently across calls within a test."""

    def __init__(self, dimension: int = 4):
        self._dimension = dimension
        self._known: Dict[str, np.ndarray] = {}

    def register(self, text: str, vector: List[float]) -> None:
        self._known[text] = np.array(vector, dtype=np.float32)

    def embed(self, texts: List[str]) -> np.ndarray:
        rows = []
        for text in texts:
            if text in self._known:
                rows.append(self._known[text])
            else:
                rng = np.random.default_rng(abs(hash(text)) % (2**32))
                rows.append(rng.random(self._dimension).astype(np.float32))
        return np.stack(rows)

    @property
    def dimension(self) -> int:
        return self._dimension
