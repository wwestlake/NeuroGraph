"""Pluggable text-embedding backends.

NeuroGraph never calls a generative LLM - "LLM-free" (matching LiteSemRAG's
own framing) means no API calls to a text-generation model. A local
sentence-embedding model is not a generative LLM; it's a small, fast,
deterministic encoder, which is exactly what keeps this whole pipeline
cheap enough to run inline in front of retrieval rather than as an
expensive preprocessing step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class EmbeddingBackend(ABC):
    """Anything that turns text into vectors. Implement this to swap in a
    different embedding model without touching nodes.py/graph.py at all."""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Returns an (N, D) float32 array, one row per input text."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError


class SentenceTransformerBackend(EmbeddingBackend):
    """Default backend: a small local sentence-transformers model.
    Downloads and caches the model on first use (standard huggingface_hub
    cache behavior) - after that, fully offline.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Imported lazily so importing neurograph itself doesn't force a
        # sentence-transformers import (and its own heavier dependency
        # chain) for callers who bring their own EmbeddingBackend.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension
