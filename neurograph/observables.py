"""Primitives an external kinematic/monitoring layer (e.g. an
Information-Space-Dynamics-style system) needs to compute its own
metrics from NeuroGraph's activity - WITHOUT NeuroGraph importing or
knowing anything about what those metrics mean. This module is the
entire coupling surface: token-set operations, embedding distances, and
a structured snapshot of what happened during one ingest call.

Keeping this boundary sharp is deliberate (see README/Roadmap): the
kinematic math (curvature, drift, stability identities, whatever a
consuming system wants to build) belongs in that consuming system, not
here. NeuroGraph stays honestly general-purpose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set, Tuple

import numpy as np

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def token_set(text: str) -> Set[str]:
    """Lowercased word tokens as a set - the raw material for Jaccard-style
    structural-separation metrics."""
    return set(_WORD_RE.findall(text.lower()))


def jaccard_distance(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return 1.0 - (len(a & b) / len(union))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)


@dataclass
class IngestObservables:
    """Everything NeuroGraph knows about a single ingest_document() call,
    exposed as plain data for an external layer to consume."""

    doc_id: str
    chunk_embeddings: List[np.ndarray]
    assigned_node_ids: List[str]
    new_node_ids: List[str]
    touched_edges: List[Tuple[str, str, float]] = field(default_factory=list)

    @property
    def anomaly_count(self) -> int:
        """How many chunks in this ingest spawned a brand-new node rather
        than joining an existing one - a raw signal an external layer
        might treat as evidence of drift, novelty, or a topic shift."""
        return len(self.new_node_ids)
