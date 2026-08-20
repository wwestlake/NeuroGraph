"""Node (cluster) creation and assignment.

Implements the mechanism described in the source research: an incoming
text chunk is assigned to whichever existing node its embedding is most
similar to, unless that similarity falls below an anomaly threshold, in
which case it becomes the seed of a new node. This is deliberately
unsupervised - no labeled training data, no fixed category list - which
is what LiteSemRAG's own described approach actually is (see README for
the distinction from a trained classifier, which is a separate, deferred
piece - see classify.py).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class Node:
    """A cluster of semantically related text chunks."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    centroid: Optional[np.ndarray] = None
    member_embeddings: List[np.ndarray] = field(default_factory=list)
    member_texts: List[str] = field(default_factory=list)

    def add_member(self, embedding: np.ndarray, text: str) -> None:
        self.member_embeddings.append(embedding)
        self.member_texts.append(text)
        # Running mean centroid - cheap to update incrementally rather than
        # recomputing over all members on every insert.
        if self.centroid is None:
            self.centroid = embedding.copy()
        else:
            n = len(self.member_embeddings)
            self.centroid = self.centroid + (embedding - self.centroid) / n

    def dispersion_score(self) -> float:
        """S-mean: the mean cosine similarity of every member embedding to
        this node's own centroid - how tightly clustered the node actually
        is. Low dispersion means a tight, coherent node; a node whose
        dispersion keeps falling as it grows is a candidate to split in a
        future version (not implemented here - see README's Roadmap)."""
        if not self.member_embeddings or self.centroid is None:
            return 0.0
        sims = [cosine_similarity(e, self.centroid) for e in self.member_embeddings]
        return float(np.mean(sims))

    def __len__(self) -> int:
        return len(self.member_embeddings)

    def to_dict(self) -> Dict[str, Any]:
        """Full round-trip fidelity, not just a snapshot - member_embeddings
        is included (not just the centroid) so a graph rebuilt via
        from_dict() keeps computing a correct running-mean centroid as more
        chunks are added later, not just displaying a frozen one. Larger
        payload than a centroid-only summary would be; a compact/summary
        mode is a reasonable future addition, not done here."""
        return {
            "id": self.id,
            "centroid": self.centroid.tolist() if self.centroid is not None else None,
            "member_embeddings": [e.tolist() for e in self.member_embeddings],
            "member_texts": list(self.member_texts),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        node = cls(id=data["id"])
        node.member_embeddings = [np.array(e, dtype=float) for e in data["member_embeddings"]]
        node.member_texts = list(data["member_texts"])
        centroid = data.get("centroid")
        node.centroid = np.array(centroid, dtype=float) if centroid is not None else None
        return node


# A threshold function receives the current NodeStore (so it can look at
# node count, overall state, etc.) and the max similarity found for the
# incoming chunk, and returns True if that similarity is high enough to
# join the matching node rather than spawn a new one. This is the exact
# seam an external ISD-style layer hooks into: replace this with a
# function of accumulated reference drift instead of a fixed constant,
# without NodeStore/Graph needing to know what "drift" means.
ThresholdFn = Callable[["NodeStore", float], bool]


def static_percentile_threshold(min_similarity: float = 0.6) -> ThresholdFn:
    """The default, simplest threshold: join the closest node if cosine
    similarity to its centroid is at least `min_similarity`, else spawn a
    new node. Not literally a percentile over historical scores (that
    needs a running distribution, a reasonable v2 refinement) - named for
    the mechanism it approximates from the source research (LiteSemRAG's
    T_anomaly), not because it computes an actual percentile itself.
    """

    def _threshold(_store: "NodeStore", max_similarity: float) -> bool:
        return max_similarity >= min_similarity

    return _threshold


class NodeStore:
    def __init__(self, threshold_fn: Optional[ThresholdFn] = None):
        self.nodes: List[Node] = []
        self.threshold_fn = threshold_fn or static_percentile_threshold()

    def assign(self, embedding: np.ndarray, text: str) -> "AssignResult":
        best_node: Optional[Node] = None
        best_similarity = -1.0
        for node in self.nodes:
            if node.centroid is None:
                continue
            sim = cosine_similarity(embedding, node.centroid)
            if sim > best_similarity:
                best_similarity = sim
                best_node = node

        if best_node is not None and self.threshold_fn(self, best_similarity):
            best_node.add_member(embedding, text)
            return AssignResult(node=best_node, is_new_node=False, similarity=best_similarity)

        new_node = Node()
        new_node.add_member(embedding, text)
        self.nodes.append(new_node)
        return AssignResult(node=new_node, is_new_node=True, similarity=best_similarity)

    def get(self, node_id: str) -> Optional[Node]:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"nodes": [n.to_dict() for n in self.nodes]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any], threshold_fn: Optional[ThresholdFn] = None) -> "NodeStore":
        # threshold_fn is a Python callable, not JSON-serializable - the
        # caller supplies it (defaults the same way __init__ does), same
        # split NeuroGraph.from_dict uses for its embedder.
        store = cls(threshold_fn=threshold_fn)
        store.nodes = [Node.from_dict(n) for n in data["nodes"]]
        return store


@dataclass
class AssignResult:
    node: Node
    is_new_node: bool
    similarity: float  # best similarity found against existing nodes (before assignment)
