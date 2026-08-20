"""The semantic graph itself: nodes (from nodes.py) connected by
co-occurrence edges, plus query-time scoring.

Two distinct weight concepts, matching the source research:
- Edge.weight: accumulated raw co-occurrence strength between two nodes,
  built up over ingestion (storage-time).
- The W(s) score computed in `NeuroGraph.query()`: a query-time ranking
  that combines accumulated edge weight, a per-node level factor, and
  similarity to the query itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .embeddings import EmbeddingBackend, SentenceTransformerBackend
from .nodes import Node, NodeStore, ThresholdFn, cosine_similarity
from .observables import IngestObservables


@dataclass
class Edge:
    node_a_id: str
    node_b_id: str
    weight: float = 0.0

    def touches(self, node_id: str) -> bool:
        return node_id in (self.node_a_id, self.node_b_id)

    def other(self, node_id: str) -> str:
        return self.node_b_id if node_id == self.node_a_id else self.node_a_id


@dataclass
class ScoredNode:
    node: Node
    score: float


class NeuroGraph:
    def __init__(
        self,
        embedder: Optional[EmbeddingBackend] = None,
        threshold_fn: Optional[ThresholdFn] = None,
    ):
        self.embedder = embedder or SentenceTransformerBackend()
        self.node_store = NodeStore(threshold_fn=threshold_fn)
        # Keyed by frozenset({a, b}) so (a, b) and (b, a) collapse to the
        # same edge regardless of insertion order.
        self._edges: Dict[frozenset, Edge] = {}
        # Per-node importance weight (LiteSemRAG's alpha_level). This repo
        # doesn't reproduce LiteSemRAG's full multi-level hierarchical
        # graph structure in v1 - defaults to a neutral 1.0 for every node;
        # override via `set_node_level()` if your application has its own
        # leveling/importance scheme.
        self._node_levels: Dict[str, float] = {}

    @property
    def nodes(self) -> List[Node]:
        return self.node_store.nodes

    @property
    def edges(self) -> List[Edge]:
        return list(self._edges.values())

    def set_node_level(self, node_id: str, level: float) -> None:
        self._node_levels[node_id] = level

    def _node_level(self, node_id: str) -> float:
        return self._node_levels.get(node_id, 1.0)

    def _edge_key(self, a_id: str, b_id: str) -> frozenset:
        return frozenset((a_id, b_id))

    def _strengthen_edge(self, a_id: str, b_id: str, amount: float = 1.0) -> Edge:
        key = self._edge_key(a_id, b_id)
        edge = self._edges.get(key)
        if edge is None:
            edge = Edge(node_a_id=a_id, node_b_id=b_id, weight=0.0)
            self._edges[key] = edge
        edge.weight += amount
        return edge

    def ingest_document(self, doc_id: str, chunks: List[str]) -> Tuple[List[Node], IngestObservables]:
        """Embeds and assigns every chunk to a node, then strengthens a
        co-occurrence edge between every distinct pair of nodes touched by
        this document. Returns the touched nodes plus an observables
        snapshot - the hook point for an external layer that wants to
        compute its own metrics from this ingest without NeuroGraph
        knowing what those metrics are (see observables.py)."""
        if not chunks:
            return [], IngestObservables(
                doc_id=doc_id, chunk_embeddings=[], assigned_node_ids=[],
                new_node_ids=[], touched_edges=[],
            )

        embeddings = self.embedder.embed(chunks)
        touched_nodes: List[Node] = []
        new_node_ids: List[str] = []
        assigned_node_ids: List[str] = []

        for embedding, text in zip(embeddings, chunks):
            result = self.node_store.assign(embedding, text)
            touched_nodes.append(result.node)
            assigned_node_ids.append(result.node.id)
            if result.is_new_node:
                new_node_ids.append(result.node.id)

        touched_edges: List[Tuple[str, str, float]] = []
        seen_pairs = set()
        for i in range(len(touched_nodes)):
            for j in range(i + 1, len(touched_nodes)):
                a, b = touched_nodes[i], touched_nodes[j]
                if a.id == b.id:
                    continue
                pair_key = self._edge_key(a.id, b.id)
                if pair_key in seen_pairs:
                    continue  # don't double-strengthen for repeated pairs within one document
                seen_pairs.add(pair_key)
                edge = self._strengthen_edge(a.id, b.id)
                touched_edges.append((a.id, b.id, edge.weight))

        observables = IngestObservables(
            doc_id=doc_id,
            chunk_embeddings=list(embeddings),
            assigned_node_ids=assigned_node_ids,
            new_node_ids=new_node_ids,
            touched_edges=touched_edges,
        )
        return touched_nodes, observables

    def query(self, query_text: str, top_k: int = 5) -> List[ScoredNode]:
        """W(s) = (sum of s's edge weights) * alpha_level(s) * cos(query, s).
        Matches the retrieval-time scoring formula from the source
        research - ranks nodes by a mix of how well-connected they are,
        their configured importance level, and direct similarity to the
        query."""
        if not self.nodes:
            return []

        query_embedding = self.embedder.embed([query_text])[0]

        scored: List[ScoredNode] = []
        for node in self.nodes:
            if node.centroid is None:
                continue
            edge_weight_sum = sum(e.weight for e in self._edges.values() if e.touches(node.id))
            similarity = cosine_similarity(query_embedding, node.centroid)
            score = edge_weight_sum * self._node_level(node.id) * similarity
            scored.append(ScoredNode(node=node, score=score))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def to_dict(self) -> Dict[str, Any]:
        """Full round-trip state - not just a display snapshot. A server
        handing this back to a caller and later receiving it again (see the
        API's stateless-but-resumable design) reconstructs the exact same
        graph, not an approximation of it."""
        return {
            "node_store": self.node_store.to_dict(),
            "edges": [
                {"node_a_id": e.node_a_id, "node_b_id": e.node_b_id, "weight": e.weight}
                for e in self.edges
            ],
            "node_levels": dict(self._node_levels),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        embedder: Optional[EmbeddingBackend] = None,
        threshold_fn: Optional[ThresholdFn] = None,
    ) -> "NeuroGraph":
        # embedder/threshold_fn aren't JSON-serializable (they're Python
        # objects/callables) - the caller supplies them, same as __init__.
        graph = cls(embedder=embedder, threshold_fn=threshold_fn)
        graph.node_store = NodeStore.from_dict(data["node_store"], threshold_fn=threshold_fn)
        for e in data["edges"]:
            key = graph._edge_key(e["node_a_id"], e["node_b_id"])
            graph._edges[key] = Edge(node_a_id=e["node_a_id"], node_b_id=e["node_b_id"], weight=e["weight"])
        graph._node_levels = dict(data.get("node_levels", {}))
        return graph
