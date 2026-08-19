"""Supervised label/category assignment on top of formed clusters.

NOT IMPLEMENTED in v1. The core graph engine (nodes.py/graph.py) is
deliberately unsupervised - no labeled data, no fixed category list,
matching LiteSemRAG's actual described mechanism. A supervised layer that
assigns human-meaningful labels to clusters (e.g. "support ticket
priority", "moderation category") is a real, useful, different piece of
work that needs training data the unsupervised engine doesn't - see the
README's Roadmap section. This module exists as a named placeholder so
that gap is documented, not silently missing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from .nodes import Node


class NodeClassifier(ABC):
    """Interface a future trained classifier would implement: given a
    Node (a formed cluster), return a label. Not implemented in v1."""

    @abstractmethod
    def classify(self, node: Node) -> str:
        raise NotImplementedError(
            "NodeClassifier has no v1 implementation - see neurograph/classify.py's "
            "module docstring and the README's Roadmap section."
        )

    @abstractmethod
    def train(self, labeled_nodes: List[tuple]) -> None:
        """labeled_nodes: List[Tuple[Node, str]]. Not implemented in v1."""
        raise NotImplementedError(
            "NodeClassifier has no v1 implementation - see neurograph/classify.py's "
            "module docstring and the README's Roadmap section."
        )
