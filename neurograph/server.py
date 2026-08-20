"""A stateless-but-resumable HTTP service around NeuroGraph.

"Stateless": the server holds no per-caller graph in memory between
requests - every request carries the full graph state as JSON (from a
prior response), so any instance can handle any request, and nothing is
lost if the server restarts. Storage is explicitly the caller's problem
(a database, a file, whatever their application already uses) - this
service's job ends at "text in, graph JSON out."

"Resumable": a caller keeps growing the same graph over time by handing
the previous /ingest response's `graph` back into the next call, rather
than every call starting from an empty graph.

The one thing that IS a shared, in-memory singleton across requests is
the embedding model (SentenceTransformerBackend) - loading it is
expensive (seconds), and it's stateless/read-only itself, so sharing it
across requests is safe and is the entire point of running this as a
long-lived server instead of one process per call.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .embeddings import SentenceTransformerBackend
from .graph import NeuroGraph

app = FastAPI(
    title="NeuroGraph",
    description="LLM-free semantic graph builder - text in, graph JSON out.",
)

# Loaded once at import time (server startup), shared by every request -
# see module docstring for why this is the one deliberate exception to
# "stateless."
_embedder = SentenceTransformerBackend()


class IngestRequest(BaseModel):
    doc_id: str
    chunks: List[str]
    graph: Optional[Dict[str, Any]] = None


class IngestResponse(BaseModel):
    graph: Dict[str, Any]
    new_node_ids: List[str]
    assigned_node_ids: List[str]


class QueryRequest(BaseModel):
    query: str
    graph: Dict[str, Any]
    top_k: int = 5


class ScoredNodeOut(BaseModel):
    node_id: str
    score: float
    member_texts: List[str]


class QueryResponse(BaseModel):
    results: List[ScoredNodeOut]


def _load_graph(graph_dict: Optional[Dict[str, Any]]) -> NeuroGraph:
    if graph_dict is None:
        return NeuroGraph(embedder=_embedder)
    return NeuroGraph.from_dict(graph_dict, embedder=_embedder)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    graph = _load_graph(req.graph)
    _touched_nodes, observables = graph.ingest_document(req.doc_id, req.chunks)
    return IngestResponse(
        graph=graph.to_dict(),
        new_node_ids=observables.new_node_ids,
        assigned_node_ids=observables.assigned_node_ids,
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    graph = _load_graph(req.graph)
    scored = graph.query(req.query, top_k=req.top_k)
    return QueryResponse(
        results=[
            ScoredNodeOut(node_id=s.node.id, score=s.score, member_texts=s.node.member_texts)
            for s in scored
        ]
    )
