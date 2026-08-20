# NeuroGraph

A small, local, **LLM-free** engine for turning a stream of text into a
semantic graph: nodes are clusters of related text formed by embedding
similarity, edges are co-occurrence relationships built up as documents
are ingested. No generative model calls, no labeled training data
required to get started.

```python
from neurograph import NeuroGraph

graph = NeuroGraph()
graph.ingest_document("doc1", [
    "The James Webb Space Telescope observes in the infrared spectrum.",
    "Sourdough bread relies on wild yeast and lactobacillus cultures.",
])

for scored in graph.query("how do telescopes see distant objects"):
    print(scored.score, scored.node.member_texts)
```

## Why

Most "organize this text" problems don't need a full LLM call per item -
they need something fast, cheap, and local that can cluster related
content and rank it by relevance. NeuroGraph does that: embed, cluster
into nodes via similarity against a threshold, track which nodes
co-occur, and rank nodes against a query. Everything runs on-device with
a small local sentence-embedding model.

## Use cases

NeuroGraph is deliberately general-purpose - the graph engine itself has
no idea what it's being used for. Some concrete applications:

- **RAG pre-filtering** - cheaply narrow a large corpus down to the most
  relevant nodes before an expensive vector search or LLM call touches
  it (see `examples/rag_prefilter_example.py`).
- **Support ticket / inbox triage** - cluster incoming text and route by
  which existing cluster (or a brand-new one) it lands in.
- **Content moderation pre-screening** - flag text that doesn't cluster
  with anything already seen as worth a closer look.
- **Document/log noise filtering** - separate boilerplate/repeated
  content from genuinely novel material in a large corpus.

## Install

```bash
pip install -r requirements.txt
# or, for local development:
pip install -e ".[dev]"
```

Requires Python 3.9+. The default embedding backend
(`sentence-transformers`) downloads a small model on first use and
caches it locally after that.

## Running as a service

For non-Python callers (or any app that wants NeuroGraph as a shared
service rather than an embedded library), `neurograph/server.py` exposes
it over HTTP:

```bash
pip install -e ".[server]"
python run_server.py   # http://127.0.0.1:8000
```

The service is **stateless but resumable**: it holds no per-caller graph
in memory between requests. Every `/ingest` response includes the full
graph as JSON; hand that same JSON back in on the next call (as
`"graph": {...}`) to keep growing the same graph over time. Storage is
explicitly not this service's job - persist that JSON however your
application already does (a database, a file, whatever fits) and pass it
back in when you want to keep building on it.

```
POST /ingest  {"doc_id": "...", "chunks": [...], "graph": <optional prior graph>}
           -> {"graph": {...}, "new_node_ids": [...], "assigned_node_ids": [...]}

POST /query   {"query": "...", "graph": {...}, "top_k": 5}
           -> {"results": [{"node_id", "score", "member_texts"}, ...]}
```

The embedding model itself loads once at server startup and is shared
across every request - the one deliberate exception to "stateless,"
since reloading a multi-second model per call would defeat the point of
running this as a long-lived server.

## How it works

1. **`embeddings.py`** - pluggable text-embedding interface. The default
   backend is a small local `sentence-transformers` model; bring your
   own by implementing `EmbeddingBackend`.
2. **`nodes.py`** - node (cluster) creation. An incoming chunk joins the
   most similar existing node if that similarity clears a threshold,
   otherwise it seeds a new node. Unsupervised - no labeled categories.
3. **`graph.py`** - co-occurrence edges between nodes touched by the same
   document, plus query-time ranking that combines accumulated edge
   weight, a per-node importance level, and similarity to the query.
4. **`observables.py`** - the extension surface. Exposes the raw
   primitives (token sets, embedding distances, per-ingest deltas) that
   an external system could use to compute its own metrics on top of
   NeuroGraph's activity, without NeuroGraph needing to know what those
   metrics are.

## Roadmap

Named and scoped out of v1 deliberately, not silently missing:

- **`classify.py` / supervised classification** - assigning
  human-meaningful labels to formed clusters needs labeled training
  data, which the unsupervised core engine doesn't require. The
  interface is stubbed; the implementation is a real, separate piece of
  future work.
- **A kinematic/monitoring layer** - `observables.py` is intentionally
  just primitives, not a finished metrics system. Building an actual
  monitoring layer (drift detection, stability thresholds, etc.) on top
  belongs in a separate, consuming project - keeping that math out of
  this repo is what keeps NeuroGraph itself general-purpose.
- **Persistent graph backend** - deliberately not NeuroGraph's job.
  `to_dict()`/`from_dict()` give any caller a full, lossless JSON
  round-trip; where/how that gets stored (a database, a file, a VFS) is
  the consuming application's decision, not this repo's.
- **Node splitting** - `Node.dispersion_score()` already measures cluster
  tightness; using a falling dispersion score to trigger splitting an
  overgrown node into two is a natural follow-up, not implemented yet.

## License

MIT - see [LICENSE](LICENSE).
