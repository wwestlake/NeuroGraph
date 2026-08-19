import pytest

from neurograph import NeuroGraph
from neurograph.nodes import static_percentile_threshold
from .fake_embedder import FakeEmbedder


def make_graph() -> tuple[NeuroGraph, FakeEmbedder]:
    embedder = FakeEmbedder(dimension=2)
    # Two clearly-separated topic clusters, plus one query vector aligned
    # with topic A.
    embedder.register("a1", [1.0, 0.0])
    embedder.register("a2", [0.98, 0.02])
    embedder.register("b1", [0.0, 1.0])
    embedder.register("b2", [0.02, 0.98])
    embedder.register("query_a", [1.0, 0.0])

    graph = NeuroGraph(embedder=embedder, threshold_fn=static_percentile_threshold(0.9))
    return graph, embedder


def test_ingest_document_creates_nodes_for_distinct_topics():
    graph, _ = make_graph()
    nodes, obs = graph.ingest_document("doc1", ["a1", "b1"])
    assert len(nodes) == 2
    assert obs.anomaly_count == 2  # both are new - nothing existed yet
    assert len(graph.nodes) == 2


def test_ingest_document_reuses_nodes_for_similar_chunks():
    graph, _ = make_graph()
    graph.ingest_document("doc1", ["a1"])
    nodes, obs = graph.ingest_document("doc2", ["a2"])
    assert obs.anomaly_count == 0  # joined the existing "a" node
    assert len(graph.nodes) == 1
    assert len(graph.nodes[0]) == 2


def test_co_occurring_chunks_create_an_edge():
    graph, _ = make_graph()
    graph.ingest_document("doc1", ["a1", "b1"])
    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert edge.weight == 1.0


def test_repeated_co_occurrence_strengthens_the_same_edge_not_duplicates():
    graph, _ = make_graph()
    graph.ingest_document("doc1", ["a1", "b1"])
    graph.ingest_document("doc2", ["a2", "b2"])
    assert len(graph.edges) == 1  # still one edge between the "a" and "b" nodes
    assert graph.edges[0].weight == 2.0


def test_query_ranks_the_matching_topic_highest():
    graph, _ = make_graph()
    graph.ingest_document("doc1", ["a1", "b1"])
    results = graph.query("query_a", top_k=2)
    assert len(results) == 2
    assert results[0].node.member_texts[0] == "a1"
    assert results[0].score >= results[1].score


def test_node_level_scales_query_score():
    graph, _ = make_graph()
    graph.ingest_document("doc1", ["a1", "b1"])
    baseline = graph.query("query_a", top_k=1)[0].score

    a_node_id = next(n.id for n in graph.nodes if n.member_texts[0] == "a1")
    graph.set_node_level(a_node_id, 5.0)
    boosted = graph.query("query_a", top_k=1)[0].score

    assert boosted == pytest.approx(baseline * 5.0)
