from neurograph import NeuroGraph
from neurograph.nodes import static_percentile_threshold
from .fake_embedder import FakeEmbedder


def make_graph():
    embedder = FakeEmbedder(dimension=2)
    embedder.register("a1", [1.0, 0.0])
    embedder.register("a2", [0.98, 0.02])
    embedder.register("b1", [0.0, 1.0])

    graph = NeuroGraph(embedder=embedder, threshold_fn=static_percentile_threshold(0.9))
    return graph, embedder


def test_to_dict_from_dict_round_trip_preserves_nodes_and_edges():
    graph, embedder = make_graph()
    graph.ingest_document("doc1", ["a1", "b1"])

    data = graph.to_dict()
    restored = NeuroGraph.from_dict(data, embedder=embedder)

    assert len(restored.nodes) == len(graph.nodes)
    assert len(restored.edges) == len(graph.edges)
    assert restored.edges[0].weight == graph.edges[0].weight

    original_texts = sorted(n.member_texts[0] for n in graph.nodes)
    restored_texts = sorted(n.member_texts[0] for n in restored.nodes)
    assert original_texts == restored_texts


def test_restored_graph_keeps_growing_correctly_not_reset():
    graph, embedder = make_graph()
    graph.ingest_document("doc1", ["a1"])
    data = graph.to_dict()

    restored = NeuroGraph.from_dict(data, embedder=embedder, threshold_fn=static_percentile_threshold(0.9))
    restored.ingest_document("doc2", ["a2"])  # should join the "a" node, not spawn a new one

    assert len(restored.nodes) == 1
    assert len(restored.nodes[0]) == 2


def test_restored_graph_centroid_matches_original_within_floating_point_tolerance():
    graph, embedder = make_graph()
    graph.ingest_document("doc1", ["a1", "a2"])

    restored = NeuroGraph.from_dict(graph.to_dict(), embedder=embedder)
    original_node = graph.nodes[0]
    restored_node = restored.nodes[0]

    assert restored_node.centroid is not None
    for a, b in zip(original_node.centroid, restored_node.centroid):
        assert abs(a - b) < 1e-9
