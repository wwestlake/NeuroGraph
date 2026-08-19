import numpy as np
import pytest

from neurograph.nodes import NodeStore, static_percentile_threshold, cosine_similarity


def test_first_chunk_always_creates_a_new_node():
    store = NodeStore()
    result = store.assign(np.array([1.0, 0.0]), "first chunk")
    assert result.is_new_node
    assert len(store.nodes) == 1


def test_similar_chunk_joins_existing_node():
    store = NodeStore(threshold_fn=static_percentile_threshold(min_similarity=0.9))
    store.assign(np.array([1.0, 0.0]), "topic A chunk 1")
    result = store.assign(np.array([0.99, 0.01]), "topic A chunk 2")
    assert not result.is_new_node
    assert len(store.nodes) == 1
    assert len(store.nodes[0]) == 2


def test_dissimilar_chunk_spawns_new_node():
    store = NodeStore(threshold_fn=static_percentile_threshold(min_similarity=0.9))
    store.assign(np.array([1.0, 0.0]), "topic A chunk")
    result = store.assign(np.array([0.0, 1.0]), "completely different topic")
    assert result.is_new_node
    assert len(store.nodes) == 2


def test_centroid_updates_as_running_mean():
    store = NodeStore(threshold_fn=static_percentile_threshold(min_similarity=0.5))
    store.assign(np.array([1.0, 0.0]), "a")
    store.assign(np.array([1.0, 0.2]), "b")
    store.assign(np.array([1.0, -0.2]), "c")
    node = store.nodes[0]
    assert len(node) == 3
    assert node.centroid is not None
    np.testing.assert_allclose(node.centroid, np.array([1.0, 0.0]), atol=1e-9)


def test_dispersion_score_is_high_for_a_tight_cluster():
    store = NodeStore(threshold_fn=static_percentile_threshold(min_similarity=0.99))
    store.assign(np.array([1.0, 0.0]), "a")
    store.assign(np.array([0.99, 0.01]), "b")
    node = store.nodes[0]
    assert node.dispersion_score() > 0.99


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_similarity_handles_zero_vector_without_dividing_by_zero():
    assert cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0
