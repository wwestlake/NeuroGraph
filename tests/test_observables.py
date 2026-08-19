import numpy as np
import pytest

from neurograph.observables import token_set, jaccard_distance, cosine_distance


def test_token_set_lowercases_and_splits_on_words():
    assert token_set("Hello, World!") == {"hello", "world"}


def test_jaccard_distance_of_identical_sets_is_zero():
    s = {"a", "b", "c"}
    assert jaccard_distance(s, s) == 0.0


def test_jaccard_distance_of_disjoint_sets_is_one():
    assert jaccard_distance({"a"}, {"b"}) == 1.0


def test_jaccard_distance_of_empty_sets_is_zero_not_a_division_error():
    assert jaccard_distance(set(), set()) == 0.0


def test_cosine_distance_of_identical_vectors_is_zero():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-6)


def test_cosine_distance_of_orthogonal_vectors_is_one():
    assert cosine_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)
