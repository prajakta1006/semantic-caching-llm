"""Tests for cosine similarity."""

import pytest

from src.similarity import cosine_similarity


def test_identical_vectors_have_similarity_one():

    score = cosine_similarity(
        [1.0, 0.0],
        [1.0, 0.0],
    )

    assert score == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero():

    score = cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    )

    assert score == pytest.approx(0.0)


def test_opposite_vectors_have_similarity_minus_one():

    score = cosine_similarity(
        [1.0, 0.0],
        [-1.0, 0.0],
    )

    assert score == pytest.approx(-1.0)


def test_dimension_mismatch_is_rejected():

    with pytest.raises(ValueError):

        cosine_similarity(
            [1.0, 0.0],
            [1.0],
        )
