"""Similarity calculations for semantic cache.

Owned by TanTan.
"""

from math import sqrt
from typing import List


def cosine_similarity(
    vector_a: List[float],
    vector_b: List[float],
) -> float:
    """Calculate cosine similarity between two vectors.

    Returns a value approximately in the range [-1, 1].
    For normalized sentence-transformer embeddings, this
    effectively represents semantic similarity.
    """

    if not vector_a or not vector_b:
        raise ValueError(
            "Vectors cannot be empty"
        )

    if len(vector_a) != len(vector_b):
        raise ValueError(
            "Vectors must have the same dimension"
        )

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    return dot_product / (
        magnitude_a * magnitude_b
    )
