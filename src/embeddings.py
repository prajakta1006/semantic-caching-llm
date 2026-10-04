"""Embedding generation for semantic cache.

Owned by TanTan.

This module converts text queries into vector representations.
The semantic cache uses these vectors for similarity matching.
"""

from typing import List

from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """Generate embeddings for semantic cache queries."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        """Initialize the embedding model.

        The model is loaded once and reused for all queries.
        """

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> List[float]:
        """Convert one text query into an embedding vector."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text cannot be empty")

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def encode_batch(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """Convert multiple texts into embedding vectors."""

        if not texts:
            return []

        if any(
            not isinstance(text, str) or not text.strip()
            for text in texts
        ):
            raise ValueError(
                "All texts must be non-empty strings"
            )

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.tolist()
