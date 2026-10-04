"""Embedding generation for the semantic cache.

Owned by TanTan.

This module converts text queries into vector representations.
The semantic cache uses these vectors for semantic matching.

The SentenceTransformer dependency is imported lazily so that
unit tests can use injected fake embedding generators without
requiring the full ML stack to load.
"""

from typing import List, Optional


class EmbeddingGenerator:
    """Generate embeddings using SentenceTransformer."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        """Initialize the embedding model.

        The model is imported and loaded only when a real
        EmbeddingGenerator is created.
        """

        if not model_name:
            raise ValueError(
                "model_name cannot be empty"
            )

        self.model_name = model_name

        self.model = self._load_model(
            model_name
        )

    @staticmethod
    def _load_model(
        model_name: str,
    ):
        """Load SentenceTransformer lazily."""

        try:
            from sentence_transformers import (
                SentenceTransformer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "SentenceTransformer could not be imported. "
                "Check the sentence-transformers installation "
                "and its dependencies."
            ) from exc

        return SentenceTransformer(
            model_name
        )

    def encode(
        self,
        text: str,
    ) -> List[float]:
        """Convert one text query into an embedding vector."""

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        if not text.strip():
            raise ValueError(
                "text cannot be empty"
            )

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
            not isinstance(text, str)
            or not text.strip()
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
