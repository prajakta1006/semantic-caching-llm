"""Demo runner for the semantic caching project.

Demonstrates the integrated TanTan + Dhano workflow:

    User Query
        ↓
    Dhano Pipeline
        ↓
    TanTan Semantic Cache
        ↓
    ┌───────────────┐
    │               │
   HIT             MISS
    │               │
    ↓               ↓
 Cached            Mock LLM
 Response            │
                     ↓
                Store in Cache
                     │
                     ↓
                  Response

The real SentenceTransformer model is attempted first.

If the local Windows environment blocks the required native
dependency, a deterministic demo embedding generator is used
only for local demonstration/testing.
"""

from src.config import config
from src.pipeline import SemanticCachingPipeline
from src.semantic_cache import SemanticCache


class DemoEmbeddingGenerator:
    """Deterministic fallback embeddings for local demonstration."""

    def encode(self, text: str):
        """Return deterministic embeddings for demo queries."""

        normalized = text.strip().lower()

        embeddings = {
            "what is machine learning?": [
                1.0,
                0.0,
            ],
            "explain machine learning": [
                0.99,
                0.01,
            ],
            "what is semantic caching?": [
                0.0,
                1.0,
            ],
            "explain semantic caching": [
                0.01,
                0.99,
            ],
            "what is an algorithm?": [
                0.7071,
                0.7071,
            ],
        }

        return embeddings.get(
            normalized,
            [0.0, 0.0],
        )


def print_response(query: str, response) -> None:
    """Print one pipeline response."""

    print()
    print(f"--- Query: '{query}' ---")

    print(
        f"  Source       : "
        f"{response.source}"
    )

    if response.decision is not None:
        print(
            f"  Decision     : "
            f"{response.decision.route}"
        )

    if response.similarity_score is not None:
        print(
            f"  Similarity   : "
            f"{response.similarity_score:.4f}"
        )

    model = response.metadata.get(
        "model",
        "N/A",
    )

    print(
        f"  Model        : "
        f"{model}"
    )

    print(
        f"  Latency      : "
        f"{response.latency_ms:.2f} ms"
    )

    print(
        f"  Response     : "
        f"{response.response_text}"
    )


def create_cache():
    """Create TanTan's semantic cache.

    Try the real SentenceTransformer model first.

    If the local environment blocks the ML dependency,
    use deterministic demo embeddings so the complete
    pipeline can still be demonstrated.
    """

    print(
        "[*] Initializing TanTan's semantic cache..."
    )

    try:
        cache = SemanticCache(
            embedding_model_name=(
                config.embedding_model_name
            )
        )

        print(
            "[+] Real SentenceTransformer "
            "embedding model loaded."
        )

        return cache, "real"

    except Exception as exc:
        print()
        print(
            "[!] Real embedding model could not "
            "be initialized."
        )

        print(
            f"[!] Reason: {exc}"
        )

        print()
        print(
            "[*] Using deterministic demo "
            "embeddings instead."
        )

        cache = SemanticCache(
            embedding_generator=(
                DemoEmbeddingGenerator()
            )
        )

        print(
            "[+] Demo embedding generator loaded."
        )

        return cache, "demo"


def main() -> None:
    """Run the integrated semantic caching demo."""

    print("=" * 70)

    print(
        "  Semantic Caching for LLM - "
        "Cost and Latency Optimization"
    )

    print(
        f"  Version: {config.version} | "
        "TanTan + Dhano Integration"
    )

    print("=" * 70)

    print(
        f"  Embedding Model      : "
        f"{config.embedding_model_name}"
    )

    print(
        f"  Similarity Threshold : "
        f"{config.similarity_threshold}"
    )

    print(
        f"  LLM Provider         : "
        f"{config.llm_provider}"
    )

    print(
        f"  Default LLM Model    : "
        f"{config.default_llm_model}"
    )

    print()

    cache, embedding_mode = create_cache()

    print()

    print(
        "[*] Initializing Dhano's LLM pipeline..."
    )

    pipeline = SemanticCachingPipeline(
        cache=cache,
    )

    print(
        "[+] Pipeline initialized."
    )

    print(
        "[+] TanTan cache connected."
    )

    print(
        "[+] Dhano pipeline connected."
    )

    print()

    print("=" * 70)
    print(
        "  SEMANTIC CACHE DEMONSTRATION"
    )
    print("=" * 70)

    if embedding_mode == "demo":
        print(
            "  Embedding mode: DEMO FALLBACK"
        )
        print(
            "  Reason: local ML dependency is "
            "blocked by Windows."
        )
    else:
        print(
            "  Embedding mode: REAL "
            "SentenceTransformer"
        )

    print("=" * 70)

    queries = [
        "What is machine learning?",
        "What is semantic caching?",
        "What is an algorithm?",
        "Explain machine learning",
    ]

    for query in queries:
        response = pipeline.process_query(
            query
        )

        print_response(
            query,
            response,
        )

    print()

    print("=" * 70)

    print(
        f"  Final Cache Size: "
        f"{cache.size()}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
