"""Demo runner for the semantic caching project.

This file demonstrates the integrated system:

    User Query
        ↓
    Dhano Pipeline
        ↓
    TanTan Semantic Cache
        ↓
    HIT  → cached response
    MISS → Mock LLM → store response
"""

from src.config import config
from src.pipeline import SemanticCachingPipeline
from src.semantic_cache import SemanticCache


def print_response(
    query: str,
    response,
) -> None:
    """Print one pipeline response."""

    print()
    print(f"--- Query: '{query}' ---")
    print(f"  Source       : {response.source}")

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

    print(f"  Model        : {model}")
    print(
        f"  Latency      : "
        f"{response.latency_ms:.2f} ms"
    )

    print(
        f"  Response     : "
        f"{response.response_text}"
    )


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
    print("[*] Initializing TanTan's semantic cache...")

    try:
        cache = SemanticCache(
            embedding_model_name=(
                config.embedding_model_name
            )
        )

        print(
            "[+] TanTan semantic cache initialized."
        )

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
            "[!] This is usually caused by the "
            "Windows Application Control policy "
            "blocking the scikit-learn DLL."
        )
        print()
        print(
            "[!] The unit tests and integration "
            "tests still work with fake embeddings."
        )

        return

    print()
    print("[*] Initializing Dhano's LLM pipeline...")

    pipeline = SemanticCachingPipeline(
        cache=cache,
    )

    print("[+] Pipeline initialized.")
    print("[+] TanTan cache connected.")
    print("[+] Dhano pipeline connected.")

    print()
    print("=" * 70)
    print("  SEMANTIC CACHE DEMONSTRATION")
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
        f"  Final Cache Size: {cache.size()}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
