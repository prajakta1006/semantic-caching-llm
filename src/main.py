"""Command-line demonstration for the semantic caching pipeline.

This entry point intentionally does not implement any teammate's
algorithm. Components are injected through the pipeline interfaces.

TanTan's semantic cache and Andi's evaluator can be connected here
once their modules are available.
"""

from src.config import config
from src.pipeline import SemanticCachingPipeline


def print_banner() -> None:
    """Print project configuration."""

    print("=" * 70)
    print(f"  {config.project_name}")
    print(
        f"  Version: {config.version} | "
        "Dhano Pipeline Integration"
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
    print("=" * 70)
    print()


def run_demo() -> None:
    """Run the pipeline without requiring TanTan/Andi modules yet."""

    print_banner()

    print(
        "[*] Initializing Dhano's LLM pipeline..."
    )

    pipeline = SemanticCachingPipeline()

    print(
        "[+] Pipeline initialized."
    )

    print(
        "[!] Semantic cache is not connected yet."
    )

    print(
        "[!] TanTan's CacheInterface implementation "
        "will be injected during integration."
    )

    print()

    queries = [
        "What is machine learning?",
        "What is semantic caching?",
        "What is an algorithm?",
    ]

    for index, query in enumerate(
        queries,
        start=1,
    ):

        print(
            f"--- Query {index}: '{query}' ---"
        )

        response = pipeline.process_query(
            query
        )

        print(
            f"  Source       : {response.source}"
        )

        print(
            f"  Decision     : "
            f"{response.decision.route}"
        )

        print(
            f"  Model        : "
            f"{response.metadata.get('model', 'N/A')}"
        )

        print(
            f"  Latency      : "
            f"{response.latency_ms:.2f} ms"
        )

        print(
            f"  Response     : "
            f"{response.response_text}"
        )

        print()


def main() -> None:
    """Application entry point."""

    run_demo()


if __name__ == "__main__":
    main()
