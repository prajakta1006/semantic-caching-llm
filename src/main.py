"""Main entry point and CLI demonstration for Semantic Caching LLM Foundation."""

import sys
from src.config import config
from src.pipeline import SemanticCachingPipeline


def print_banner() -> None:
    print("=" * 70)
    print(f"  {config.project_name}")
    print(f"  Version: {config.version} | TanTan Foundation Setup")
    print("=" * 70)
    print(f"  Embedding Model Target: {config.embedding_model_name}")
    print(f"  Similarity Threshold : {config.similarity_threshold}")
    print(f"  Cache Directory      : {config.cache_dir}")
    print("=" * 70 + "\n")


def run_demo() -> None:
    print_banner()

    print("[*] Initializing Semantic Caching Pipeline Foundation...")
    pipeline = SemanticCachingPipeline()
    print("[+] Pipeline ready with modular extension points.\n")

    test_queries = [
        "What is machine learning?",
        "What is machine learning?",  # Demonstrates cache hit on exact match in foundation
        "What is semantic caching?",
    ]

    for idx, query in enumerate(test_queries, start=1):
        print(f"--- Query {idx}: '{query}' ---")
        response = pipeline.process_query(query)

        print(f"  [Source]          : {response.source.upper()}")
        print(f"  [Decision Route]  : {response.decision.route if response.decision else 'N/A'}")
        print(f"  [Similarity Score]: {response.similarity_score}")
        print(f"  [Latency]         : {response.latency_ms:.2f} ms")
        print(f"  [Response Text]   : {response.response_text}")
        print()

    # Show metrics
    metrics = pipeline.evaluator.get_metrics()
    print("=" * 70)
    print("  Pipeline Session Metrics (Foundation Mock):")
    print(f"  - Total Queries : {metrics.total_queries}")
    print(f"  - Cache Hits    : {metrics.cache_hits}")
    print(f"  - Cache Misses  : {metrics.cache_misses}")
    print(f"  - Hit Rate      : {metrics.hit_rate * 100:.1f}%")
    print(f"  - Avg Latency   : {metrics.avg_latency_ms:.2f} ms")
    print("=" * 70)


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
