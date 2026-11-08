"""Interactive demo runner for the semantic caching project.

Integrated components:
- TanTan Semantic Cache
- Dhano LLM Pipeline
- Praj Cache Optimizer
- Andi Evaluation Tracker
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.cache_optimizer import CacheOptimizer
from src.config import config
from src.evaluator import EvaluationTracker
from src.pipeline import SemanticCachingPipeline
from src.semantic_cache import SemanticCache


# ----------------------------------------------------------------------
# Display configuration
# ----------------------------------------------------------------------

def get_display_model():
    """Return the configured model corresponding to the active provider."""

    if config.llm_provider.lower() == "mock":
        return "mock-llm-v1"

    return config.default_llm_model


# ----------------------------------------------------------------------
# Fallback embedding generator
# ----------------------------------------------------------------------

class DemoEmbeddingGenerator:
    """Deterministic fallback embeddings for local demonstration."""

    def encode(self, text: str):
        """Return deterministic embeddings for known demo queries."""

        normalized = text.strip().lower()

        embeddings = {
            "what is machine learning?": [1.0, 0.0],
            "define machine learning": [0.99, 0.01],
            "what is semantic caching?": [0.0, 1.0],
            "define semantic caching": [0.01, 0.99],
            "what is an algorithm?": [0.7071, 0.7071],
            "define an algorithm": [0.70, 0.71],
        }

        return embeddings.get(normalized, [0.0, 0.0])


# ----------------------------------------------------------------------
# Cache creation
# ----------------------------------------------------------------------

def create_cache():
    """Create the semantic cache and wrap it with the optimizer."""

    print("[*] Initializing TanTan's semantic cache...")

    try:
        semantic_cache = SemanticCache(
            embedding_model_name=config.embedding_model_name
        )

        print("[+] Real SentenceTransformer embedding model loaded.")
        embedding_mode = "REAL"

    except Exception as exc:
        print()
        print("[!] Real embedding model could not be initialized.")
        print(f"[!] Reason: {exc}")
        print()
        print("[*] Using deterministic demo embeddings instead.")

        semantic_cache = SemanticCache(
            embedding_generator=DemoEmbeddingGenerator()
        )

        print("[+] Demo embedding generator loaded.")
        embedding_mode = "DEMO"

    optimizer = CacheOptimizer(
        cache=semantic_cache,
        max_size=3,
        ttl_seconds=None,
    )

    print("[+] Praj cache optimizer connected.")
    print("[+] Cache capacity : 3 entries")
    print("[+] Eviction policy : LRU")
    print("[+] TTL             : Disabled")

    return optimizer, embedding_mode


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------

def print_header():
    """Print the main application header."""

    print()
    print("=" * 76)
    print("        SEMANTIC CACHE - COST & LATENCY OPTIMIZATION")
    print("=" * 76)

    print(f" Embedding Model : {config.embedding_model_name}")
    print(f" Similarity      : {config.similarity_threshold}")
    print(f" LLM Provider    : {config.llm_provider}")
    print(f" LLM Model       : {get_display_model()}")

    print("=" * 76)


# ----------------------------------------------------------------------
# Cache display
# ----------------------------------------------------------------------

def print_cache(cache):
    """Display current cache contents and optimizer metadata."""

    entries_method = getattr(cache, "entries", None)

    if not callable(entries_method):
        print("[!] Cache entry inspection is unavailable.")
        return

    entries = list(entries_method())

    print()
    print("-" * 76)
    print(f" CACHE CONTENTS ({len(entries)} entries)")
    print("-" * 76)

    if not entries:
        print(" Cache is empty.")

    else:
        for index, entry in enumerate(entries, start=1):
            frequency = cache.get_frequency(entry.query)

            print(f" [{index}] {entry.query}")
            print(f"      Frequency : {frequency}")

            if entry.similarity_score is not None:
                print(
                    f"      Similarity: "
                    f"{entry.similarity_score:.4f}"
                )

    print("-" * 76)
    print(f" Capacity : {cache.max_size}")
    print(f" Current  : {cache.size()}")
    print("-" * 76)


# ----------------------------------------------------------------------
# Response display
# ----------------------------------------------------------------------

def print_response(
    query: str,
    response,
    previous_cache_size: int,
    current_cache_size: int,
):
    """Display detailed information about one query."""

    print()
    print("=" * 76)
    print(f" QUERY: {query}")
    print("=" * 76)

    print(f" Source       : {response.source.upper()}")

    if response.decision is not None:
        print(f" Decision     : {response.decision.route}")

    if response.similarity_score is not None:
        print(f" Similarity   : {response.similarity_score:.4f}")
    else:
        print(" Similarity   : N/A")

    print(f" Latency      : {response.latency_ms:.2f} ms")

    metadata = response.metadata or {}

    llm_metadata = metadata.get("llm_metadata") or {}

    provider = metadata.get(
        "provider",
        config.llm_provider,
    )

    actual_model = metadata.get(
        "model",
        llm_metadata.get(
            "model_used",
            get_display_model(),
        ),
    )

    print(f" Provider     : {provider}")
    print(f" Model        : {actual_model}")

    if response.source == "cache":

        print()
        print("[+] CACHE HIT")
        print("[+] LLM CALL AVOIDED")
        print("[+] Cached response returned")

    elif response.source == "error":

        print()
        print("[!] CACHE MISS")
        print("[!] LLM GENERATION FAILED")
        print("[!] No failed response was stored in cache")

        error = metadata.get("llm_error")

        if error:
            print(f"[!] Error details: {error}")

    else:

        print()
        print("[+] CACHE MISS")
        print("[>] Query successfully processed by LLM")

        if llm_metadata.get("fallback_used"):
            print("[!] Fallback model was used.")

        if current_cache_size > previous_cache_size:
            print("[+] Response stored in cache")

        elif current_cache_size == previous_cache_size:
            print("[+] Cache optimizer processed the response")

    print()
    print(" RESPONSE")
    print("-" * 76)
    print(response.response_text)
    print("-" * 76)


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------

def print_statistics(
    total_queries: int,
    cache_hits: int,
    cache_misses: int,
    llm_call_attempts: int,
    successful_llm_calls: int,
    failed_llm_calls: int,
    cache,
):
    """Display accurate session statistics."""

    print()
    print("=" * 76)
    print("                       SESSION STATISTICS")
    print("=" * 76)

    if total_queries > 0:
        hit_rate = cache_hits / total_queries * 100
    else:
        hit_rate = 0.0

    print(f" Total Queries           : {total_queries}")
    print(f" Cache Hits              : {cache_hits}")
    print(f" Cache Misses            : {cache_misses}")
    print(f" Cache Hit Rate          : {hit_rate:.2f}%")

    print("-" * 76)

    print(f" LLM Call Attempts       : {llm_call_attempts}")
    print(f" Successful LLM Calls    : {successful_llm_calls}")
    print(f" Failed LLM Calls        : {failed_llm_calls}")
    print(f" LLM Calls Avoided       : {cache_hits}")

    print("-" * 76)

    print(f" Cache Size              : {cache.size()}/{cache.max_size}")
    print(" Eviction Policy         : LRU")

    print("=" * 76)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    """Run the interactive semantic caching demonstration."""

    print_header()

    cache, embedding_mode = create_cache()

    print()
    print(f" Embedding Mode : {embedding_mode}")

    print()
    print("[*] Initializing Dhano's LLM pipeline...")

    evaluator = EvaluationTracker()

    pipeline = SemanticCachingPipeline(
        cache=cache,
        evaluator=evaluator,
    )

    print("[+] Pipeline initialized.")
    print("[+] TanTan semantic cache connected.")
    print("[+] Dhano LLM pipeline connected.")
    print("[+] Praj cache optimizer connected.")
    print("[+] Andi evaluation tracker connected.")

    print()
    print("=" * 76)
    print("                       INTERACTIVE DEMO")
    print("=" * 76)

    print(" Enter ANY query.")
    print(" Type 'cache' to view cached entries.")
    print(" Type 'stats' to view session statistics.")
    print(" Type 'benchmark' to run evaluation benchmark.")
    print(" Type 'exit' to stop the demo.")

    print("=" * 76)

    total_queries = 0
    cache_hits = 0
    cache_misses = 0

    llm_call_attempts = 0
    successful_llm_calls = 0
    failed_llm_calls = 0

    while True:

        print()

        try:
            raw_query = input(" Enter query > ").strip()

        except (KeyboardInterrupt, EOFError):
            print()
            print("Demo terminated.")
            break

        if not raw_query:
            print("[!] Please enter a query.")
            continue

        command = raw_query.lower()

        if command == "exit":
            print()
            print("Exiting semantic cache demo...")
            break

        if command == "cache":
            print_cache(cache)
            continue

        if command == "stats":

            print_statistics(
                total_queries=total_queries,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                llm_call_attempts=llm_call_attempts,
                successful_llm_calls=successful_llm_calls,
                failed_llm_calls=failed_llm_calls,
                cache=cache,
            )

            continue

        if command == "benchmark":

            from evaluation.benchmark import (
                print_benchmark_report,
                run_benchmark,
            )

            print()
            print("[*] Running benchmark suite...")

            bench_results = run_benchmark()

            print_benchmark_report(bench_results)
            continue

        previous_cache_size = cache.size()

        response = pipeline.process_query(raw_query)

        current_cache_size = cache.size()

        total_queries += 1

        if response.source == "cache":

            cache_hits += 1

        else:

            cache_misses += 1

            metadata = response.metadata or {}

            attempted = metadata.get(
                "llm_call_attempted",
                response.source in ("llm", "error"),
            )

            succeeded = metadata.get(
                "llm_call_succeeded",
                response.source == "llm",
            )

            failed = metadata.get(
                "llm_call_failed",
                response.source == "error",
            )

            if attempted:
                llm_call_attempts += 1

            if succeeded:
                successful_llm_calls += 1

            if failed:
                failed_llm_calls += 1

        print_response(
            query=raw_query,
            response=response,
            previous_cache_size=previous_cache_size,
            current_cache_size=current_cache_size,
        )

        print_statistics(
            total_queries=total_queries,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            llm_call_attempts=llm_call_attempts,
            successful_llm_calls=successful_llm_calls,
            failed_llm_calls=failed_llm_calls,
            cache=cache,
        )


if __name__ == "__main__":
    main()