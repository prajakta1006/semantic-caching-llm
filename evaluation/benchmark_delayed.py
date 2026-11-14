
"""Realistic delayed-mock benchmark runner.

Uses the existing benchmark implementation but replaces its MockLLM
with a configurable delayed version. No real API calls are made.
"""

import json
import os
import time
from pathlib import Path

import evaluation.benchmark as benchmark_module
from src.llm import MockLLM


class DelayedMockLLM(MockLLM):
    """Mock LLM with configurable simulated generation latency."""

    def __init__(
        self,
        default_model: str = "mock-llm-v1",
        delay_seconds: float = 0.5,
    ) -> None:
        super().__init__(default_model=default_model)

        if delay_seconds < 0:
            raise ValueError("Delay cannot be negative.")

        self.delay_seconds = delay_seconds

    def generate(self, query, model=None):
        time.sleep(self.delay_seconds)
        return super().generate(query, model)


def main():
    """Run the existing benchmark with delayed MockLLM."""

    delay_ms = float(
        os.getenv("BENCHMARK_LLM_DELAY_MS", "500")
    )

    if delay_ms < 0:
        raise ValueError("BENCHMARK_LLM_DELAY_MS cannot be negative.")

    delay_seconds = delay_ms / 1000.0

    # Replace the MockLLM reference used by the existing benchmark.
    # Both baseline and cached experiments will use this same class.
    benchmark_module.MockLLM = lambda default_model="mock-llm-v1": (
        DelayedMockLLM(
            default_model=default_model,
            delay_seconds=delay_seconds,
        )
    )

    print("=" * 66)
    print("       REALISTIC DELAYED-MOCK BENCHMARK")
    print("=" * 66)
    print(f"Simulated LLM delay : {delay_ms:.0f} ms per generation")
    print("Real Gemini API     : Disabled")
    print("Embedding model     : Real SentenceTransformer")
    print("=" * 66)

    results = benchmark_module.run_benchmark(
        save_results=True
    )

    # Record the simulation setting in the JSON report.
    results["experiment"]["simulated_llm_delay_ms"] = delay_ms

    results["experiment"]["llm_mode"] = (
        f"Delayed MockLLM ({delay_ms:.0f} ms simulated delay)"
    )

    results["experiment"]["note"] = (
        "Both experiments use the same configurable simulated LLM "
        "generation delay. Cached latency includes real embedding "
        "generation, cache lookup and simulated LLM generation on misses. "
        "This is a controlled simulation, not a live Gemini benchmark."
    )

    output_path = (
        Path(__file__).resolve().parent
        / "results"
        / "benchmark_results.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\n[+] Simulation settings saved to benchmark_results.json")

    benchmark_module.print_benchmark_report(results)


if __name__ == "__main__":
    main()
