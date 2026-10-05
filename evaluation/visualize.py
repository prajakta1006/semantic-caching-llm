"""Visualization helper owned by Andi.

Generates comparative charts (LLM Calls, Cost, Latency) from benchmark results.
"""

import json
from pathlib import Path
from typing import Optional


def generate_benchmark_charts(
    results_json_path: Optional[Path] = None,
    output_chart_path: Optional[Path] = None,
) -> Path:
    """Generate comparative visualization chart from benchmark results JSON.

    Args:
        results_json_path: Path to benchmark_results.json.
        output_chart_path: Target image file path (.png).

    Returns:
        Path to generated PNG chart.
    """
    if results_json_path is None:
        results_json_path = (
            Path(__file__).resolve().parent
            / "results"
            / "benchmark_results.json"
        )

    if not results_json_path.exists():
        from evaluation.benchmark import run_benchmark

        print("[*] Benchmark results not found; running benchmark first...")
        run_benchmark(save_results=True)

    with open(results_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    base = data["baseline"]
    cached = data["cached"]
    comp = data["comparison"]

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to generate benchmark charts. "
            "Please install matplotlib or run without visualization."
        ) from exc

    # Set up figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Semantic Cache Performance Evaluation ({data['experiment']['number_of_queries']} Queries)",
        fontsize=14,
        fontweight="bold",
    )

    categories = ["Without Cache", "With Cache"]

    # Chart 1: LLM Calls
    llm_calls = [base["llm_calls"], cached["llm_calls"]]
    bars1 = axes[0].bar(
        categories, llm_calls, color=["#e74c3c", "#2ecc71"], width=0.5
    )
    axes[0].set_title("LLM Calls Needed")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.5,
            f"{int(yval)}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Chart 2: Estimated Cost ($)
    costs = [base["estimated_cost"], cached["estimated_cost"]]
    bars2 = axes[1].bar(
        categories, costs, color=["#e74c3c", "#2ecc71"], width=0.5
    )
    axes[1].set_title("Estimated Cost ($)")
    axes[1].set_ylabel("Cost ($)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.001,
            f"${yval:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Chart 3: Hit Rate & Cost Savings Summary
    metrics_names = ["Hit Rate (%)", "Cost Savings (%)", "LLM Reduction (%)"]
    values = [
        cached["hit_rate_pct"],
        comp["cost_reduction_pct"],
        comp["llm_calls_reduced_pct"],
    ]
    bars3 = axes[2].bar(
        metrics_names, values, color=["#3498db", "#9b59b6", "#1abc9c"], width=0.5
    )
    axes[2].set_title("Optimization Impact")
    axes[2].set_ylabel("Percentage (%)")
    axes[2].set_ylim(0, 100)
    axes[2].grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars3:
        yval = bar.get_height()
        axes[2].text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 2,
            f"{yval:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()

    if output_chart_path is None:
        output_chart_path = (
            Path(__file__).resolve().parent / "results" / "benchmark_chart.png"
        )

    output_chart_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_chart_path, dpi=300)
    plt.close()

    print(f"[+] Benchmark chart saved to {output_chart_path}")
    return output_chart_path


def main():
    """CLI entrypoint: py -m evaluation.visualize"""
    chart_path = generate_benchmark_charts()
    print(f"Visualization complete: {chart_path}")


if __name__ == "__main__":
    main()
