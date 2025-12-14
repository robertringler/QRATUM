#!/usr/bin/env python
"""One-click end-to-end benchmark runner for QuASIM-Own.

This script runs the complete benchmark suite and generates all reports.
"""

from pathlib import Path

from quasim.ownai.determinism import set_seed
from quasim.ownai.eval.benchmark import run_benchmark_suite
from quasim.ownai.eval.reporting import (
    generate_markdown_report,
    save_results_csv,
    save_results_json,
)
from quasim.ownai.integration.model_card import generate_model_card
from quasim.ownai.integration.terc_observables import (
    collect_terc_observables,
    save_terc_observables,
)


def main():
    """Run complete benchmark suite."""
    print("=" * 60)
    print("QuASIM-Own: Complete Benchmark Suite")
    print("=" * 60)
    print()

    # Set global seed
    set_seed(1337)

    # Run benchmarks
    print("Running standard benchmark suite...")
    print("This may take 15-30 minutes depending on your hardware.")
    print()

    results = run_benchmark_suite(suite="std", n_repeats=3)

    if not results:
        print("❌ No results generated!")
        return

    print()
    print(f"✅ Completed {len(results)} benchmark runs")
    print()

    # Create reports directory
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)

    # Save results
    print("Saving results...")

    # CSV
    csv_path = report_dir / "benchmark_results.csv"
    save_results_csv(results, csv_path)
    print(f"  ✅ CSV: {csv_path}")

    # JSON
    json_path = report_dir / "benchmark_results.json"
    save_results_json(results, json_path)
    print(f"  ✅ JSON: {json_path}")

    # Markdown report
    md_path = Path("docs/ownai/benchmarks.md")
    generate_markdown_report(
        results,
        md_path,
        title="QuASIM-Own Benchmark Results",
    )
    print(f"  ✅ Markdown: {md_path}")

    # TERC observables
    observables = collect_terc_observables(results)
    terc_path = report_dir / "terc_observables.json"
    save_terc_observables(observables, terc_path)
    print(f"  ✅ TERC observables: {terc_path}")

    # Generate model cards for each model
    print()
    print("Generating model cards...")

    models = {r.model_name for r in results}
    for model_name in models:
        card_path = Path(f"docs/ownai/model_card_{model_name}.md")
        model_results = [r for r in results if r.model_name == model_name]
        generate_model_card(model_name, model_results, card_path)
        print(f"  ✅ Model card: {card_path}")

    # Print summary
    print()
    print("=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    print(f"Total runs: {len(results)}")
    print(f"Stability margin: {observables['stability_margin']:.3f}")
    print(f"QGH consensus: {'✅ Yes' if observables['qgh_consensus_status'] else '❌ No'}")
    print(f"Emergent complexity: {observables['emergent_complexity']:.3f}")
    print(f"Goal progress: {observables['goal_progress']:.3f}")
    print()
    print("✅ All benchmarks and reports generated successfully!")
    print()


if __name__ == "__main__":
    main()
