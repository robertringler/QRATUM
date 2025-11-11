"""QuASIM-Own CLI: Train, evaluate, benchmark, and export models."""

import json
from datetime import datetime
from pathlib import Path

import click

from quasim.ownai.determinism import set_seed
from quasim.ownai.eval.benchmark import benchmark_model, run_benchmark_suite
from quasim.ownai.eval.reporting import (
    generate_markdown_report,
    save_results_csv,
    save_results_json,
)
from quasim.ownai.integration.model_card import generate_model_card
from quasim.ownai.integration.qgh_hooks import QGHLedger, create_run_id
from quasim.ownai.integration.terc_observables import (
    collect_terc_observables,
    save_terc_observables,
)


@click.group()
def cli():
    """QuASIM-Own: Deterministic AI with Symbolic-Latent Transformer (Modo)."""
    pass


@cli.command()
@click.option("--model", default="slt", help="Model type (slt, mlp, cnn)")
@click.option("--task", default="text-cls", help="Task type")
@click.option("--dataset", default="imdb-mini", help="Dataset name")
@click.option("--seed", default=1337, help="Random seed")
@click.option("--out", default="runs/default", help="Output directory")
def train(model: str, task: str, dataset: str, seed: int, out: str):
    """Train a model on a dataset."""
    click.echo(f"Training {model} on {task}/{dataset} with seed={seed}")

    set_seed(seed)
    output_dir = Path(out)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = benchmark_model(model, task, dataset, seed)

        # Save results
        results_file = output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "model": model,
                    "task": task,
                    "dataset": dataset,
                    "seed": seed,
                    "primary_metric": result.primary_metric,
                    "secondary_metric": result.secondary_metric,
                    "prediction_hash": result.prediction_hash,
                },
                f,
                indent=2,
            )

        # Log to QGH ledger
        ledger = QGHLedger()
        run_id = create_run_id(model, task, dataset, seed)
        ledger.append_run(
            run_id=run_id,
            model=model,
            task=task,
            dataset=dataset,
            seed=seed,
            prediction_hash=result.prediction_hash,
        )

        click.echo("✅ Training complete!")
        click.echo(f"   Primary metric: {result.primary_metric:.4f}")
        click.echo(f"   Results saved to: {results_file}")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise


@cli.command()
@click.option("--run", required=True, help="Run directory")
@click.option("--metrics", default="all", help="Metrics to compute")
def eval(run: str, metrics: str):
    """Evaluate a trained model."""
    run_dir = Path(run)

    if not run_dir.exists():
        click.echo(f"❌ Run directory not found: {run_dir}", err=True)
        return

    results_file = run_dir / "results.json"
    if not results_file.exists():
        click.echo(f"❌ Results file not found: {results_file}", err=True)
        return

    with open(results_file) as f:
        results = json.load(f)

    click.echo(f"📊 Evaluation Results for {run}")
    click.echo(f"   Model: {results.get('model', 'N/A')}")
    click.echo(f"   Task: {results.get('task', 'N/A')}")
    click.echo(f"   Primary Metric: {results.get('primary_metric', 'N/A')}")
    click.echo(f"   Secondary Metric: {results.get('secondary_metric', 'N/A')}")


@cli.command()
@click.option("--suite", default="std", help="Suite type (quick, std, full)")
@click.option("--repeat", default=3, help="Number of repeats")
@click.option("--cpu-only", is_flag=True, default=True, help="CPU-only mode")
@click.option("--report", default="reports/own", help="Report output directory")
def benchmark(suite: str, repeat: int, cpu_only: bool, report: str):
    """Run comprehensive benchmarks."""
    click.echo(f"🚀 Running {suite} benchmark suite with {repeat} repeats...")

    if cpu_only:
        click.echo("   Running in CPU-only mode")

    try:
        # Run benchmarks
        results = run_benchmark_suite(suite=suite, n_repeats=repeat)

        if not results:
            click.echo("❌ No results generated", err=True)
            return

        # Save results
        report_dir = Path(report)
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV
        csv_path = report_dir / f"results_{timestamp}.csv"
        save_results_csv(results, csv_path)
        click.echo(f"   ✅ CSV saved: {csv_path}")

        # JSON
        json_path = report_dir / f"results_{timestamp}.json"
        save_results_json(results, json_path)
        click.echo(f"   ✅ JSON saved: {json_path}")

        # Markdown report
        md_path = report_dir / f"summary_{timestamp}.md"
        generate_markdown_report(results, md_path)
        click.echo(f"   ✅ Markdown report: {md_path}")

        # TERC observables
        observables = collect_terc_observables(results)
        terc_path = report_dir / f"terc_observables_{timestamp}.json"
        save_terc_observables(observables, terc_path)
        click.echo(f"   ✅ TERC observables: {terc_path}")

        # Summary
        click.echo("\n📈 Benchmark Summary:")
        click.echo(f"   Total runs: {len(results)}")
        click.echo(f"   Stability margin: {observables['stability_margin']:.3f}")
        click.echo(f"   QGH consensus: {'✅' if observables['qgh_consensus_status'] else '❌'}")
        click.echo(f"   Goal progress: {observables['goal_progress']:.3f}")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise


@cli.command()
@click.option("--run", required=True, help="Run directory")
@click.option("--format", default="json", help="Export format (json, onnx)")
@click.option("--out", required=True, help="Output file path")
def export(run: str, format: str, out: str):
    """Export a trained model."""
    click.echo(f"Exporting model from {run} to {out} (format: {format})")

    # For now, just copy the results file
    run_dir = Path(run)
    results_file = run_dir / "results.json"

    if not results_file.exists():
        click.echo(f"❌ Results file not found: {results_file}", err=True)
        return

    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file) as f:
        data = json.load(f)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    click.echo(f"✅ Exported to: {output_path}")


@cli.command()
@click.option("--run", required=True, help="Run directory")
@click.option("--out", required=True, help="Output model card path")
def modelcard(run: str, out: str):
    """Generate model card for a trained model."""
    run_dir = Path(run)
    results_file = run_dir / "results.json"

    if not results_file.exists():
        click.echo(f"❌ Results file not found: {results_file}", err=True)
        return

    with open(results_file) as f:
        result_data = json.load(f)

    # Create a dummy BenchmarkResult for card generation
    from quasim.ownai.eval.benchmark import BenchmarkResult

    result = BenchmarkResult(
        task=result_data.get("task", "unknown"),
        model_name=result_data.get("model", "unknown"),
        dataset=result_data.get("dataset", "unknown"),
        seed=result_data.get("seed", 42),
        primary_metric=result_data.get("primary_metric", 0.0),
        secondary_metric=result_data.get("secondary_metric", 0.0),
        latency_p50=result_data.get("latency_p50", 0.0),
        latency_p95=result_data.get("latency_p95", 0.0),
        throughput=result_data.get("throughput", 0.0),
        model_size_mb=result_data.get("model_size_mb", 0.0),
        energy_proxy=result_data.get("energy_proxy", 0.0),
        prediction_hash=result_data.get("prediction_hash", ""),
    )

    output_path = Path(out)
    generate_model_card(result.model_name, [result], output_path)

    click.echo(f"✅ Model card generated: {output_path}")


if __name__ == "__main__":
    cli()
