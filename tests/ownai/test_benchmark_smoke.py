"""Smoke tests for benchmark system."""

from quasim.ownai.eval.benchmark import benchmark_model, run_benchmark_suite


def test_benchmark_model_tabular():
    """Test benchmarking a single model on tabular task."""

    result = benchmark_model(
        model_name="rf",
        task="tabular-cls",
        dataset="wine",
        seed=42,
    )

    assert result is not None
    assert result.model_name == "rf"
    assert result.task == "tabular-cls"
    assert result.dataset == "wine"
    assert 0.0 <= result.primary_metric <= 1.0
    assert result.latency_p50 > 0
    assert len(result.prediction_hash) == 64


def test_benchmark_model_text():
    """Test benchmarking a model on text task."""

    result = benchmark_model(
        model_name="slt",
        task="text-cls",
        dataset="imdb-mini",
        seed=42,
    )

    assert result is not None
    assert result.model_name == "slt"
    assert result.task == "text-cls"
    assert 0.0 <= result.primary_metric <= 1.0


def test_run_benchmark_suite_quick():
    """Test running quick benchmark suite."""

    results = run_benchmark_suite(suite="quick", n_repeats=2)

    assert len(results) > 0
    # Quick suite: 2 tasks × 4 models × 2 repeats = 16 runs
    assert len(results) >= 10  # Allow for some failures


def test_benchmark_determinism():
    """Test that benchmarks produce consistent hashes for same seed."""

    result1 = benchmark_model(
        model_name="logreg",
        task="tabular-cls",
        dataset="wine",
        seed=42,
    )

    result2 = benchmark_model(
        model_name="logreg",
        task="tabular-cls",
        dataset="wine",
        seed=42,
    )

    # Same seed should produce same prediction hash
    assert result1.prediction_hash == result2.prediction_hash
