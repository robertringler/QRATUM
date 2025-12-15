"""Benchmark orchestration for QuASIM-Own models and baselines."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from quasim.ownai.data import loaders
from quasim.ownai.data.preprocess import StandardScaler
from quasim.ownai.determinism import hash_preds, set_seed
from quasim.ownai.models.mlp import DeterministicMLP
from quasim.ownai.models.slt import build_slt
from quasim.ownai.train.metrics import (accuracy, estimate_energy_proxy,
                                        estimate_model_size_mb, f1_score, mae,
                                        measure_latency, measure_throughput,
                                        rmse)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run.

    Attributes
    ----------
    task : str
        Task name
    model_name : str
        Model name
    dataset : str
        Dataset name
    seed : int
        Random seed used
    primary_metric : float
        Primary metric (accuracy for cls, mae for reg)
    secondary_metric : float
        Secondary metric (f1 for cls, rmse for reg)
    latency_p50 : float
        Median latency in ms
    latency_p95 : float
        95th percentile latency in ms
    throughput : float
        Predictions per second
    model_size_mb : float
        Model size in MB
    energy_proxy : float
        Energy consumption proxy
    prediction_hash : str
        Hash of predictions for determinism check
    """

    task: str
    model_name: str
    dataset: str
    seed: int
    primary_metric: float
    secondary_metric: float
    latency_p50: float
    latency_p95: float
    throughput: float
    model_size_mb: float
    energy_proxy: float
    prediction_hash: str


def benchmark_model(
    model_name: str,
    task: str,
    dataset: str,
    seed: int = 42,
) -> BenchmarkResult:
    """Benchmark a single model on a task/dataset.

    Parameters
    ----------
    model_name : str
        Name of the model ('slt', 'mlp', 'rf', 'logreg', etc.)
    task : str
        Task type ('tabular-cls', 'text-cls', 'vision-cls', 'ts-reg')
    dataset : str
        Dataset name
    seed : int
        Random seed

    Returns
    -------
    BenchmarkResult
        Benchmark results
    """
    set_seed(seed)

    # Load data
    if "text" in task:
        X, y = loaders.load_text(dataset)
        X_train, X_test = X[:800], X[800:]
        y_train, y_test = y[:800], y[800:]
    elif "vision" in task:
        X, y = loaders.load_vision(dataset)
        X_train, X_test = X[:800], X[800:]
        y_train, y_test = y[:800], y[800:]
    elif "ts" in task:
        X, y = loaders.load_timeseries(dataset)
        X_train, X_test = X[:800], X[800:]
        y_train, y_test = y[:800], y[800:]
    else:  # tabular
        X, y = loaders.load_tabular(dataset)

        # Scale features for tabular
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test = X[:800], X[800:]
        y_train, y_test = y[:800], y[800:]

    # Create model
    if model_name == "slt":
        model = build_slt(task=task, seed=seed)
    elif model_name == "mlp":
        task_type = "classification" if "cls" in task else "regression"
        model = DeterministicMLP(task=task_type, seed=seed)
    elif model_name == "rf":
        from quasim.baselines.sklearn_models import (
            get_random_forest_classifier, get_random_forest_regressor)

        if "cls" in task:
            model = get_random_forest_classifier(seed=seed)
        else:
            model = get_random_forest_regressor(seed=seed)
    elif model_name == "logreg":
        from quasim.baselines.sklearn_models import get_logistic_regression

        model = get_logistic_regression(seed=seed)
    elif model_name == "linearsvc":
        from quasim.baselines.sklearn_models import get_linear_svc

        model = get_linear_svc(seed=seed)
    elif model_name == "xgboost":
        from quasim.baselines.xgboost_lightgbm import (get_xgboost_classifier,
                                                       get_xgboost_regressor)

        if "cls" in task:
            model = get_xgboost_classifier(seed=seed)
        else:
            model = get_xgboost_regressor(seed=seed)
    elif model_name == "lightgbm":
        from quasim.baselines.xgboost_lightgbm import (get_lightgbm_classifier,
                                                       get_lightgbm_regressor)

        if "cls" in task:
            model = get_lightgbm_classifier(seed=seed)
        else:
            model = get_lightgbm_regressor(seed=seed)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    pred_hash = hash_preds(y_pred)

    # Compute metrics
    if "cls" in task:
        primary = accuracy(y_test, y_pred)
        secondary = f1_score(y_test, y_pred)
    else:  # regression
        primary = mae(y_test, y_pred)
        secondary = rmse(y_test, y_pred)

    # Performance metrics
    latency = measure_latency(model, X_test[:10], n_runs=20)
    throughput = measure_throughput(model, X_test[:10], duration_sec=0.5)
    model_size = estimate_model_size_mb(model)
    energy = estimate_energy_proxy(latency["p50_ms"])

    return BenchmarkResult(
        task=task,
        model_name=model_name,
        dataset=dataset,
        seed=seed,
        primary_metric=primary,
        secondary_metric=secondary,
        latency_p50=latency["p50_ms"],
        latency_p95=latency["p95_ms"],
        throughput=throughput,
        model_size_mb=model_size,
        energy_proxy=energy,
        prediction_hash=pred_hash,
    )


def run_benchmark_suite(
    suite: Literal["quick", "std", "full"] = "std",
    n_repeats: int = 3,
    output_dir: Path | None = None,
) -> list[BenchmarkResult]:
    """Run full benchmark suite.

    Parameters
    ----------
    suite : str
        Suite type: 'quick' (tabular+text), 'std' (all tasks), 'full' (all+more repeats)
    n_repeats : int
        Number of repetitions per configuration
    output_dir : Path, optional
        Output directory for results

    Returns
    -------
    list[BenchmarkResult]
        All benchmark results
    """
    # Define benchmark configurations
    if suite == "quick":
        tasks = [
            ("tabular-cls", "wine"),
            ("text-cls", "imdb-mini"),
        ]
        models = ["slt", "mlp", "rf", "logreg"]
    elif suite == "std":
        tasks = [
            ("tabular-cls", "wine"),
            ("text-cls", "imdb-mini"),
            ("vision-cls", "mnist-1k"),
            ("ts-reg", "synthetic-arma"),
        ]
        models = ["slt", "mlp", "rf", "logreg"]
    else:  # full
        tasks = [
            ("tabular-cls", "wine"),
            ("tabular-cls", "adult"),
            ("text-cls", "imdb-mini"),
            ("text-cls", "agnews-mini"),
            ("vision-cls", "mnist-1k"),
            ("vision-cls", "cifar10-subset"),
            ("ts-reg", "synthetic-arma"),
            ("ts-reg", "etth1-mini"),
        ]
        models = ["slt", "mlp", "rf", "logreg", "linearsvc"]
        n_repeats = max(n_repeats, 5)

    results = []

    # Run benchmarks
    total = len(tasks) * len(models) * n_repeats
    count = 0

    for task, dataset in tasks:
        for model_name in models:
            # Skip incompatible combinations
            if task == "ts-reg" and model_name in ["logreg", "linearsvc"]:
                continue

            for repeat in range(n_repeats):
                seed = 42 + repeat

                try:
                    result = benchmark_model(
                        model_name=model_name,
                        task=task,
                        dataset=dataset,
                        seed=seed,
                    )
                    results.append(result)

                    count += 1
                    print(
                        f"[{count}/{total}] {model_name} on {task}/{dataset} (seed={seed}): "
                        f"primary={result.primary_metric:.4f}"
                    )

                except Exception as e:
                    print(f"Error benchmarking {model_name} on {task}/{dataset}: {e}")
                    continue

    return results
