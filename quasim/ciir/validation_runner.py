r"""Multi-scenario CIIR → QuASIM → QRATUM validation runner.

Executes a suite of simulation scenarios with varying parameter
configurations to validate convergence, stability, and observer
non-commutativity across:

    1. Baseline — Default R, D, B, 3 constraints, 2 observers, 200 steps
    2. Constraint sweep — 1–10 constraints
    3. Observer sweep — 1–5 observers, including non-commutative pairs
    4. Step sweep — 50, 100, 500, 1000 steps
    5. High-dimensional — D ≥ 16, R ≥ 8
    6. Stress test — Large batch of simultaneous agents (multi-ψ)

Each scenario collects per-step metrics and produces a structured
JSON report with convergence analysis, anomaly detection, and
observer non-commutativity metrics.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quasim.ciir.simulation.engine import CIIRSimulationEngine, SimulationConfig

# ================================================================
# Result data structures
# ================================================================


@dataclass
class ScenarioMetrics:
    """Per-step metrics collected during a scenario run."""

    steps: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    gradient_norms: list[float] = field(default_factory=list)
    purities: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    constraint_losses: list[float] = field(default_factory=list)
    observer_losses: list[float] = field(default_factory=list)
    entropy_losses: list[float] = field(default_factory=list)
    step_times_ms: list[float] = field(default_factory=list)


@dataclass
class ScenarioResult:
    """Result of a single validation scenario."""

    name: str
    config: dict[str, Any]
    metrics: ScenarioMetrics
    summary: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0
    passed: bool = True
    anomalies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "name": self.name,
            "config": self.config,
            "summary": self.summary,
            "elapsed_s": round(self.elapsed_s, 3),
            "passed": self.passed,
            "anomalies": self.anomalies,
            "metrics": {
                "steps": self.metrics.steps,
                "losses": self.metrics.losses,
                "gradient_norms": self.metrics.gradient_norms,
                "purities": self.metrics.purities,
                "entropies": self.metrics.entropies,
            },
        }


@dataclass
class ValidationReport:
    """Complete validation report across all scenarios."""

    scenarios: list[ScenarioResult] = field(default_factory=list)
    total_elapsed_s: float = 0.0

    @property
    def n_passed(self) -> int:
        return sum(1 for s in self.scenarios if s.passed)

    @property
    def n_total(self) -> int:
        return len(self.scenarios)

    def to_dict(self) -> dict[str, Any]:
        """Serialize full report."""
        return {
            "summary": {
                "total_scenarios": self.n_total,
                "passed": self.n_passed,
                "failed": self.n_total - self.n_passed,
                "total_elapsed_s": round(self.total_elapsed_s, 3),
            },
            "scenarios": [s.to_dict() for s in self.scenarios],
        }

    def save_json(self, path: str) -> None:
        """Save report as JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


# ================================================================
# Scenario execution
# ================================================================


def _run_scenario(
    name: str,
    config: SimulationConfig,
    verbose: bool = True,
) -> ScenarioResult:
    """Run a single validation scenario and collect metrics.

    Parameters
    ----------
    name : str
        Human-readable scenario name.
    config : SimulationConfig
        Simulation configuration for this scenario.
    verbose : bool
        Print progress to console.

    Returns
    -------
    ScenarioResult
        Collected metrics and analysis.
    """
    t0 = time.time()

    # Suppress screenshot output for validation runs
    config.screenshot_interval = 0

    engine = CIIRSimulationEngine(config)
    engine.init()

    metrics = ScenarioMetrics()
    anomalies: list[str] = []

    for _step_idx in range(config.n_steps):
        if not engine.running:
            break

        snapshot = engine.step()
        metrics.steps.append(snapshot.step)
        metrics.losses.append(snapshot.total_loss)
        metrics.gradient_norms.append(snapshot.gradient_norm)
        metrics.purities.append(snapshot.purity)
        metrics.entropies.append(snapshot.entropy)
        metrics.constraint_losses.append(snapshot.constraint_loss)
        metrics.observer_losses.append(snapshot.observer_loss)
        metrics.entropy_losses.append(snapshot.entropy_loss)
        metrics.step_times_ms.append(snapshot.step_time_ms)

        # Anomaly detection
        if np.isnan(snapshot.total_loss):
            anomalies.append(f"NaN loss at step {snapshot.step}")
            break
        if np.isinf(snapshot.total_loss):
            anomalies.append(f"Inf loss at step {snapshot.step}")
            break
        if snapshot.gradient_norm > 1e10:
            anomalies.append(f"Gradient explosion at step {snapshot.step}: {snapshot.gradient_norm:.2e}")

    elapsed = time.time() - t0

    # Build summary
    summary = _compute_summary(metrics, engine, config)

    # Determine pass/fail
    passed = len(anomalies) == 0
    if metrics.losses and len(metrics.losses) >= 2 and metrics.losses[-1] > metrics.losses[0] * 10:
            anomalies.append("Loss divergence detected")
            passed = False

    result = ScenarioResult(
        name=name,
        config=_config_to_dict(config),
        metrics=metrics,
        summary=summary,
        elapsed_s=elapsed,
        passed=passed,
        anomalies=anomalies,
    )

    if verbose:
        status = "✓ PASS" if passed else "✗ FAIL"
        n_steps = len(metrics.steps)
        final_loss = metrics.losses[-1] if metrics.losses else float("nan")
        print(f"  [{status}] {name}: {n_steps} steps, loss={final_loss:.6f}, {elapsed:.2f}s")
        if anomalies:
            for a in anomalies:
                print(f"         ⚠ {a}")

    return result


def _compute_summary(
    metrics: ScenarioMetrics,
    engine: CIIRSimulationEngine,
    config: SimulationConfig,
) -> dict[str, Any]:
    """Compute summary statistics from scenario metrics."""
    summary: dict[str, Any] = {}

    if not metrics.losses:
        return summary

    losses = np.array(metrics.losses)
    grads = np.array(metrics.gradient_norms)
    purities = np.array(metrics.purities)
    entropies = np.array(metrics.entropies)
    step_times = np.array(metrics.step_times_ms)

    summary["n_steps"] = len(metrics.steps)
    summary["initial_loss"] = float(losses[0])
    summary["final_loss"] = float(losses[-1])
    summary["loss_reduction_pct"] = float((losses[0] - losses[-1]) / max(abs(losses[0]), 1e-12) * 100)
    summary["mean_gradient_norm"] = float(grads.mean())
    summary["final_gradient_norm"] = float(grads[-1])
    summary["mean_purity"] = float(purities.mean())
    summary["final_purity"] = float(purities[-1])
    summary["mean_entropy"] = float(entropies.mean())
    summary["final_entropy"] = float(entropies[-1])
    summary["mean_step_time_ms"] = float(step_times.mean())
    summary["total_time_ms"] = float(step_times.sum())
    summary["throughput_steps_per_s"] = float(len(metrics.steps) / max(step_times.sum() / 1000, 1e-12))

    # Convergence check
    if len(losses) >= 10:
        tail = losses[-10:]
        summary["converged"] = bool(np.std(tail) < config.convergence_tol)
        summary["loss_std_tail10"] = float(np.std(tail))
    else:
        summary["converged"] = False

    # Observer non-commutativity
    if engine.ciir_observer is not None:
        comm_matrix = engine.ciir_observer.commutator_matrix()
        summary["max_commutator_norm"] = float(np.max(np.abs(comm_matrix)))
        summary["non_commutative"] = bool(np.max(np.abs(comm_matrix)) > 1e-10)

    return summary


def _config_to_dict(config: SimulationConfig) -> dict[str, Any]:
    """Convert SimulationConfig to serializable dict."""
    return {
        "rank": config.rank,
        "rep_dim": config.rep_dim,
        "batch_size": config.batch_size,
        "n_constraints": config.n_constraints,
        "n_observers": config.n_observers,
        "n_steps": config.n_steps,
        "learning_rate": config.learning_rate,
        "entropy_weight": config.entropy_weight,
        "seed": config.seed,
    }


# ================================================================
# Validation scenarios
# ================================================================


def run_baseline(verbose: bool = True) -> ScenarioResult:
    """Scenario 1: Baseline — default parameters."""
    config = SimulationConfig(
        rank=4,
        rep_dim=8,
        batch_size=4,
        n_constraints=3,
        n_observers=2,
        n_steps=200,
        learning_rate=0.01,
        entropy_weight=0.01,
        seed=42,
        output_dir="/tmp/ciir_validation/baseline",
        log_metrics=False,
        log_tensors=False,
    )
    return _run_scenario("baseline", config, verbose=verbose)


def run_constraint_sweep(verbose: bool = True) -> list[ScenarioResult]:
    """Scenario 2: Sweep number of constraints from 1 to 10."""
    results = []
    for n_c in [1, 3, 5, 7, 10]:
        config = SimulationConfig(
            rank=4,
            rep_dim=8,
            batch_size=4,
            n_constraints=n_c,
            n_observers=2,
            n_steps=100,
            learning_rate=0.01,
            entropy_weight=0.01,
            seed=42,
            output_dir=f"/tmp/ciir_validation/constraint_sweep_C{n_c}",
            log_metrics=False,
            log_tensors=False,
        )
        result = _run_scenario(f"constraint_sweep_C{n_c}", config, verbose=verbose)
        results.append(result)
    return results


def run_observer_sweep(verbose: bool = True) -> list[ScenarioResult]:
    """Scenario 3: Sweep number of observers from 1 to 5."""
    results = []
    for n_o in [1, 2, 3, 5]:
        config = SimulationConfig(
            rank=4,
            rep_dim=8,
            batch_size=4,
            n_constraints=3,
            n_observers=n_o,
            n_steps=100,
            learning_rate=0.01,
            entropy_weight=0.01,
            seed=42,
            output_dir=f"/tmp/ciir_validation/observer_sweep_O{n_o}",
            log_metrics=False,
            log_tensors=False,
        )
        result = _run_scenario(f"observer_sweep_O{n_o}", config, verbose=verbose)
        results.append(result)
    return results


def run_step_sweep(verbose: bool = True) -> list[ScenarioResult]:
    """Scenario 4: Sweep step counts: 50, 100, 500, 1000."""
    results = []
    for n_steps in [50, 100, 500, 1000]:
        config = SimulationConfig(
            rank=4,
            rep_dim=8,
            batch_size=4,
            n_constraints=3,
            n_observers=2,
            n_steps=n_steps,
            learning_rate=0.01,
            entropy_weight=0.01,
            seed=42,
            output_dir=f"/tmp/ciir_validation/step_sweep_S{n_steps}",
            log_metrics=False,
            log_tensors=False,
        )
        result = _run_scenario(f"step_sweep_S{n_steps}", config, verbose=verbose)
        results.append(result)
    return results


def run_high_dimensional(verbose: bool = True) -> ScenarioResult:
    """Scenario 5: High-dimensional test — D=16, R=8."""
    config = SimulationConfig(
        rank=8,
        rep_dim=16,
        batch_size=4,
        n_constraints=3,
        n_observers=2,
        n_steps=100,
        learning_rate=0.005,
        entropy_weight=0.01,
        seed=42,
        output_dir="/tmp/ciir_validation/high_dim",
        log_metrics=False,
        log_tensors=False,
    )
    return _run_scenario("high_dimensional_D16_R8", config, verbose=verbose)


def run_stress_test(verbose: bool = True) -> ScenarioResult:
    """Scenario 6: Stress test — large batch (multi-ψ simulation)."""
    config = SimulationConfig(
        rank=4,
        rep_dim=8,
        batch_size=32,
        n_constraints=5,
        n_observers=3,
        n_steps=100,
        learning_rate=0.01,
        entropy_weight=0.01,
        seed=42,
        output_dir="/tmp/ciir_validation/stress_test",
        log_metrics=False,
        log_tensors=False,
    )
    return _run_scenario("stress_test_B32_C5_O3", config, verbose=verbose)


# ================================================================
# Full validation suite
# ================================================================


def run_all_validations(
    verbose: bool = True,
    output_path: str | None = None,
) -> ValidationReport:
    """Run all validation scenarios and produce a structured report.

    Parameters
    ----------
    verbose : bool
        Print progress to console.
    output_path : str, optional
        Path to save JSON report. If None, no file is saved.

    Returns
    -------
    ValidationReport
        Complete validation report.
    """
    t0 = time.time()
    report = ValidationReport()

    if verbose:
        print("╔══════════════════════════════════════════════════╗")
        print("║  CIIR Multi-Scenario Validation Suite            ║")
        print("╚══════════════════════════════════════════════════╝\n")

    # 1. Baseline
    if verbose:
        print("Scenario 1: Baseline")
    report.scenarios.append(run_baseline(verbose=verbose))

    # 2. Constraint sweep
    if verbose:
        print("\nScenario 2: Constraint Sweep")
    report.scenarios.extend(run_constraint_sweep(verbose=verbose))

    # 3. Observer sweep
    if verbose:
        print("\nScenario 3: Observer Sweep")
    report.scenarios.extend(run_observer_sweep(verbose=verbose))

    # 4. Step sweep
    if verbose:
        print("\nScenario 4: Step Sweep")
    report.scenarios.extend(run_step_sweep(verbose=verbose))

    # 5. High-dimensional
    if verbose:
        print("\nScenario 5: High-Dimensional")
    report.scenarios.append(run_high_dimensional(verbose=verbose))

    # 6. Stress test
    if verbose:
        print("\nScenario 6: Stress Test")
    report.scenarios.append(run_stress_test(verbose=verbose))

    report.total_elapsed_s = time.time() - t0

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Validation Complete: {report.n_passed}/{report.n_total} passed")
        print(f"Total elapsed: {report.total_elapsed_s:.2f}s")

    if output_path:
        report.save_json(output_path)
        if verbose:
            print(f"Report saved to: {output_path}")

    return report


def generate_convergence_plots(
    report: ValidationReport,
    output_dir: str = "/tmp/ciir_validation/plots",
) -> list[str]:
    """Generate convergence plots for all scenarios in a validation report.

    Parameters
    ----------
    report : ValidationReport
        Completed validation report.
    output_dir : str
        Directory to save plot files.

    Returns
    -------
    list of str
        Paths to generated plot files.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    paths: list[str] = []

    # 1. Loss convergence overlay
    fig, ax = plt.subplots(figsize=(12, 6))
    for scenario in report.scenarios:
        if scenario.metrics.losses:
            ax.plot(
                scenario.metrics.steps,
                scenario.metrics.losses,
                label=scenario.name,
                linewidth=1,
                alpha=0.7,
            )
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("CIIR Loss Convergence — All Scenarios")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "loss_convergence_all.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    # 2. Purity vs step
    fig, ax = plt.subplots(figsize=(12, 6))
    for scenario in report.scenarios:
        if scenario.metrics.purities:
            ax.plot(
                scenario.metrics.steps,
                scenario.metrics.purities,
                label=scenario.name,
                linewidth=1,
                alpha=0.7,
            )
    ax.set_xlabel("Step")
    ax.set_ylabel("Purity Tr(ρ²)")
    ax.set_title("Purity Evolution — All Scenarios")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "purity_all.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    # 3. Entropy vs step
    fig, ax = plt.subplots(figsize=(12, 6))
    for scenario in report.scenarios:
        if scenario.metrics.entropies:
            ax.plot(
                scenario.metrics.steps,
                scenario.metrics.entropies,
                label=scenario.name,
                linewidth=1,
                alpha=0.7,
            )
    ax.set_xlabel("Step")
    ax.set_ylabel("Entropy S(ρ)")
    ax.set_title("Entropy Evolution — All Scenarios")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "entropy_all.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    # 4. Gradient norm (log scale)
    fig, ax = plt.subplots(figsize=(12, 6))
    for scenario in report.scenarios:
        if scenario.metrics.gradient_norms:
            ax.semilogy(
                scenario.metrics.steps,
                scenario.metrics.gradient_norms,
                label=scenario.name,
                linewidth=1,
                alpha=0.7,
            )
    ax.set_xlabel("Step")
    ax.set_ylabel("‖∇L‖")
    ax.set_title("Gradient Norm — All Scenarios")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "gradient_norm_all.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    # 5. Summary bar chart: final loss per scenario
    fig, ax = plt.subplots(figsize=(14, 6))
    names = [s.name for s in report.scenarios if s.metrics.losses]
    final_losses = [s.metrics.losses[-1] for s in report.scenarios if s.metrics.losses]
    colors = ["green" if s.passed else "red" for s in report.scenarios if s.metrics.losses]
    bars = ax.bar(range(len(names)), final_losses, color=colors, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Final Loss")
    ax.set_title("Final Loss per Scenario")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, final_losses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    plt.tight_layout()
    path = os.path.join(output_dir, "final_loss_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    return paths
