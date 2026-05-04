r"""Long-duration CIIR → QuASIM → QRATUM simulation runner.

Executes a continuous simulation for a configurable wall-clock duration
(default 8 hours) with:

    - Incremental checkpoint saving every N steps
    - Per-step metrics: loss, gradient norm, purity, entropy, constraint
      residuals, observer non-commutativity
    - Performance tracking: steps/s, memory usage
    - Anomaly detection and recovery (NaN, Inf, gradient explosion)
    - Summary plots at completion

Usage (programmatic)::

    from quasim.ciir.long_duration_runner import run_long_duration
    result = run_long_duration(duration_hours=0.01, seed=42)

Usage (CLI)::

    quasim-ciir long-run --duration-hours 8 --seed 42 --output-dir ciir_long
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
# Configuration
# ================================================================


@dataclass
class LongRunConfig:
    """Configuration for a long-duration simulation run.

    Parameters
    ----------
    duration_hours : float
        Maximum wall-clock duration in hours (0 for step-limited only).
    max_steps : int
        Maximum number of steps (0 for time-limited only).
    rank : int
        Ontic dimension R.
    rep_dim : int
        Cognitive representation dimension D.
    batch_size : int
        Number of parallel states B.
    n_constraints : int
        Number of constraint operators.
    n_observers : int
        Number of observer operators.
    learning_rate : float
        Gradient descent step size η.
    entropy_weight : float
        Entropy regularisation coefficient γ.
    seed : int
        Random seed for reproducibility.
    checkpoint_interval : int
        Save incremental checkpoint every N steps.
    log_interval : int
        Print console summary every N steps.
    output_dir : str
        Directory for output files.
    recovery_max_attempts : int
        Maximum consecutive recovery attempts before aborting.
    """

    duration_hours: float = 8.0
    max_steps: int = 0
    rank: int = 4
    rep_dim: int = 8
    batch_size: int = 4
    n_constraints: int = 3
    n_observers: int = 2
    learning_rate: float = 0.01
    entropy_weight: float = 0.01
    seed: int = 42
    checkpoint_interval: int = 10000
    log_interval: int = 1000
    output_dir: str = "ciir_long_run"
    recovery_max_attempts: int = 5


# ================================================================
# Metrics collection
# ================================================================


@dataclass
class StepRecord:
    """Single-step metrics record."""

    step: int
    timestamp_s: float
    loss: float
    gradient_norm: float
    purity: float
    entropy: float
    constraint_loss: float
    observer_loss: float
    entropy_loss: float
    step_time_ms: float
    max_commutator_norm: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to dictionary."""
        return {
            "step": self.step,
            "timestamp_s": round(self.timestamp_s, 3),
            "loss": self.loss,
            "gradient_norm": self.gradient_norm,
            "purity": self.purity,
            "entropy": self.entropy,
            "constraint_loss": self.constraint_loss,
            "observer_loss": self.observer_loss,
            "entropy_loss": self.entropy_loss,
            "step_time_ms": round(self.step_time_ms, 3),
            "max_commutator_norm": self.max_commutator_norm,
        }


@dataclass
class LongRunResult:
    """Complete result of a long-duration simulation run.

    Contains all per-step records, summary statistics, and anomaly log.
    """

    config: dict[str, Any] = field(default_factory=dict)
    records: list[StepRecord] = field(default_factory=list)
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    recovery_count: int = 0
    aborted: bool = False
    abort_reason: str = ""
    total_elapsed_s: float = 0.0

    @property
    def n_steps(self) -> int:
        """Total number of completed steps."""
        return len(self.records)

    @property
    def final_loss(self) -> float:
        """Loss at the last completed step."""
        return self.records[-1].loss if self.records else float("nan")

    def summary(self) -> dict[str, Any]:
        """Compute summary statistics from collected records."""
        if not self.records:
            return {"n_steps": 0}

        losses = np.array([r.loss for r in self.records])
        grads = np.array([r.gradient_norm for r in self.records])
        purities = np.array([r.purity for r in self.records])
        entropies = np.array([r.entropy for r in self.records])
        step_times = np.array([r.step_time_ms for r in self.records])

        s: dict[str, Any] = {
            "n_steps": self.n_steps,
            "total_elapsed_s": round(self.total_elapsed_s, 3),
            "throughput_steps_per_s": round(self.n_steps / max(self.total_elapsed_s, 1e-12), 1),
            "initial_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "min_loss": float(losses.min()),
            "loss_reduction_pct": round(
                (float(losses[0]) - float(losses[-1])) / max(abs(float(losses[0])), 1e-12) * 100,
                2,
            ),
            "mean_gradient_norm": round(float(grads.mean()), 6),
            "final_gradient_norm": round(float(grads[-1]), 6),
            "mean_purity": round(float(purities.mean()), 6),
            "final_purity": round(float(purities[-1]), 6),
            "mean_entropy": round(float(entropies.mean()), 6),
            "final_entropy": round(float(entropies[-1]), 6),
            "mean_step_time_ms": round(float(step_times.mean()), 3),
            "recovery_count": self.recovery_count,
            "n_anomalies": len(self.anomalies),
            "aborted": self.aborted,
        }

        # Convergence check (last 100 steps)
        if len(losses) >= 100:
            tail = losses[-100:]
            s["converged"] = bool(np.std(tail) < 1e-6)
            s["loss_std_tail100"] = round(float(np.std(tail)), 8)

        return s

    def to_dict(self) -> dict[str, Any]:
        """Serialise full result."""
        return {
            "config": self.config,
            "summary": self.summary(),
            "anomalies": self.anomalies,
            "records": [r.to_dict() for r in self.records],
        }

    def save_json(self, path: str) -> None:
        """Save full result as JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def save_checkpoint(self, path: str) -> None:
        """Save incremental checkpoint (summary + last 100 records)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "config": self.config,
            "summary": self.summary(),
            "anomalies": self.anomalies,
            "n_total_records": len(self.records),
            "recent_records": [r.to_dict() for r in self.records[-100:]],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


# ================================================================
# Anomaly detection and recovery
# ================================================================


def _detect_anomaly(record: StepRecord) -> str | None:
    """Check a step record for numerical anomalies.

    Returns a description string if an anomaly is detected, else None.
    """
    if np.isnan(record.loss):
        return f"NaN loss at step {record.step}"
    if np.isinf(record.loss):
        return f"Inf loss at step {record.step}"
    if np.isnan(record.gradient_norm):
        return f"NaN gradient at step {record.step}"
    if record.gradient_norm > 1e10:
        return f"Gradient explosion at step {record.step}: {record.gradient_norm:.2e}"
    if np.isnan(record.purity):
        return f"NaN purity at step {record.step}"
    return None


def _recover_engine(
    engine: CIIRSimulationEngine,
    config: LongRunConfig,
    attempt: int,
) -> CIIRSimulationEngine:
    """Attempt recovery by re-initialising the engine state.

    Uses a perturbed seed to generate fresh initial conditions while
    keeping the same theory (constraints, observers).

    Parameters
    ----------
    engine : CIIRSimulationEngine
        The current engine (may have corrupted state).
    config : LongRunConfig
        Original configuration.
    attempt : int
        Recovery attempt number (used to perturb seed).

    Returns
    -------
    CIIRSimulationEngine
        A fresh engine with re-initialised state.
    """
    recovery_seed = config.seed + 1000 * (attempt + 1)
    sim_config = SimulationConfig(
        rank=config.rank,
        rep_dim=config.rep_dim,
        batch_size=config.batch_size,
        n_constraints=config.n_constraints,
        n_observers=config.n_observers,
        n_steps=0,  # We drive the loop ourselves
        learning_rate=config.learning_rate,
        entropy_weight=config.entropy_weight,
        seed=recovery_seed,
        screenshot_interval=0,
        output_dir=config.output_dir,
        log_metrics=False,
        log_tensors=False,
    )
    new_engine = CIIRSimulationEngine(sim_config)
    new_engine.init()
    # Carry forward step counter
    new_engine.current_step = engine.current_step
    return new_engine


# ================================================================
# Main runner
# ================================================================


def run_long_duration(
    config: LongRunConfig | None = None,
    *,
    duration_hours: float | None = None,
    max_steps: int | None = None,
    seed: int | None = None,
    output_dir: str | None = None,
    verbose: bool = True,
) -> LongRunResult:
    """Run a long-duration CIIR simulation.

    Executes the CIIR → QuASIM → QRATUM pipeline continuously for
    ``duration_hours`` wall-clock time (or ``max_steps`` steps,
    whichever is reached first).

    Parameters
    ----------
    config : LongRunConfig, optional
        Full configuration. Individual overrides below take precedence.
    duration_hours : float, optional
        Override wall-clock duration in hours.
    max_steps : int, optional
        Override maximum step count.
    seed : int, optional
        Override random seed.
    output_dir : str, optional
        Override output directory.
    verbose : bool
        Print progress to console.

    Returns
    -------
    LongRunResult
        Full metrics, anomalies, and summary.
    """
    if config is None:
        config = LongRunConfig()
    if duration_hours is not None:
        config.duration_hours = duration_hours
    if max_steps is not None:
        config.max_steps = max_steps
    if seed is not None:
        config.seed = seed
    if output_dir is not None:
        config.output_dir = output_dir

    duration_s = config.duration_hours * 3600.0

    # Build simulation engine
    sim_config = SimulationConfig(
        rank=config.rank,
        rep_dim=config.rep_dim,
        batch_size=config.batch_size,
        n_constraints=config.n_constraints,
        n_observers=config.n_observers,
        n_steps=0,  # We drive the loop ourselves
        learning_rate=config.learning_rate,
        entropy_weight=config.entropy_weight,
        seed=config.seed,
        screenshot_interval=0,
        output_dir=config.output_dir,
        log_metrics=False,
        log_tensors=False,
    )
    engine = CIIRSimulationEngine(sim_config)
    engine.init()

    # Compute commutator matrix once
    max_comm_norm = 0.0
    if engine.ciir_observer is not None:
        comm_matrix = engine.ciir_observer.commutator_matrix()
        max_comm_norm = float(np.max(np.abs(comm_matrix)))

    result = LongRunResult(
        config={
            "duration_hours": config.duration_hours,
            "max_steps": config.max_steps,
            "rank": config.rank,
            "rep_dim": config.rep_dim,
            "batch_size": config.batch_size,
            "n_constraints": config.n_constraints,
            "n_observers": config.n_observers,
            "learning_rate": config.learning_rate,
            "entropy_weight": config.entropy_weight,
            "seed": config.seed,
            "checkpoint_interval": config.checkpoint_interval,
        },
    )

    os.makedirs(config.output_dir, exist_ok=True)
    consecutive_failures = 0
    t_start = time.time()

    if verbose:
        print("╔══════════════════════════════════════════════════╗")
        print("║  CIIR Long-Duration Simulation Runner            ║")
        print("╠══════════════════════════════════════════════════╣")
        print(f"║  Duration:   {config.duration_hours}h ({duration_s:.0f}s)")
        if config.max_steps > 0:
            print(f"║  Max steps:  {config.max_steps}")
        print(f"║  Manifold:   R={config.rank} D={config.rep_dim} B={config.batch_size}")
        print(f"║  Theory:     C={config.n_constraints} O={config.n_observers}")
        print(f"║  Seed:       {config.seed}")
        print(f"║  Output:     {config.output_dir}")
        print("╚══════════════════════════════════════════════════╝\n")

    step_idx = 0
    while True:
        elapsed = time.time() - t_start

        # Time limit check
        if duration_s > 0 and elapsed >= duration_s:
            break

        # Step limit check
        if config.max_steps > 0 and step_idx >= config.max_steps:
            break

        # Execute one step
        snapshot = engine.step()
        step_idx += 1

        record = StepRecord(
            step=snapshot.step,
            timestamp_s=elapsed,
            loss=snapshot.total_loss,
            gradient_norm=snapshot.gradient_norm,
            purity=snapshot.purity,
            entropy=snapshot.entropy,
            constraint_loss=snapshot.constraint_loss,
            observer_loss=snapshot.observer_loss,
            entropy_loss=snapshot.entropy_loss,
            step_time_ms=snapshot.step_time_ms,
            max_commutator_norm=max_comm_norm,
        )

        # Anomaly detection
        anomaly = _detect_anomaly(record)
        if anomaly is not None:
            result.anomalies.append({"step": record.step, "type": anomaly, "timestamp_s": elapsed})
            consecutive_failures += 1

            if verbose:
                print(f"  ⚠ Anomaly at step {record.step}: {anomaly}")

            if consecutive_failures >= config.recovery_max_attempts:
                result.aborted = True
                result.abort_reason = (
                    f"Exceeded {config.recovery_max_attempts} consecutive recovery attempts"
                )
                if verbose:
                    print(f"  ✗ Aborting: {result.abort_reason}")
                break

            # Attempt recovery
            if verbose:
                print(
                    f"  → Recovery attempt {consecutive_failures}/"
                    f"{config.recovery_max_attempts}"
                )
            result.recovery_count += 1
            engine = _recover_engine(engine, config, consecutive_failures)

            # Recompute commutator matrix after recovery
            if engine.ciir_observer is not None:
                comm_matrix = engine.ciir_observer.commutator_matrix()
                max_comm_norm = float(np.max(np.abs(comm_matrix)))
            continue
        else:
            consecutive_failures = 0

        result.records.append(record)

        # Console logging
        if verbose and step_idx % config.log_interval == 0:
            steps_per_s = step_idx / max(elapsed, 1e-12)
            print(
                f"  Step {record.step:>8d} | "
                f"loss={record.loss:>10.6f} | "
                f"‖∇L‖={record.gradient_norm:.2e} | "
                f"purity={record.purity:.4f} | "
                f"S={record.entropy:.4f} | "
                f"{steps_per_s:.0f} steps/s | "
                f"elapsed={elapsed:.1f}s"
            )

        # Incremental checkpoint
        if config.checkpoint_interval > 0 and step_idx % config.checkpoint_interval == 0:
            ckpt_path = os.path.join(config.output_dir, f"checkpoint_step{record.step}.json")
            result.save_checkpoint(ckpt_path)
            if verbose:
                print(f"  📁 Checkpoint saved: {ckpt_path}")

    result.total_elapsed_s = time.time() - t_start

    # Save final results
    final_path = os.path.join(config.output_dir, "final_results.json")
    result.save_json(final_path)

    if verbose:
        s = result.summary()
        print(f"\n{'=' * 50}")
        print("Long-Duration Simulation Complete")
        print(f"  Steps:       {s['n_steps']}")
        print(f"  Elapsed:     {s['total_elapsed_s']:.1f}s")
        print(f"  Throughput:  {s['throughput_steps_per_s']:.0f} steps/s")
        print(f"  Loss:        {s.get('initial_loss', 'N/A')} → {s.get('final_loss', 'N/A')}")
        print(f"  Recoveries:  {result.recovery_count}")
        print(f"  Anomalies:   {len(result.anomalies)}")
        print(f"  Aborted:     {result.aborted}")
        print(f"  Output:      {final_path}")

    return result


# ================================================================
# Summary plots
# ================================================================


def generate_long_run_plots(
    result: LongRunResult,
    output_dir: str | None = None,
) -> list[str]:
    """Generate summary plots from a long-duration simulation.

    Parameters
    ----------
    result : LongRunResult
        Completed simulation result.
    output_dir : str, optional
        Directory for plot files. Defaults to ``result.config["output_dir"]``.

    Returns
    -------
    list of str
        Paths to generated plot files.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = result.config.get("output_dir", "ciir_long_run")
    os.makedirs(output_dir, exist_ok=True)
    paths: list[str] = []

    if not result.records:
        return paths

    steps = [r.step for r in result.records]
    timestamps = [r.timestamp_s for r in result.records]

    # 1. Loss vs step
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, [r.loss for r in result.records], linewidth=0.5, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("CIIR Long-Duration: Loss vs Step")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(output_dir, "loss_vs_step.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # 2. Loss vs time
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        [t / 3600 for t in timestamps],
        [r.loss for r in result.records],
        linewidth=0.5,
        alpha=0.8,
    )
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Loss")
    ax.set_title("CIIR Long-Duration: Loss vs Time")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(output_dir, "loss_vs_time.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # 3. Purity vs step
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, [r.purity for r in result.records], linewidth=0.5, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Purity Tr(ρ²)")
    ax.set_title("CIIR Long-Duration: Purity vs Step")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(output_dir, "purity_vs_step.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # 4. Entropy vs step
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, [r.entropy for r in result.records], linewidth=0.5, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Entropy S(ρ)")
    ax.set_title("CIIR Long-Duration: Entropy vs Step")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(output_dir, "entropy_vs_step.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # 5. Constraint violations vs step
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        steps,
        [r.constraint_loss for r in result.records],
        linewidth=0.5,
        alpha=0.8,
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Constraint Loss")
    ax.set_title("CIIR Long-Duration: Constraint Violations vs Step")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(output_dir, "constraint_loss_vs_step.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # 6. Gradient norm (log) vs step
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(
        steps,
        [r.gradient_norm for r in result.records],
        linewidth=0.5,
        alpha=0.8,
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("‖∇L‖")
    ax.set_title("CIIR Long-Duration: Gradient Norm vs Step")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(output_dir, "gradient_norm_vs_step.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    return paths
