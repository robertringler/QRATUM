"""Temporal Compression Experiment Module.

Implements the complete experimental framework for validating
deterministic temporal compression in computational systems.

Features:
- Configurable experimental parameters
- Multiple repetitions for statistical validity
- Automated metrics collection
- Cryptographic verification trail
- Comprehensive result reporting
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .automata import BranchingSimulator, CellularAutomaton
from .metrics import TemporalMetrics
from .verification import EventType, StateVerifier, VerificationChain


@dataclass
class ExperimentConfig:
    """Configuration for temporal compression experiment.

    Attributes:
        system_rule: Cellular automaton rule number
        system_size: Number of cells in automaton
        seed: Random seed for reproducibility
        simulated_steps: Total steps to simulate
        checkpoint_interval: Steps between checkpoints
        num_repetitions: Number of repetitions for verification
        time_unit_per_step: Simulated time per step (for ratio calculation)
        enable_branching: Whether to test branching simulations
        branch_points: Steps at which to create branches
        output_dir: Directory for results output
    """

    system_rule: int = 110
    system_size: int = 200
    seed: int = 42
    simulated_steps: int = 10000
    checkpoint_interval: int = 1000
    num_repetitions: int = 3
    time_unit_per_step: float = 1.0
    enable_branching: bool = False
    branch_points: list[int] = field(default_factory=lambda: [1000, 5000])
    output_dir: str = "artifacts/temporal_compression"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "system_rule": self.system_rule,
            "system_size": self.system_size,
            "seed": self.seed,
            "simulated_steps": self.simulated_steps,
            "checkpoint_interval": self.checkpoint_interval,
            "num_repetitions": self.num_repetitions,
            "time_unit_per_step": self.time_unit_per_step,
            "enable_branching": self.enable_branching,
            "branch_points": self.branch_points,
            "output_dir": self.output_dir,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ExperimentResult:
    """Results from temporal compression experiment.

    Attributes:
        config: Experiment configuration
        metrics: Temporal compression metrics
        verification_chain: Cryptographic verification chain
        final_state_hashes: Final state hashes from each run
        is_reproducible: Whether runs were bitwise identical
        compression_ratio: Mean temporal compression ratio
        execution_time: Total experiment execution time
        hypothesis_evaluation: Evaluation of hypotheses
    """

    config: ExperimentConfig
    metrics: TemporalMetrics
    verification_chain: VerificationChain
    final_state_hashes: list[str]
    is_reproducible: bool
    compression_ratio: float
    execution_time: float
    hypothesis_evaluation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "verification": self.verification_chain.export_for_verification(),
            "final_state_hashes": self.final_state_hashes,
            "is_reproducible": self.is_reproducible,
            "compression_ratio": self.compression_ratio,
            "execution_time": self.execution_time,
            "hypothesis_evaluation": self.hypothesis_evaluation,
        }


class TemporalCompressionExperiment:
    """Main experiment class for temporal compression validation.

    Implements the complete experimental methodology:
    1. Deterministic state evolution with fixed precision
    2. Forward temporal simulation with wall-clock timing
    3. Temporal compression ratio quantification
    4. Cryptographic verification of state transitions
    5. Reproducibility confirmation across runs
    6. Optional branching simulation exploration

    Example:
        config = ExperimentConfig(
            system_rule=110,
            system_size=200,
            simulated_steps=10000,
            num_repetitions=3,
        )
        experiment = TemporalCompressionExperiment(config)
        result = experiment.run()

        print(f"Compression ratio: {result.compression_ratio:.2f}×")
        print(f"Reproducible: {result.is_reproducible}")
    """

    def __init__(
        self,
        config: ExperimentConfig | None = None,
        verbose: bool = True,
    ) -> None:
        """Initialize experiment.

        Args:
            config: Experiment configuration
            verbose: Whether to print progress messages
        """
        self.config = config or ExperimentConfig()
        self.verbose = verbose

        self.metrics = TemporalMetrics()
        self.verification_chain = VerificationChain()
        self.state_verifier = StateVerifier()

        self._run_hashes: list[str] = []
        self._run_results: list[dict[str, Any]] = []

    def log(self, message: str) -> None:
        """Print log message if verbose."""
        if self.verbose:
            print(f"[EXPERIMENT] {message}")

    def run(self) -> ExperimentResult:
        """Execute the complete experiment.

        Returns:
            ExperimentResult with all metrics and verification data
        """
        experiment_start = time.time()
        self.log("Starting temporal compression experiment")
        self.log(
            f"Configuration: Rule {self.config.system_rule}, "
            f"Size {self.config.system_size}, "
            f"Steps {self.config.simulated_steps}"
        )

        # Record experiment start
        self.verification_chain.add_event(
            EventType.EXPERIMENT_START,
            step=0,
            state_hash="0" * 64,
            data={"config": self.config.to_dict()},
        )

        # Run multiple repetitions
        for rep in range(self.config.num_repetitions):
            self.log(f"Running repetition {rep + 1}/{self.config.num_repetitions}")
            run_result = self._run_single_experiment(rep)
            self._run_results.append(run_result)
            self._run_hashes.append(run_result["final_hash"])

        # Verify reproducibility
        is_reproducible = self.metrics.verify_reproducibility(self._run_hashes)
        self.log(f"Reproducibility verified: {is_reproducible}")

        # Record reproducibility check
        self.verification_chain.add_event(
            EventType.REPRODUCIBILITY_CHECK,
            step=self.config.simulated_steps,
            state_hash=self._run_hashes[0] if self._run_hashes else "0" * 64,
            data={
                "is_reproducible": is_reproducible,
                "num_runs": len(self._run_hashes),
                "unique_hashes": list(set(self._run_hashes)),
            },
        )

        # Optional branching exploration
        if self.config.enable_branching:
            self.log("Running branching exploration...")
            self._run_branching_exploration()

        # Record experiment end
        experiment_end = time.time()
        self.verification_chain.add_event(
            EventType.EXPERIMENT_END,
            step=self.config.simulated_steps,
            state_hash=self._run_hashes[-1] if self._run_hashes else "0" * 64,
            data={
                "total_time": experiment_end - experiment_start,
                "compression_ratio": self.metrics.get_compression_ratio(),
            },
        )

        # Evaluate hypotheses
        hypothesis_eval = self._evaluate_hypotheses()

        # Create result
        result = ExperimentResult(
            config=self.config,
            metrics=self.metrics,
            verification_chain=self.verification_chain,
            final_state_hashes=self._run_hashes,
            is_reproducible=is_reproducible,
            compression_ratio=self.metrics.get_compression_ratio(),
            execution_time=experiment_end - experiment_start,
            hypothesis_evaluation=hypothesis_eval,
        )

        self.log("Experiment complete!")
        self.log(f"Compression ratio: {result.compression_ratio:.2f}×")
        self.log(f"Total execution time: {result.execution_time:.2f}s")

        return result

    def _run_single_experiment(self, repetition: int) -> dict[str, Any]:
        """Run a single experiment repetition.

        Args:
            repetition: Repetition number

        Returns:
            Dictionary with run results
        """
        # Create fresh automaton with same seed
        automaton = CellularAutomaton(
            rule=self.config.system_rule,
            size=self.config.system_size,
            seed=self.config.seed,
            track_history=False,
        )

        # Start timing
        self.metrics.start_measurement()
        run_start = time.time()

        # Simulate with checkpoints
        steps_per_batch = self.config.checkpoint_interval
        total_steps = self.config.simulated_steps
        batches = total_steps // steps_per_batch

        for batch in range(batches):
            batch_start = time.time()

            # Evolve automaton
            automaton.evolve(steps_per_batch)

            # Record state hash for verification
            state_hash = automaton.get_state_hash()

            # Add checkpoint to verification chain (only first run)
            if repetition == 0:
                hash_start = time.time()
                self.verification_chain.add_checkpoint(
                    step=automaton.step,
                    state_hash=state_hash,
                    checkpoint_id=f"rep{repetition}_batch{batch}",
                    data={"batch": batch, "repetition": repetition},
                )
                self.metrics.record_merkle_time(time.time() - hash_start)

        # Handle remaining steps
        remaining = total_steps % steps_per_batch
        if remaining > 0:
            automaton.evolve(remaining)

        # End timing
        simulated_time = total_steps * self.config.time_unit_per_step
        measurement = self.metrics.end_measurement(
            steps=total_steps,
            simulated_time=simulated_time,
        )

        # Final state hash
        final_hash = automaton.get_state_hash()

        return {
            "repetition": repetition,
            "final_hash": final_hash,
            "final_step": automaton.step,
            "wall_time": measurement.wall_clock_duration,
            "compression_ratio": measurement.compression_ratio,
        }

    def _run_branching_exploration(self) -> None:
        """Run branching simulation exploration."""
        # Create base automaton
        base = CellularAutomaton(
            rule=self.config.system_rule,
            size=self.config.system_size,
            seed=self.config.seed,
        )

        branching = BranchingSimulator(base)

        for branch_step in self.config.branch_points:
            # Evolve base to branch point
            if base.step < branch_step:
                base.evolve(branch_step - base.step)

            # Create branches with different interventions
            branch_id = f"branch_{branch_step}"
            perturbation = [self.config.system_size // 2]  # Flip middle cell

            branching.create_branch(
                branch_id=branch_id,
                intervention=f"flip_cell_{perturbation[0]}",
                perturbation_indices=perturbation,
            )

            # Run branch forward
            steps_remaining = self.config.simulated_steps - branch_step
            if steps_remaining > 0:
                result = branching.run_branch(branch_id, steps_remaining)

                # Record branch in verification chain
                self.verification_chain.add_event(
                    EventType.BRANCH_EVOLVE,
                    step=result.final_step,
                    state_hash=result.final_state.state_hash,
                    data={
                        "branch_id": branch_id,
                        "divergence": result.divergence_measure,
                    },
                )

    def _evaluate_hypotheses(self) -> dict[str, Any]:
        """Evaluate experimental hypotheses.

        Returns H₀/H₁ evaluation based on results.
        """
        compression_ratio = self.metrics.get_compression_ratio()
        is_reproducible = self.metrics.statistics.reproducibility_verified
        std_ratio = self.metrics.statistics.std_compression_ratio

        # Threshold definitions
        MEANINGFUL_COMPRESSION = 10.0  # 10× faster than real-time
        CONSISTENCY_THRESHOLD = 0.1  # 10% variation acceptable

        # Evaluate H₀ (null hypothesis)
        h0_supported = (
            compression_ratio < MEANINGFUL_COMPRESSION
            or not is_reproducible
            or (std_ratio / max(compression_ratio, 1e-6)) > CONSISTENCY_THRESHOLD
        )

        # Evaluate H₁ (alternative hypothesis)
        h1_supported = (
            compression_ratio >= MEANINGFUL_COMPRESSION
            and is_reproducible
            and (std_ratio / max(compression_ratio, 1e-6)) <= CONSISTENCY_THRESHOLD
        )

        # Determine conclusion
        if h1_supported:
            conclusion = "REJECT_H0"
            interpretation = (
                f"Deterministic temporal simulation achieved {compression_ratio:.2f}× "
                f"temporal compression with verified reproducibility and auditable results. "
                f"H₁ is supported."
            )
        elif h0_supported:
            conclusion = "FAIL_TO_REJECT_H0"
            if compression_ratio >= MEANINGFUL_COMPRESSION and is_reproducible:
                # High compression achieved but with high variance
                cv = std_ratio / max(compression_ratio, 1e-6)
                interpretation = (
                    f"Temporal compression ratio of {compression_ratio:.2f}× achieved with "
                    f"verified reproducibility. However, coefficient of variation ({cv:.2%}) "
                    f"exceeds threshold ({CONSISTENCY_THRESHOLD:.0%}). "
                    f"Additional runs recommended to reduce variance."
                )
            else:
                interpretation = (
                    f"Temporal compression ratio of {compression_ratio:.2f}× does not provide "
                    f"statistically meaningful advantage or lacks reproducibility. "
                    f"H₀ cannot be rejected."
                )
        else:
            conclusion = "INCONCLUSIVE"
            interpretation = (
                f"Results are inconclusive. Compression ratio: {compression_ratio:.2f}×, "
                f"Reproducible: {is_reproducible}. Additional investigation needed."
            )

        return {
            "compression_ratio": compression_ratio,
            "is_reproducible": is_reproducible,
            "std_ratio": std_ratio,
            "thresholds": {
                "meaningful_compression": MEANINGFUL_COMPRESSION,
                "consistency_threshold": CONSISTENCY_THRESHOLD,
            },
            "h0_supported": h0_supported,
            "h1_supported": h1_supported,
            "conclusion": conclusion,
            "interpretation": interpretation,
        }

    def save_results(
        self,
        result: ExperimentResult,
        output_dir: str | Path | None = None,
    ) -> Path:
        """Save experiment results to files.

        Args:
            result: Experiment result to save
            output_dir: Output directory (uses config default if None)

        Returns:
            Path to output directory
        """
        output_path = Path(output_dir or self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save main results
        results_file = output_path / "experiment_results.json"
        with open(results_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        self.log(f"Results saved to {results_file}")

        # Save verification chain separately
        chain_file = output_path / "verification_chain.json"
        with open(chain_file, "w") as f:
            json.dump(result.verification_chain.export_for_verification(), f, indent=2)
        self.log(f"Verification chain saved to {chain_file}")

        # Save summary metrics
        metrics_file = output_path / "metrics_summary.json"
        with open(metrics_file, "w") as f:
            json.dump(result.metrics.get_summary(), f, indent=2)
        self.log(f"Metrics saved to {metrics_file}")

        return output_path

    def generate_report(self, result: ExperimentResult) -> str:
        """Generate markdown report from results.

        Args:
            result: Experiment result

        Returns:
            Markdown report string
        """
        report_lines = [
            "# Temporal Compression Experiment Report",
            "",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "**Platform:** QRATUM Temporal Compression Framework v1.0",
            "",
            "## Executive Summary",
            "",
            f"This experiment validates deterministic temporal compression using a "
            f"Rule {self.config.system_rule} cellular automaton with {self.config.system_size} cells.",
            "",
            f"**Key Result:** Temporal compression ratio of **{result.compression_ratio:.2f}×** "
            f"achieved with {'verified' if result.is_reproducible else 'unverified'} reproducibility.",
            "",
            "## Configuration",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| System Rule | {self.config.system_rule} |",
            f"| System Size | {self.config.system_size} cells |",
            f"| Simulated Steps | {self.config.simulated_steps:,} |",
            f"| Repetitions | {self.config.num_repetitions} |",
            f"| Random Seed | {self.config.seed} |",
            "",
            "## Results",
            "",
            "### Temporal Compression Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Mean Compression Ratio | {result.metrics.statistics.mean_compression_ratio:.2f}× |",
            f"| Std. Dev. | ±{result.metrics.statistics.std_compression_ratio:.2f}× |",
            f"| Min Ratio | {result.metrics.statistics.min_compression_ratio:.2f}× |",
            f"| Max Ratio | {result.metrics.statistics.max_compression_ratio:.2f}× |",
            "",
            "### Verification",
            "",
            f"- **Chain Integrity:** {'✅ Valid' if result.verification_chain.verify_integrity() else '❌ Invalid'}",
            f"- **Reproducibility:** {'✅ Verified' if result.is_reproducible else '❌ Not Verified'}",
            f"- **Audit Entries:** {len(result.verification_chain.entries)}",
            "",
            "### Hypothesis Evaluation",
            "",
            f"**Conclusion:** {result.hypothesis_evaluation['conclusion']}",
            "",
            f"{result.hypothesis_evaluation['interpretation']}",
            "",
            "## Verification Data",
            "",
            "Final state hashes from each repetition:",
            "",
        ]

        for i, hash_val in enumerate(result.final_state_hashes):
            report_lines.append(f"- Run {i + 1}: `{hash_val[:16]}...`")

        report_lines.extend(
            [
                "",
                f"**Chain Proof:** `{result.verification_chain.get_chain_proof()[:32]}...`",
                "",
                "---",
                "",
                "*This report was generated by the QRATUM Temporal Compression Framework.*",
            ]
        )

        return "\n".join(report_lines)


def run_quick_validation() -> bool:
    """Run a quick validation experiment.

    Returns:
        True if validation passes
    """
    config = ExperimentConfig(
        system_size=50,
        simulated_steps=1000,
        num_repetitions=2,
        checkpoint_interval=100,
    )

    experiment = TemporalCompressionExperiment(config, verbose=False)
    result = experiment.run()

    return result.is_reproducible and result.compression_ratio > 0
