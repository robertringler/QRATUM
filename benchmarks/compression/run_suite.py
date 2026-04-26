#!/usr/bin/env python3
"""MERA Compression Benchmark Suite Orchestrator.

This script orchestrates comprehensive benchmarking of QRATUM's MERA tensor
compression algorithm across diverse quantum state types.

Usage:
    # Run with default configuration
    python run_suite.py

    # Specify custom configuration
    python run_suite.py --config test_cases.yaml

    # Specify output directory
    python run_suite.py --output artifacts/compression/

    # Run specific qubit sizes
    python run_suite.py --qubits 10,15,20

    # Enable verbose logging
    python run_suite.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Add repository root to path for imports
_script_dir = Path(__file__).parent.absolute()
_repo_root = _script_dir.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Import generators and validators
from benchmarks.compression.generators.quantum_states import (  # noqa: E402
    generate_ghz_state,
    generate_product_state,
    generate_random_circuit_state,
    generate_random_state,
    generate_w_state,
)
from benchmarks.compression.validators.compression_ratio import (  # noqa: E402
    aggregate_compression_statistics,
    validate_compression_claim,
)
from benchmarks.compression.validators.fidelity import (  # noqa: E402
    validate_fidelity_bound,
)

# Import AHTC compression
try:
    from quasim.holo.anti_tensor import compress  # noqa: E402

    AHTC_AVAILABLE = True
except ImportError:
    AHTC_AVAILABLE = False
    warnings.warn("AHTC module not available. Benchmarks may fail.", UserWarning)


class BenchmarkResult:
    """Container for benchmark test results."""

    def __init__(
        self,
        test_name: str,
        state_type: str,
        n_qubits: int,
        compression_ratio: float,
        fidelity: float,
        runtime: float,
        metadata: dict[str, Any],
    ):
        self.test_name = test_name
        self.state_type = state_type
        self.n_qubits = n_qubits
        self.compression_ratio = compression_ratio
        self.fidelity = fidelity
        self.runtime = runtime
        self.metadata = metadata
        self.success = True
        self.error = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""

        return {
            "test_name": self.test_name,
            "state_type": self.state_type,
            "n_qubits": self.n_qubits,
            "compression_ratio": self.compression_ratio,
            "fidelity": self.fidelity,
            "runtime": self.runtime,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }


class BenchmarkOrchestrator:
    """Orchestrates benchmark execution and result collection."""

    def __init__(self, config: dict[str, Any], output_dir: Path, verbose: bool = False):
        self.config = config
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.results: list[BenchmarkResult] = []

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track statistics
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

    def run_all_tests(self) -> None:
        """Execute all test cases defined in configuration."""

        test_cases = self.config.get("test_cases", [])

        print(f"\n{'=' * 70}")
        print("MERA Compression Benchmark Suite")
        print(f"{'=' * 70}")
        print(f"Output directory: {self.output_dir}")
        print(f"Total test case groups: {len(test_cases)}")
        print(f"{'=' * 70}\n")

        for test_case in test_cases:
            self._run_test_case(test_case)

        print(f"\n{'=' * 70}")
        print("Benchmark Complete!")
        print(f"Total tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"{'=' * 70}\n")

    def _run_test_case(self, test_case: dict[str, Any]) -> None:
        """Run a single test case group."""

        name = test_case.get("name", "unnamed")
        generator = test_case.get("generator")
        qubits_list = test_case.get("qubits", [10])

        print(f"\n--- Test Case: {name} ---")
        print(f"Generator: {generator}")
        print(f"Qubit sizes: {qubits_list}")

        for n_qubits in qubits_list:
            # Handle different generator configurations
            if generator == "random":
                seeds = test_case.get("seeds", [42])
                for seed in seeds:
                    self._run_single_test(
                        test_case, n_qubits, generator, extra_params={"seed": seed}
                    )

            elif generator in ["ghz", "w"]:
                self._run_single_test(test_case, n_qubits, generator)

            elif generator == "product":
                product_type = test_case.get("product_type", "zero")
                self._run_single_test(
                    test_case,
                    n_qubits,
                    generator,
                    extra_params={"product_type": product_type},
                )

            elif generator == "circuit":
                depth = test_case.get("circuit_depth", 10)
                seeds = test_case.get("seeds", [42])
                for seed in seeds:
                    self._run_single_test(
                        test_case,
                        n_qubits,
                        generator,
                        extra_params={"depth": depth, "seed": seed},
                    )

            else:
                print(f"  ⚠️  Unknown generator: {generator}")

    def _run_single_test(
        self,
        test_case: dict[str, Any],
        n_qubits: int,
        generator: str,
        extra_params: dict[str, Any] | None = None,
    ) -> None:
        """Run a single benchmark test."""

        self.total_tests += 1
        extra_params = extra_params or {}

        # Generate test identifier
        test_id = self._generate_test_id(test_case.get("name", "test"), n_qubits, extra_params)

        try:
            # Generate quantum state
            state = self._generate_state(generator, n_qubits, extra_params)

            if state is None:
                print(f"  ⚠️  Test {test_id}: State generation failed (skipped)")
                self.failed_tests += 1
                return

            # Get compression parameters
            fidelity_target = float(test_case.get("fidelity_target", 0.995))
            epsilon_list = test_case.get("epsilon", [1e-3])

            # Handle single epsilon vs list
            if not isinstance(epsilon_list, list):
                epsilon_list = [epsilon_list]

            # Test with first epsilon (can extend to test all)
            epsilon = float(epsilon_list[0])

            # Run compression benchmark
            start_time = time.time()
            compressed, fidelity_achieved, metadata = compress(
                state, fidelity=fidelity_target, epsilon=epsilon
            )
            runtime = time.time() - start_time

            # Validate fidelity
            fidelity_valid = validate_fidelity_bound(fidelity_achieved, fidelity_target)

            # Compute metrics
            compression_ratio = metadata.get("compression_ratio", 0.0)

            # Create result
            result = BenchmarkResult(
                test_name=test_id,
                state_type=generator,
                n_qubits=n_qubits,
                compression_ratio=compression_ratio,
                fidelity=fidelity_achieved,
                runtime=runtime,
                metadata=metadata,
            )

            self.results.append(result)
            self.passed_tests += 1

            # Save artifact
            self._save_artifact(test_id, state, compressed, metadata)

            # Print result
            status_icon = "✅" if fidelity_valid else "⚠️"
            print(
                f"  {status_icon} {test_id}: "
                f"{compression_ratio:.2f}× @ F={fidelity_achieved:.4f} "
                f"({runtime:.3f}s)"
            )

        except Exception as e:
            print(f"  ❌ Test {test_id}: FAILED - {e}")
            if self.verbose:
                import traceback

                traceback.print_exc()
            self.failed_tests += 1

            # Create failed result
            result = BenchmarkResult(
                test_name=test_id,
                state_type=generator,
                n_qubits=n_qubits,
                compression_ratio=0.0,
                fidelity=0.0,
                runtime=0.0,
                metadata={},
            )
            result.success = False
            result.error = str(e)
            self.results.append(result)

    def _generate_state(
        self, generator: str, n_qubits: int, params: dict[str, Any]
    ) -> np.ndarray | None:
        """Generate quantum state based on generator type."""

        try:
            if generator == "random":
                seed = params.get("seed")
                return generate_random_state(n_qubits, seed=seed)

            elif generator == "ghz":
                return generate_ghz_state(n_qubits)

            elif generator == "w":
                return generate_w_state(n_qubits)

            elif generator == "product":
                product_type = params.get("product_type", "zero")
                return generate_product_state(n_qubits, state_type=product_type)

            elif generator == "circuit":
                depth = params.get("depth", 10)
                seed = params.get("seed")
                return generate_random_circuit_state(n_qubits, depth=depth, seed=seed)

            else:
                warnings.warn(f"Unknown generator: {generator}", UserWarning)
                return None

        except Exception as e:
            warnings.warn(f"State generation failed: {e}", UserWarning)
            return None

    def _generate_test_id(self, base_name: str, n_qubits: int, params: dict[str, Any]) -> str:
        """Generate unique test identifier."""

        parts = [base_name.replace(" ", "_"), f"{n_qubits}q"]

        if "seed" in params:
            parts.append(f"seed{params['seed']}")
        if "depth" in params:
            parts.append(f"d{params['depth']}")
        if "product_type" in params:
            parts.append(params["product_type"])

        return "_".join(parts)

    def _save_artifact(
        self, test_id: str, original: np.ndarray, compressed: np.ndarray, metadata: dict
    ) -> None:
        """Save test artifact to .npz file."""

        artifact_path = self.output_dir / f"{test_id}.npz"

        np.savez_compressed(
            artifact_path,
            original=original,
            compressed=compressed,
            metadata=np.array([metadata], dtype=object),
        )

        if self.verbose:
            print(f"    Artifact saved: {artifact_path}")

    def generate_summary(self) -> dict[str, Any]:
        """Generate summary statistics from all results."""

        if not self.results:
            return {"error": "No results to summarize"}

        # Extract metrics
        compression_ratios = [r.compression_ratio for r in self.results if r.success]
        fidelities = [r.fidelity for r in self.results if r.success]
        runtimes = [r.runtime for r in self.results if r.success]

        # Compute statistics
        if compression_ratios:
            ratio_stats = aggregate_compression_statistics(compression_ratios)
            fidelity_stats = aggregate_compression_statistics(fidelities)
            runtime_stats = aggregate_compression_statistics(runtimes)
        else:
            ratio_stats = {}
            fidelity_stats = {}
            runtime_stats = {}

        # Group by state type
        by_state_type: dict[str, list[BenchmarkResult]] = {}
        for result in self.results:
            if result.success:
                state_type = result.state_type
                if state_type not in by_state_type:
                    by_state_type[state_type] = []
                by_state_type[state_type].append(result)

        # Compute per-type statistics
        state_type_summary = {}
        for state_type, results in by_state_type.items():
            ratios = [r.compression_ratio for r in results]
            fids = [r.fidelity for r in results]
            state_type_summary[state_type] = {
                "count": len(results),
                "mean_ratio": float(np.mean(ratios)),
                "std_ratio": float(np.std(ratios)),
                "mean_fidelity": float(np.mean(fids)),
                "std_fidelity": float(np.std(fids)),
            }

        summary = {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "compression_ratio_stats": ratio_stats,
            "fidelity_stats": fidelity_stats,
            "runtime_stats": runtime_stats,
            "by_state_type": state_type_summary,
            "results": [r.to_dict() for r in self.results],
        }

        return summary

    def save_summary_json(self, filename: str = "compression_summary.json") -> None:
        """Save summary to JSON file."""

        summary = self.generate_summary()
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Summary saved to: {output_path}")

    def generate_markdown_report(self) -> str:
        """Generate markdown report."""

        summary = self.generate_summary()

        # Extract statistics
        ratio_stats = summary.get("compression_ratio_stats", {})
        fidelity_stats = summary.get("fidelity_stats", {})
        runtime_stats = summary.get("runtime_stats", {})

        # Build report
        report_lines = [
            "# MERA Compression Benchmark Results",
            "",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "**Platform:** QRATUM v2.0.0",
            f"**Test Cases:** {summary.get('total_tests', 0)} "
            f"({summary.get('passed_tests', 0)} passed, "
            f"{summary.get('failed_tests', 0)} failed)",
            "",
            "## Summary Metrics",
            "",
            "| Metric | Min | Median | Max | Mean ± Std |",
            "|--------|-----|--------|-----|------------|",
        ]

        # Compression ratio row
        if ratio_stats:
            report_lines.append(
                f"| Compression Ratio | {ratio_stats.get('min', 0):.1f}× | "
                f"{ratio_stats.get('median', 0):.1f}× | "
                f"{ratio_stats.get('max', 0):.1f}× | "
                f"{ratio_stats.get('mean', 0):.1f}× ± "
                f"{ratio_stats.get('std', 0):.1f}× |"
            )

        # Fidelity row
        if fidelity_stats:
            report_lines.append(
                f"| Fidelity | {fidelity_stats.get('min', 0):.4f} | "
                f"{fidelity_stats.get('median', 0):.4f} | "
                f"{fidelity_stats.get('max', 0):.4f} | "
                f"{fidelity_stats.get('mean', 0):.4f} ± "
                f"{fidelity_stats.get('std', 0):.4f} |"
            )

        # Runtime row
        if runtime_stats:
            report_lines.append(
                f"| Runtime (s) | {runtime_stats.get('min', 0):.2f} | "
                f"{runtime_stats.get('median', 0):.2f} | "
                f"{runtime_stats.get('max', 0):.2f} | "
                f"{runtime_stats.get('mean', 0):.2f} ± "
                f"{runtime_stats.get('std', 0):.2f} |"
            )

        # Per-state-type summary
        report_lines.extend(
            [
                "",
                "## Results by State Type",
                "",
            ]
        )

        by_state_type = summary.get("by_state_type", {})
        for state_type, stats in by_state_type.items():
            report_lines.extend(
                [
                    f"### {state_type.upper()} States ({stats.get('count', 0)} tests)",
                    "",
                    f"- **Mean Compression:** {stats.get('mean_ratio', 0):.2f}× "
                    f"± {stats.get('std_ratio', 0):.2f}×",
                    f"- **Mean Fidelity:** {stats.get('mean_fidelity', 0):.4f} "
                    f"± {stats.get('std_fidelity', 0):.4f}",
                    "",
                ]
            )

        # Validation against claims
        report_lines.extend(
            [
                "## Comparison to Documented Claims",
                "",
                "| Claim Source | Claimed Ratio | Measured Median | Status |",
                "|--------------|---------------|-----------------|--------|",
            ]
        )

        # Check against documented claims from config
        validation_config = self.config.get("validation", {})
        claims = validation_config.get("documented_claims", [])
        measured_median = ratio_stats.get("median", 0.0)

        for claim in claims:
            source = claim.get("source", "Unknown")
            claimed = claim.get("claimed_ratio", 0.0)
            claimed_min = claim.get("claimed_min", 0.0)
            claimed_max = claim.get("claimed_max", 100.0)

            validation = validate_compression_claim(
                measured_median, claimed_min, claimed_max, claimed
            )

            status_icons = {
                "VALIDATED": "✅",
                "WITHIN_RANGE": "⚠️",
                "BELOW_MIN": "❌",
                "EXCEEDS_MAX": "⚠️",
            }
            status = validation.get("status", "UNKNOWN")
            icon = status_icons.get(status, "❓")

            report_lines.append(
                f"| {source} | {claimed:.1f}× | {measured_median:.1f}× | {icon} {status} |"
            )

        # Validation status
        report_lines.extend(
            [
                "",
                "## Validation Status",
                "",
            ]
        )

        # Count how many tests meet various criteria
        successful = [r for r in self.results if r.success]
        high_fidelity = [r for r in successful if r.fidelity >= 0.995]
        good_ratio = [r for r in successful if 10.0 <= r.compression_ratio <= 50.0]

        report_lines.extend(
            [
                f"- ✅ Fidelity ≥ 0.995: {len(high_fidelity)}/{len(successful)} "
                f"({100 * len(high_fidelity) / max(len(successful), 1):.1f}%)",
                f"- ✅ Compression 10-50×: {len(good_ratio)}/{len(successful)} "
                f"({100 * len(good_ratio) / max(len(successful), 1):.1f}%)",
                "",
                "## Raw Data",
                "",
                f"See `{self.output_dir}/*.npz` for individual test results.",
                "",
            ]
        )

        return "\n".join(report_lines)

    def save_markdown_report(self, filename: str = "compression_report.md") -> None:
        """Save markdown report to file."""

        report = self.generate_markdown_report()
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            f.write(report)

        print(f"✅ Report saved to: {output_path}")


def load_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from YAML file."""

    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> int:
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="MERA Compression Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "test_cases.yaml",
        help="Path to test configuration YAML file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/compression"),
        help="Output directory for artifacts and reports",
    )

    parser.add_argument(
        "--qubits",
        type=str,
        help="Comma-separated list of qubit counts to test (overrides config)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Check AHTC availability
    if not AHTC_AVAILABLE:
        print("❌ ERROR: AHTC module not available. Cannot run benchmarks.")
        print("   Check that quasim.holo.anti_tensor is properly installed.")
        return 1

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"❌ ERROR: Configuration file not found: {args.config}")
        return 1
    except yaml.YAMLError as e:
        print(f"❌ ERROR: Invalid YAML configuration: {e}")
        return 1

    # Override qubits if specified
    if args.qubits:
        qubits_list = [int(q.strip()) for q in args.qubits.split(",")]
        # Update all test cases
        for test_case in config.get("test_cases", []):
            test_case["qubits"] = qubits_list

    # Create orchestrator and run benchmarks
    orchestrator = BenchmarkOrchestrator(config, args.output, verbose=args.verbose)

    try:
        orchestrator.run_all_tests()
        orchestrator.save_summary_json()
        orchestrator.save_markdown_report()

        # Return success if any tests passed
        return 0 if orchestrator.passed_tests > 0 else 1

    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        return 130

    except Exception as e:
        print(f"\n❌ ERROR: Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
