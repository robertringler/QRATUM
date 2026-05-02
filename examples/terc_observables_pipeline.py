#!/usr/bin/env python3
"""TERC Observables Pipeline Example

Demonstrates end-to-end pipeline for extracting TERC observables
from REVULTRA and QGH algorithms.
"""

import json

import numpy as np

from quasim.terc_bridge import (beta_metrics_from_cipher, emergent_complexity,
                                ioc_period_candidates, qgh_consensus_status)
from quasim.terc_bridge.registry import list_observables


def tier1_validation_pipeline():
    """Tier-1 validation: Initial state checks."""

    print("=" * 70)
    print("TERC Tier-1 Validation Pipeline")
    print("=" * 70)

    # Simulate QuASIM state as ciphertext
    quasim_state = "QUANTUMSTATE" * 20  # 220 chars

    print(f"Input state length: {len(quasim_state)} characters")
    print()

    # Extract beta metrics
    print("1. Beta Metrics (State Surrogates)")
    print("-" * 40)
    beta = beta_metrics_from_cipher(quasim_state)
    for key, value in beta.items():
        print(f"  {key}: {value:.4f}")
    print()

    # Extract complexity
    print("2. Emergent Complexity")
    print("-" * 40)
    complexity = emergent_complexity(quasim_state)
    print(f"  Score: {complexity['score']:.2f}")
    print(f"  Entropy: {complexity['entropy']:.4f}")
    print()

    # Detect periods
    print("3. Period Candidates")
    print("-" * 40)
    periods = ioc_period_candidates(quasim_state, max_period=20)
    print(f"  Detected periods: {periods}")
    print()

    # Tier-1 validation checks
    print("4. Tier-1 Validation Checks")
    print("-" * 40)

    checks = {
        "coherence": beta["beta_coherence"] > 0.5,
        "entropy": 0.3 < beta["beta_entropy"] < 0.9,
        "complexity": complexity["score"] < 80,
    }

    for check, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check:15s}: {status}")

    tier1_passed = all(checks.values())
    print(f"\nTier-1 Result: {'✓ PASSED' if tier1_passed else '✗ FAILED'}")
    print()

    return {
        "tier": 1,
        "passed": tier1_passed,
        "observables": {
            "beta_metrics": beta,
            "complexity": complexity,
            "periods": periods,
        },
        "checks": checks,
    }


def tier5_robustness_pipeline():
    """Tier-5 validation: Robustness and stability."""

    print("=" * 70)
    print("TERC Tier-5 Robustness Pipeline")
    print("=" * 70)

    # Simulate distributed system with 20 nodes
    num_nodes = 20
    state_dim = 10

    print(f"Testing {num_nodes} nodes, dimension {state_dim}")
    print()

    # Generate distributed states
    rng = np.random.default_rng(42)
    initial_states = rng.normal(0, 1, size=(num_nodes, state_dim))

    # Extract consensus observables
    print("1. Consensus Status")
    print("-" * 40)
    consensus = qgh_consensus_status(initial_states, damping=0.5, max_iterations=100)
    print(f"  Converged: {consensus['converged']}")
    print(f"  Iterations: {consensus['iterations']}")
    print(f"  Stability: {consensus['stability']:.4f}")
    print(f"  Robustness: {consensus['robustness']:.4f}")
    print(f"  Final variance: {consensus['final_variance']:.6f}")
    print()

    # Tier-5 validation checks
    print("2. Tier-5 Robustness Checks")
    print("-" * 40)

    checks = {
        "convergence": consensus["converged"],
        "stability": consensus["stability"] > 0.7,
        "robustness": consensus["robustness"] > 0.5,
        "variance": consensus["final_variance"] < 0.1,
    }

    for check, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check:15s}: {status}")

    tier5_passed = all(checks.values())
    print(f"\nTier-5 Result: {'✓ PASSED' if tier5_passed else '✗ FAILED'}")
    print()

    return {
        "tier": 5,
        "passed": tier5_passed,
        "observables": {"consensus_status": consensus},
        "checks": checks,
    }


def complete_validation_pipeline():
    """Complete TERC validation pipeline."""

    print("=" * 70)
    print("Complete TERC Validation Pipeline")
    print("=" * 70)
    print()

    # List available observables
    print("Available Observables:")
    print("-" * 40)
    observables = list_observables()
    for obs in observables:
        print(f"  • {obs}")
    print()

    # Run both tiers
    tier1_results = tier1_validation_pipeline()
    tier5_results = tier5_robustness_pipeline()

    # Overall assessment
    print("=" * 70)
    print("Overall TERC Assessment")
    print("=" * 70)

    overall_passed = tier1_results["passed"] and tier5_results["passed"]

    print(f"Tier-1 (Initial State): {'✓ PASSED' if tier1_results['passed'] else '✗ FAILED'}")
    print(f"Tier-5 (Robustness):    {'✓ PASSED' if tier5_results['passed'] else '✗ FAILED'}")
    print()
    print(f"Overall Status: {'✓ SYSTEM VALIDATED' if overall_passed else '✗ VALIDATION FAILED'}")
    print()

    # Export complete report
    report = {
        "validation_complete": overall_passed,
        "tier1": tier1_results,
        "tier5": tier5_results,
    }

    output_file = "terc_validation_report.json"
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)

    print(f"Complete report exported to {output_file}")
    print()

    return report


def observable_extraction_examples():
    """Show examples of extracting individual observables."""

    print("=" * 70)
    print("Observable Extraction Examples")
    print("=" * 70)
    print()

    # Example 1: From ciphertext
    print("Example 1: From Ciphertext")
    print("-" * 40)
    ciphertext = "ATTACKATDAWN" * 10
    beta = beta_metrics_from_cipher(ciphertext)
    print(f"Beta entropy: {beta['beta_entropy']:.4f}")
    print(f"Beta complexity: {beta['beta_complexity']:.4f}")
    print()

    # Example 2: Period detection
    print("Example 2: Period Detection")
    print("-" * 40)
    periodic_text = "ABC" * 30
    periods = ioc_period_candidates(periodic_text, max_period=10)
    print(f"Detected periods in 'ABC' * 30: {periods}")
    print()

    # Example 3: Consensus from random states
    print("Example 3: Consensus Metrics")
    print("-" * 40)
    states = np.random.rand(5, 3)
    consensus = qgh_consensus_status(states)
    print(f"Converged: {consensus['converged']}")
    print(f"Stability: {consensus['stability']:.4f}")
    print()


def main():
    """Run complete pipeline."""

    print("\n")
    print("#" * 70)
    print("# TERC Observables Pipeline Demonstration")
    print("#" * 70)
    print("\n")

    # Show individual observable examples
    observable_extraction_examples()

    # Run complete validation
    report = complete_validation_pipeline()

    print("=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)
    print()
    print("Key Results:")
    print(
        f"  • Tier-1 checks: {sum(report['tier1']['checks'].values())}/{len(report['tier1']['checks'])} passed"
    )
    print(
        f"  • Tier-5 checks: {sum(report['tier5']['checks'].values())}/{len(report['tier5']['checks'])} passed"
    )
    print(f"  • Overall: {'VALIDATED ✓' if report['validation_complete'] else 'FAILED ✗'}")
    print()


if __name__ == "__main__":
    main()
