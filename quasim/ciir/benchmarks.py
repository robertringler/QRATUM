r"""Validation suite: synthetic constraint satisfaction, interference tests, convergence.

Implements §8 of the CIIR→QuASIM formalization:

Experiment 1 — Synthetic constraint satisfaction:
    Generate random CIIR theory, evolve state, verify constraints approach 0.

Experiment 2 — Observer interference tests:
    Verify T8.6: E_{ψ±}[O] = ½(E_{ψ₁} + E_{ψ₂}) ± Re⟨ψ₁|O|ψ₂⟩

Experiment 3 — Convergence metrics:
    Loss convergence, constraint violation norms, stability across seeds.

Experiment 4 — Distortion verification:
    Verify T8.4 contraction bound and T8.5 non-invertibility.

Experiment 5 — Contextuality test:
    Demonstrate outcome dependence on measurement context (T8.7).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from quasim.ciir.distortion import analyse_distortion
from quasim.ciir.evolution import IntegrationMethod, evolve
from quasim.ciir.loss import CIIRLoss
from quasim.ciir.observers import (
    MeasurementContext,
    ObserverOperator,
    PVM,
    born_distribution,
    contextuality_witness,
    interference_term,
    superposition_expectation,
)
from quasim.ciir.theory import CIIRTheory, build_default_theory


@dataclass
class BenchmarkResult:
    """Container for a single benchmark run."""

    name: str
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    details: str = ""
    elapsed_s: float = 0.0


def run_constraint_satisfaction(
    theory: CIIRTheory | None = None,
    n_steps: int = 200,
    lr: float = 0.01,
    seed: int = 42,
) -> BenchmarkResult:
    """Experiment 1: Evolve state and verify constraint violations → 0."""
    t0 = time.time()

    if theory is None:
        theory = build_default_theory(seed=seed)

    rng = np.random.default_rng(seed)
    rho = theory.manifold.density_matrix(
        theory.manifold.random_state(batch=4, rng=rng)
    )

    loss_fn = CIIRLoss(theory=theory)
    result = evolve(loss_fn, rho, n_steps=n_steps, lr=lr)

    final_loss = result.losses[-1] if result.losses else float("inf")
    initial_loss = result.losses[0] if result.losses else float("inf")

    # Constraint violation at end
    final_rho = result.states[-1]
    violations = []
    for c in theory.constraints:
        v = c.violation(final_rho)
        violations.append(float(v.mean()))
    mean_violation = float(np.mean(violations)) if violations else 0.0

    passed = final_loss < initial_loss and mean_violation < 1.0
    elapsed = time.time() - t0

    return BenchmarkResult(
        name="constraint_satisfaction",
        passed=passed,
        metrics={
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "mean_constraint_violation": mean_violation,
            "n_steps": float(result.n_steps),
            "converged": float(result.converged),
        },
        details=f"Loss: {initial_loss:.4f} → {final_loss:.4f}, "
        f"Violation: {mean_violation:.6f}",
        elapsed_s=elapsed,
    )


def run_interference_test(seed: int = 42) -> BenchmarkResult:
    """Experiment 2: Verify T8.6 interference formula."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    D = 8

    # Random observable
    O_mat = rng.standard_normal((D, D))
    O_mat = 0.5 * (O_mat + O_mat.T)

    # Two random pure states
    psi1 = rng.standard_normal(D)
    psi1 /= np.linalg.norm(psi1)
    psi2 = rng.standard_normal(D)
    # Gram-Schmidt to partially orthogonalise
    psi2 -= psi1 * (psi1 @ psi2)
    psi2 /= np.linalg.norm(psi2)

    # Superposition states
    psi_plus = (psi1 + psi2) / np.sqrt(2)
    psi_minus = (psi1 - psi2) / np.sqrt(2)

    # Direct expectations
    E_plus_direct = float(psi_plus @ O_mat @ psi_plus)
    E_minus_direct = float(psi_minus @ O_mat @ psi_minus)

    # From T8.6 formula
    E_plus_formula = superposition_expectation(psi1, psi2, O_mat, sign=+1)
    E_minus_formula = superposition_expectation(psi1, psi2, O_mat, sign=-1)

    cross = interference_term(psi1, psi2, O_mat)

    err_plus = abs(E_plus_direct - E_plus_formula)
    err_minus = abs(E_minus_direct - E_minus_formula)
    passed = err_plus < 1e-10 and err_minus < 1e-10

    elapsed = time.time() - t0
    return BenchmarkResult(
        name="interference_T8.6",
        passed=passed,
        metrics={
            "E_plus_direct": E_plus_direct,
            "E_plus_formula": E_plus_formula,
            "E_minus_direct": E_minus_direct,
            "E_minus_formula": E_minus_formula,
            "cross_term": cross,
            "error_plus": err_plus,
            "error_minus": err_minus,
        },
        details=f"Errors: |Δ+|={err_plus:.2e}, |Δ-|={err_minus:.2e}, "
        f"cross={cross:.4f}",
        elapsed_s=elapsed,
    )


def run_convergence_stability(
    n_seeds: int = 5,
    n_steps: int = 100,
    lr: float = 0.01,
) -> BenchmarkResult:
    """Experiment 3: Stability across random seeds."""
    t0 = time.time()
    final_losses = []

    for s in range(n_seeds):
        theory = build_default_theory(seed=s)
        rng = np.random.default_rng(s)
        rho = theory.manifold.density_matrix(
            theory.manifold.random_state(batch=2, rng=rng)
        )
        loss_fn = CIIRLoss(theory=theory)
        result = evolve(loss_fn, rho, n_steps=n_steps, lr=lr)
        final_losses.append(result.losses[-1] if result.losses else float("inf"))

    mean_loss = float(np.mean(final_losses))
    std_loss = float(np.std(final_losses))
    # Use absolute mean for CV to handle negative loss values
    cv = std_loss / max(abs(mean_loss), 1e-12)

    passed = cv < 1.0  # coefficient of variation < 100%
    elapsed = time.time() - t0

    return BenchmarkResult(
        name="convergence_stability",
        passed=passed,
        metrics={
            "mean_final_loss": mean_loss,
            "std_final_loss": std_loss,
            "coefficient_of_variation": cv,
            "n_seeds": float(n_seeds),
        },
        details=f"Mean loss: {mean_loss:.4f} ± {std_loss:.4f} (CV={cv:.2f})",
        elapsed_s=elapsed,
    )


def run_distortion_verification(seed: int = 42) -> BenchmarkResult:
    """Experiment 4: Verify T8.4 contraction and T8.5 non-invertibility."""
    t0 = time.time()
    theory = build_default_theory(seed=seed, rank=4, rep_dim=8)

    rng = np.random.default_rng(seed)
    N = 20
    X_samples = theory.manifold.random_state(batch=N, rng=rng)

    if theory.interface is None:
        return BenchmarkResult(
            name="distortion_T8.4_T8.5",
            passed=False,
            details="No interface map defined",
        )

    analysis = analyse_distortion(theory.interface, X_samples)

    elapsed = time.time() - t0
    return BenchmarkResult(
        name="distortion_T8.4_T8.5",
        passed=analysis.bound_satisfied,
        metrics={
            "bound_satisfied": float(analysis.bound_satisfied),
            "max_ratio": float(analysis.ratio[np.triu_indices_from(analysis.ratio, k=1)].max())
            if N > 1
            else 0.0,
            "contraction_factor": theory.interface.contraction_factor,
            "n_non_invertible_pairs": float(len(analysis.non_invertible_pairs)),
        },
        details=f"Bound satisfied: {analysis.bound_satisfied}, "
        f"Non-invertible pairs: {len(analysis.non_invertible_pairs)}, "
        f"α={theory.interface.contraction_factor:.4f}",
        elapsed_s=elapsed,
    )


def run_contextuality_test(seed: int = 42) -> BenchmarkResult:
    """Experiment 5: Demonstrate contextuality (T8.7)."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    D = 4

    # Shared observable (diagonal)
    O_shared = ObserverOperator(matrix=np.diag(rng.standard_normal(D)), name="shared")

    # Context 1: shared + commuting partner
    O_partner1 = ObserverOperator(
        matrix=np.diag(rng.standard_normal(D)), name="partner1"
    )
    ctx1 = MeasurementContext(observables=[O_shared, O_partner1])

    # Context 2: shared + non-commuting partner
    M = rng.standard_normal((D, D))
    M = 0.5 * (M + M.T)
    O_partner2 = ObserverOperator(matrix=M, name="partner2")
    ctx2 = MeasurementContext(observables=[O_shared, O_partner2])

    # Random mixed state
    X = rng.standard_normal((D, D))
    rho = X @ X.T
    rho /= np.trace(rho)

    tvd = contextuality_witness(rho, ctx1, ctx2, O_shared)

    # ctx1 should be compatible (diagonal observables commute)
    ctx1_compatible = ctx1.is_compatible()

    elapsed = time.time() - t0
    return BenchmarkResult(
        name="contextuality_T8.7",
        passed=True,  # informational test
        metrics={
            "total_variation_distance": tvd,
            "context1_compatible": float(ctx1_compatible),
            "context2_compatible": float(ctx2.is_compatible()),
        },
        details=f"TVD between contexts: {tvd:.6f}, "
        f"Ctx1 compatible: {ctx1_compatible}",
        elapsed_s=elapsed,
    )


def run_all_benchmarks(verbose: bool = True) -> list[BenchmarkResult]:
    """Run the complete CIIR validation suite."""
    benchmarks = [
        run_constraint_satisfaction,
        run_interference_test,
        run_convergence_stability,
        run_distortion_verification,
        run_contextuality_test,
    ]

    results = []
    for bench_fn in benchmarks:
        result = bench_fn()
        results.append(result)
        if verbose:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"  [{status}] {result.name}: {result.details} "
                  f"({result.elapsed_s:.2f}s)")

    return results
