"""CIIR simulation experiments (§III of the mega-prompt specification).

Each experiment returns a plain-dict result that the report module
formats into the final structured output.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quasim.ciir.core import (
    density_matrix,
    measurement_probability,
    normalize,
)
from quasim.ciir.engine import CIIRSimulationEngine, SimulationConfig
from quasim.ciir.operators import frobenius_norm

# ------------------------------------------------------------------
# Experiment 1: Order Effect
# ------------------------------------------------------------------


def experiment_order_effect(
    cfg: SimulationConfig | None = None,
) -> dict[str, Any]:
    """Track Δ_t = P(A→B) − P(B→A) over time.

    Returns:
        Dict with ``delta_trajectory``, ``mean``, ``variance``,
        ``is_nonstationary``.
    """
    engine = CIIRSimulationEngine(cfg)
    engine.run()
    metrics = engine.collect_metrics()
    oes = [r.order_effect for r in engine.history]

    # Non-stationarity: compare first-half mean to second-half mean
    half = len(oes) // 2
    first_half = np.mean(oes[:half])
    second_half = np.mean(oes[half:])
    nonstationary = abs(first_half - second_half) > 1e-6

    return {
        "delta_trajectory": oes,
        "mean": metrics["order_effect_mean"],
        "variance": metrics["order_effect_var"],
        "first_half_mean": float(first_half),
        "second_half_mean": float(second_half),
        "is_nonstationary": nonstationary,
    }


# ------------------------------------------------------------------
# Experiment 2: Path Dependence
# ------------------------------------------------------------------


def experiment_path_dependence(
    cfg: SimulationConfig | None = None,
) -> dict[str, Any]:
    """Compare sequence S1 = AAAABBBB vs S2 = ABABABAB.

    Returns:
        Dict with state distance, operator distance, ``path_dependent``.
    """
    cfg = cfg or SimulationConfig()

    # Build two identical engines (same seed)
    e1 = CIIRSimulationEngine(cfg)
    e2 = CIIRSimulationEngine(cfg)

    # Sequence S1: 4×A then 4×B
    for label in ["A"] * 4 + ["B"] * 4:
        result = e1.op_sys.operators[label] @ e1.psi
        if np.linalg.norm(result) > 1e-15:
            e1.psi = normalize(result)
        rho = density_matrix(e1.psi)
        e1.op_sys.qratum_update(rho)

    # Sequence S2: alternating A B
    for label in ["A", "B"] * 4:
        result = e2.op_sys.operators[label] @ e2.psi
        if np.linalg.norm(result) > 1e-15:
            e2.psi = normalize(result)
        rho = density_matrix(e2.psi)
        e2.op_sys.qratum_update(rho)

    # Compare final states
    state_dist = float(np.linalg.norm(e1.psi - e2.psi))

    # Compare final operators
    op_dists: dict[str, float] = {}
    for lbl in e1.op_sys.operators:
        op_dists[lbl] = frobenius_norm(
            e1.op_sys.operators[lbl] - e2.op_sys.operators[lbl]
        )
    mean_op_dist = float(np.mean(list(op_dists.values())))

    return {
        "state_distance": state_dist,
        "operator_distances": op_dists,
        "mean_operator_distance": mean_op_dist,
        "path_dependent": state_dist > 1e-8 or mean_op_dist > 1e-8,
    }


# ------------------------------------------------------------------
# Experiment 3: Operator Drift
# ------------------------------------------------------------------


def experiment_operator_drift(
    cfg: SimulationConfig | None = None,
) -> dict[str, Any]:
    """Measure ‖ΔM_i‖_F over the simulation lifetime.

    Returns:
        Dict with drift trajectories and convergence classification.
    """
    engine = CIIRSimulationEngine(cfg)
    engine.run()

    drifts_by_label: dict[str, list[float]] = {}
    for rec in engine.history:
        for lbl, d in rec.operator_drifts.items():
            drifts_by_label.setdefault(lbl, []).append(d)

    classification: dict[str, str] = {}
    for lbl, dvals in drifts_by_label.items():
        arr = np.array(dvals)
        tail = arr[-len(arr) // 4 :]
        if np.mean(tail) < 1e-8:
            classification[lbl] = "converged"
        elif np.std(tail) > np.mean(tail):
            classification[lbl] = "oscillating"
        else:
            classification[lbl] = "drifting"

    return {
        "drift_trajectories": drifts_by_label,
        "classification": classification,
    }


# ------------------------------------------------------------------
# Experiment 4: Phase Transitions
# ------------------------------------------------------------------


def experiment_phase_transitions(
    base_cfg: SimulationConfig | None = None,
) -> dict[str, Any]:
    """Sweep constraint strength, η, and Hamiltonian scale.

    Returns:
        Dict with sweep data and detected transition flag.
    """
    base = base_cfg or SimulationConfig(num_steps=80)

    sweep_results: list[dict[str, Any]] = []

    for cs in (0.0, 0.1, 0.3, 0.5, 0.8, 1.0):
        for eta in (0.01, 0.05, 0.10, 0.20):
            for h_scale in (0.5, 1.0, 2.0):
                c = SimulationConfig(
                    dim=base.dim,
                    dt=base.dt,
                    num_steps=base.num_steps,
                    eta=eta,
                    constraint_strength=cs,
                    hamiltonian_scale=h_scale,
                    seed=base.seed,
                )
                eng = CIIRSimulationEngine(c)
                eng.run()
                m = eng.collect_metrics()
                sweep_results.append(
                    {
                        "constraint_strength": cs,
                        "eta": eta,
                        "hamiltonian_scale": h_scale,
                        "order_effect_mean": m["order_effect_mean"],
                        "entropy_final": m["entropy_trajectory"][-1],
                        "mean_drift_A": m["mean_operator_drifts"].get("A", 0),
                    }
                )

    # Detect transition: sign change in order effect across sweep
    oe_signs = [
        np.sign(r["order_effect_mean"]) for r in sweep_results
    ]
    sign_flips = sum(
        1
        for i in range(1, len(oe_signs))
        if oe_signs[i] != oe_signs[i - 1] and oe_signs[i] != 0
    )
    transition_detected = sign_flips > 0

    # Entropy spikes
    entropy_vals = [r["entropy_final"] for r in sweep_results]
    entropy_spike = bool(max(entropy_vals) - min(entropy_vals) > 0.1)

    return {
        "sweep": sweep_results,
        "sign_flips": sign_flips,
        "transition_detected": transition_detected,
        "entropy_spike": entropy_spike,
    }


# ------------------------------------------------------------------
# Experiment 5: Triadic Interference
# ------------------------------------------------------------------


def experiment_triadic_interference(
    cfg: SimulationConfig | None = None,
) -> dict[str, Any]:
    """Compare P(A→B) vs P(A→C→B) at each timestep.

    Returns:
        Dict with deviation trajectory and mean δ.
    """
    engine = CIIRSimulationEngine(cfg)
    engine.run()

    deviations: list[float] = []

    for rec in engine.history:
        psi_t = rec.psi
        m_a = engine.op_sys.operators["A"]
        m_b = engine.op_sys.operators["B"]
        m_c = engine.op_sys.operators["C"]

        # Direct: A → B
        try:
            psi_a = normalize(m_a @ psi_t)
            p_ab = measurement_probability(psi_a, m_b)
        except ValueError:
            deviations.append(0.0)
            continue

        # Triadic: A → C → B
        try:
            psi_ac = normalize(m_c @ psi_a)
            p_acb = measurement_probability(psi_ac, m_b)
        except ValueError:
            deviations.append(0.0)
            continue

        deviations.append(abs(p_ab - p_acb))

    return {
        "deviation_trajectory": deviations,
        "mean_deviation": float(np.mean(deviations)),
        "max_deviation": float(np.max(deviations)),
        "triadic_interference_present": float(np.mean(deviations)) > 1e-8,
    }


# ------------------------------------------------------------------
# Run all experiments
# ------------------------------------------------------------------


def run_all_experiments(
    cfg: SimulationConfig | None = None,
) -> dict[str, Any]:
    """Execute all five experiments and return combined results."""
    return {
        "order_effect": experiment_order_effect(cfg),
        "path_dependence": experiment_path_dependence(cfg),
        "operator_drift": experiment_operator_drift(cfg),
        "phase_transitions": experiment_phase_transitions(cfg),
        "triadic_interference": experiment_triadic_interference(cfg),
    }
