"""Structured report generator for CIIR simulation results.

Produces the structured text report specified in §IV of the
mega-prompt, including executive summary, key findings,
quantitative metrics, regime classification, falsifiability
verdict, and final determination.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np


def _fmt(x: float, decimals: int = 6) -> str:
    return f"{x:.{decimals}f}"


def generate_report(results: dict[str, Any]) -> str:
    """Generate a research-grade structured report.

    Args:
        results: Combined results dict from
            :func:`quasim.ciir.experiments.run_all_experiments`.

    Returns:
        Multi-section plain-text report.
    """
    buf = io.StringIO()
    w = buf.write

    # ── 1. Executive Summary ──────────────────────────────────────
    oe = results["order_effect"]
    pd = results["path_dependence"]
    od = results["operator_drift"]
    pt = results["phase_transitions"]
    ti = results["triadic_interference"]

    qratum_active = any(
        cl != "converged" for cl in od["classification"].values()
    )
    non_classical = (
        oe["is_nonstationary"]
        or ti["triadic_interference_present"]
        or pd["path_dependent"]
    )

    qratum_status = "ACTIVE" if qratum_active else "INACTIVE"
    qratum_desc = (
        "operators exhibit non-trivial drift across timesteps"
        if qratum_active
        else "operators have converged to fixed points"
    )

    w("=" * 72 + "\n")
    w("  CIIR → QuASIM → QRATUM  SIMULATION REPORT\n")
    w("=" * 72 + "\n\n")

    w("1. EXECUTIVE SUMMARY\n")
    w("-" * 40 + "\n")
    w(
        f"The CIIR → QuASIM → QRATUM pipeline was executed for the full "
        f"parameter space.  QRATUM operator recursion is "
        f"{qratum_status}: {qratum_desc}.  "
    )
    if non_classical:
        w(
            "The system deviates from both classical (static, commutative) "
            "and standard quantum cognition (fixed non-commutative operators) models.  "
        )
    else:
        w(
            "The system does NOT deviate from standard quantum cognition.  "
        )
    w(
        f"Order-effect non-stationarity: {oe['is_nonstationary']}.  "
        f"Path dependence: {pd['path_dependent']}.  "
        f"Triadic interference δ = {_fmt(ti['mean_deviation'])}.  "
        f"Phase transitions detected: {pt['transition_detected']}.\n\n"
    )

    # ── 2. Key Findings ───────────────────────────────────────────
    w("2. KEY FINDINGS\n")
    w("-" * 40 + "\n")

    w("A. Order Effects\n")
    w(f"   Mean Δ  = {_fmt(oe['mean'])}\n")
    w(f"   Var(Δ)  = {_fmt(oe['variance'])}\n")
    w(
        f"   First-half mean = {_fmt(oe['first_half_mean'])}, "
        f"second-half mean = {_fmt(oe['second_half_mean'])}\n"
    )
    w(f"   Non-stationary: {oe['is_nonstationary']}\n\n")

    w("B. Operator Evolution\n")
    for lbl, cls in od["classification"].items():
        w(f"   {lbl}: {cls}\n")
    w("\n")

    w("C. Path Dependence\n")
    w(f"   State distance (S1 vs S2):   {_fmt(pd['state_distance'])}\n")
    w(f"   Mean operator distance:      {_fmt(pd['mean_operator_distance'])}\n")
    w(f"   Path dependent: {pd['path_dependent']}\n\n")

    w("D. Phase Transitions\n")
    w(f"   Sign flips in Δ across sweep: {pt['sign_flips']}\n")
    w(f"   Entropy spike detected:       {pt['entropy_spike']}\n")
    w(f"   Transition detected:          {pt['transition_detected']}\n\n")

    w("E. Triadic Interference\n")
    w(f"   Mean δ = |P_AB − P_ACB| = {_fmt(ti['mean_deviation'])}\n")
    w(f"   Max  δ = {_fmt(ti['max_deviation'])}\n")
    w(f"   Present: {ti['triadic_interference_present']}\n\n")

    # ── 3. Quantitative Metrics ───────────────────────────────────
    w("3. QUANTITATIVE METRICS\n")
    w("-" * 40 + "\n")
    w(f"   Order effect mean:     {_fmt(oe['mean'])}\n")
    w(f"   Order effect variance: {_fmt(oe['variance'])}\n")

    # Mean operator drift (from drift experiment)
    for lbl, dvals in od["drift_trajectories"].items():
        arr = np.array(dvals)
        w(
            f"   Mean drift({lbl}): {_fmt(float(np.mean(arr)))}, "
            f"final drift({lbl}): {_fmt(dvals[-1])}\n"
        )
    w("\n")

    # ── 4. Regime Classification ──────────────────────────────────
    w("4. REGIME CLASSIFICATION\n")
    w("-" * 40 + "\n")

    all_converged = all(
        c == "converged" for c in od["classification"].values()
    )
    commutative = not oe["is_nonstationary"] and oe["variance"] < 1e-10

    if all_converged and commutative:
        regime = "Classical (static, commutative)"
    elif all_converged and not commutative:
        regime = "Quantum-like (non-commutative, fixed operators)"
    else:
        regime = "QRATUM (adaptive operators)"

    w(f"   Regime: {regime}\n\n")

    # ── 5. Falsifiability Verdict ─────────────────────────────────
    w("5. FALSIFIABILITY VERDICT\n")
    w("-" * 40 + "\n")
    verdicts = {
        "Non-stationary order effects": oe["is_nonstationary"],
        "Operator drift": qratum_active,
        "Path dependence": pd["path_dependent"],
        "Triadic interference": ti["triadic_interference_present"],
    }
    for desc, val in verdicts.items():
        mark = "✔" if val else "✗"
        w(f"   {mark} {desc}\n")
    w("\n")

    # ── 6. Final Determination ────────────────────────────────────
    w("6. FINAL DETERMINATION\n")
    w("-" * 40 + "\n")
    all_pass = all(verdicts.values())
    if all_pass:
        w(
            "   YES — This system produces behavior not reducible to "
            "classical or standard quantum cognition.  All four "
            "falsifiability criteria are satisfied: operator drift is "
            "non-zero (∂M/∂t ≠ 0), the order effect Δ_t is "
            "time-dependent, path dependence is present, and triadic "
            "interference δ > 0.\n"
        )
    else:
        failing = [d for d, v in verdicts.items() if not v]
        w(
            f"   NO — The following criteria are NOT met: "
            f"{', '.join(failing)}.  The system cannot be fully "
            f"classified as QRATUM-class.\n"
        )
    w("\n")

    # ── 7. Success Criteria Check ─────────────────────────────────
    w("7. QRATUM-CLASS SUCCESS CRITERIA\n")
    w("-" * 40 + "\n")
    w(f"   ∂M/∂t ≠ 0 :  {qratum_active}\n")
    w(f"   Δ_t time-dependent :  {oe['is_nonstationary']}\n")
    w(f"   δ_triadic > 0 :  {ti['triadic_interference_present']}\n")
    qratum_class = (
        qratum_active
        and oe["is_nonstationary"]
        and ti["triadic_interference_present"]
    )
    w(f"   QRATUM-class: {qratum_class}\n")
    w("=" * 72 + "\n")

    return buf.getvalue()
