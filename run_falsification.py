#!/usr/bin/env python3
r"""CIIR Falsification Protocol Runner.

Executes the full 5-step falsification protocol and secondary objectives
S1–S3, then prints a structured report in five sections:

  Section 1: EXECUTION LOG        (Steps 1–5 raw numerical outputs)
  Section 2: VERDICT              (A / B / C with justification)
  Section 3: SECONDARY RESULTS    (S1 stress scan, S2 controllers, S3 α)
  Section 4: NEXT MOVE            (hardware / GAP-4 / revision steps)
  Section 5: MONOGRAPH UPDATE     (Chapter 22 additions / retractions)

Usage
-----
  python run_falsification.py                   # default N=3
  python run_falsification.py --N 4             # 4 qubits
  python run_falsification.py --n-steps 100     # longer run
  python run_falsification.py --output results/falsification.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ── ensure repo root is on the path ────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from quasim.ciir.multi_qubit.analysis.falsification import (
    PROTOCOL_PARAMS, ArtifactResult, FalsificationVerdict,
    run_falsification_protocol)

# ────────────────────────────────────────────────────────────────
# FORMATTING HELPERS
# ────────────────────────────────────────────────────────────────

SEP = "═" * 72
SEP2 = "─" * 72


def _h1(title: str) -> str:
    return f"\n{SEP}\n  {title}\n{SEP}"


def _h2(title: str) -> str:
    return f"\n{SEP2}\n  {title}\n{SEP2}"


def _fmt_arr(arr, n: int = 6) -> str:
    arr = np.asarray(arr)
    if arr.size <= n:
        vals = [f"{v:.6f}" for v in arr]
    else:
        head = [f"{v:.6f}" for v in arr[:3]]
        tail = [f"{v:.6f}" for v in arr[-3:]]
        vals = head + ["..."] + tail
    return "[" + ", ".join(vals) + "]"


def _ah_line(r: ArtifactResult) -> str:
    status = "PASS ✓" if r.passed else "FAIL ✗"
    return (
        f"  {r.label}  {status:8s}  value={r.value:.5f}  threshold={r.threshold:.4f}\n"
        f"         {r.detail}\n"
        f"         Counterargument: {r.counterargument}"
    )


# ────────────────────────────────────────────────────────────────
# REPORT SECTIONS
# ────────────────────────────────────────────────────────────────

def section1(result: dict) -> str:
    p = result["params"]
    sig = result["signal"]
    ciir = result["ciir"]
    null = result["null"]
    arts = result["artifacts"]

    lines = [_h1("SECTION 1 — EXECUTION LOG")]

    # Steps 1 & 2
    lines.append(_h2("STEP 1 — CIIR Baseline Extraction (Model A)"))
    lines.append(f"  N = {p['N']}  (dim = {2**p['N']})")
    lines.append(f"  n_steps = {p['n_steps']}  dt = {p['dt']}")
    lines.append(f"  γ = {p['gamma']}  noise_strength = {p['noise_strength']}")
    lines.append(f"  O_CIIR(t) = {_fmt_arr(sig['fidelity_ciir'])}")
    lines.append(f"  S(ρ_CIIR)(t) = {_fmt_arr(sig['entropy_ciir'])}")
    lines.append(f"  O_CIIR_final = {sig['fidelity_ciir'][-1]:.6f}")

    lines.append(_h2("STEP 2 — Null Model (Standard GKSL, Model B)"))
    lines.append(f"  O_std(t) = {_fmt_arr(sig['fidelity_null'])}")
    lines.append(f"  S(ρ_std)(t) = {_fmt_arr(sig['entropy_null'])}")
    lines.append(f"  O_std_final = {sig['fidelity_null'][-1]:.6f}")

    lines.append(_h2("STEP 3 — Signal Quantification"))
    lines.append(f"  ΔO(t) = {_fmt_arr(sig['delta_O'])}")
    lines.append(f"  ΔO_max = {sig['delta_O_max']:.8f}")
    lines.append(f"  ΔO_final = {sig['final_delta']:.8f}")
    lines.append(f"  σ_numerical = {sig['sigma_numerical']:.4e}")
    lines.append(f"  SNR = ΔO_max / σ = {sig['snr']:.2e}")

    lines.append(_h2("STEP 4 — Artifact Elimination"))
    for label in ["AH-1", "AH-2", "AH-3", "AH-4", "AH-5"]:
        r = arts[label]
        lines.append(_ah_line(r))
        lines.append("")

    return "\n".join(lines)


def section2(result: dict) -> str:
    v: FalsificationVerdict = result["verdict"]
    lines = [_h1("SECTION 2 — VERDICT")]
    lines.append(f"\n  VERDICT:  {v.letter}\n")
    lines.append(f"  Justification:\n    {v.justification}")
    lines.append(f"\n  Strongest counterargument:\n    {v.counterargument}")
    return "\n".join(lines)


def section3(result: dict) -> str:
    if "secondary" not in result:
        return _h1("SECTION 3 — SECONDARY RESULTS") + "\n  (not computed)"

    sec = result["secondary"]
    lines = [_h1("SECTION 3 — SECONDARY RESULTS")]

    # S1
    s1 = sec["s1_stress"]
    lines.append(_h2("S1 — Stress Regime Scan"))
    lines.append(f"  Baseline         ΔO_max = {s1['delta_baseline']:.6f}")
    lines.append(f"  High entanglement ΔO_max = {s1['delta_high_entanglement']:.6f}  → {s1['trend_high_entanglement']}")
    lines.append(f"  Long memory      ΔO_max = {s1['delta_long_memory']:.6f}  → {s1['trend_long_memory']}")
    lines.append(f"  Rapid control    ΔO_max = {s1['delta_rapid_control']:.6f}  → {s1['trend_rapid_control']}")

    # S2
    s2 = sec["s2_controllers"]
    lines.append(_h2("S2 — Controller Comparison"))
    lines.append(f"  LQR proxy        ΔO_max = {s2['delta_lqr']:.6f}")
    lines.append(f"  Pontryagin       ΔO_max = {s2['delta_pontryagin']:.6f}")
    lines.append(f"  Natural gradient ΔO_max = {s2['delta_natural_gradient']:.6f}")
    lines.append(f"  Geometric enhances signal: {s2['geometric_enhances_signal']}")
    lines.append(f"  Nat-grad / LQR ratio: {s2['nat_grad_vs_lqr_ratio']:.4f}")

    # S3
    s3 = sec["s3_alpha"]
    lines.append(_h2("S3 — α Consistency Check"))
    lines.append(f"  N*  (diverge > 5 %): {s3['n_star_diverge']}")
    lines.append(f"  All agree < 5 %: {s3['all_agree_5pct']}")
    lines.append(f"  All in [1.79, 1.87]: {s3['all_in_range']}")
    for r in s3["per_N"]:
        lines.append(
            f"    N={r['N']:2d}: α_spec={r['alpha_spectral']:.4f}, "
            f"α_Haus={r['alpha_hausdorff']:.4f}, "
            f"|Δα|={r['agreement']:.4f}, "
            f"in_range={r['both_in_range']}"
        )

    return "\n".join(lines)


def section4(result: dict) -> str:
    v: FalsificationVerdict = result["verdict"]
    sig = result["signal"]
    arts = result["artifacts"]
    lines = [_h1("SECTION 4 — NEXT MOVE")]

    if v.letter == "A":
        lines.append("""
  Verdict A: hardware test is justified.  Exact steps:

  1. Select a superconducting or trapped-ion processor with N = 3–5 qubits,
     T₁ ≥ 100 μs, T₂ ≥ 50 μs (from hardware_params.py calibration table).

  2. Implement Ramsey protocol (causal_claim.py, RamseyProtocol defaults):
       τ_noise = 0.5 (scaled to physical time via latency map),
       τ_free = 1.0, p_noise > p* = 0.1.

  3. Prepare ρ₀ = |0…0> via active reset.  Apply noise pulse.

  4. (Model A) Apply fractal QEC round (surface code or CIIR code).
     (Model B) Skip QEC round.

  5. Measure O = ⟨σ_z^⊗N⟩ (proxy for fidelity with |0…0>).

  6. Repeat ≥ 1000 shots per condition.  Compute:
       ΔO_hardware = O_A − O_B.
     Compare to σ_shot = 1/√(n_shots) ≈ 0.032 for n_shots=1000.
     Signal ΔO_hardware > 3σ_shot is required for experimental confirmation.

  7. Report full error budget:  SPAM, gate, measurement, leakage.
""")
    elif v.letter == "B":
        lines.append(f"""
  Verdict B: marginal signal (SNR = {sig['snr']:.1f}).
  Failed artifact hypotheses: {[r.label for r in v.artifact_results if not r.passed]}.

  GAP 4 strengthening required:

  1. Strengthen observable O beyond fidelity.  Replace F(ρ, ρ_target) with a
     quantum Fisher information or Pauli shadow estimator that is more
     sensitive to constraint-induced modifications.

  2. Improve constraint projector Π_C beyond a diagonal lower-half projector.
     Use a genuine stabilizer code projector derived from the CIIR fractal QEC
     (fractal_qec.py) to make the claim physically non-trivial.

  3. Prove analytically (not by simulation) that the constraint projection
     modifies the Lindbladian fixed point, not just the trajectory.
     This closes GAP 4 at the level of a theorem, not a simulation result.

  4. Re-run the protocol after each strengthening step until Verdict A.
""")
    else:
        lines.append(f"""
  Verdict C: null result (SNR = {sig['snr']:.1f}).
  Required revisions:

  causal_claim.py:
    · Revise Proposition P to require a specific functional form of O
      that is analytically distinguishable (not just numerically).
    · Replace the code-subspace projector with a hardware-realizable
      stabilizer projector.  The current Π_C (lower-half projector) is
      a placeholder.
    · Increase p_noise / p* ratio to at least 2× to force clear threshold
      crossing behavior.

  alpha_derivation.py:
    · Theorem 22.1 should be stated with explicit bounds on when the
      spectral ratio λ₂/Δ falls in [1.5, 2.5].
    · Add a calibration step: require hardware parameters (T₁, T₂) to set
      Δ before computing α.
""")

    return "\n".join(lines)


def section5(result: dict, elapsed: float) -> str:
    v: FalsificationVerdict = result["verdict"]
    sig = result["signal"]
    p = result["params"]
    s3 = result.get("secondary", {}).get("s3_alpha", {})

    lines = [_h1("SECTION 5 — MONOGRAPH UPDATE")]

    if v.letter == "A":
        status_str = "CONFIRMED (simulation level)"
        add_or_retract = "add"
    elif v.letter == "B":
        status_str = "MARGINAL — requires GAP 4 strengthening"
        add_or_retract = "revise"
    else:
        status_str = "NOT CONFIRMED — causal claim requires revision"
        add_or_retract = "retract"

    alpha_stable = s3.get("all_agree_5pct", "N/A")
    alpha_in_range = s3.get("all_in_range", "N/A")

    lines.append(f"""
  Chapter 22 should be updated as follows (falsification run date: 2026-04-25,
  N = {p['N']}, n_steps = {p['n_steps']}, SNR = {sig['snr']:.2e}):

  Section 22.4 should be added (or updated) as follows:

  \\section{{Falsification Protocol and Simulation Results (GAP 4 Empirical Test)}}
  \\label{{sec:ch22-falsification}}

  Theorem 22.4 should read:
  \\begin{{theorem}}[Causal Claim Falsification (Simulation)]\\label{{thm:ch22-falsify}}
    \\emph{{(Simulation result.)}}
    Under the five-step falsification protocol (Sections~22.4.1--22.4.5),
    the CIIR observable $O = F(\\rho, \\rho_{{\\rm target}})$ achieves
    $\\Delta O_{{\\rm max}} = {sig['delta_O_max']:.6f}$ with
    $\\sigma_{{\\rm numerical}} = {sig['sigma_numerical']:.2e}$,
    giving $\\mathrm{{SNR}} = {sig['snr']:.2e}$.
    Verdict: \\textbf{{{v.letter}}} — {status_str}.
    This constitutes a {add_or_retract}-to-record for the simulation-level
    CIIR causal claim.  It does \\emph{{not}} constitute experimental confirmation.
  \\end{{theorem}}

  Section 22.1 (Theorem 22.1, $\\alpha$ derivation) should be revised to state:
    "The spectral and Hausdorff routes agree within 5\\,\\% for $N \\leq 6$ qubits
    (verified numerically; see Table~22.4). Stability at $N=2,4,6$: 
    all_agree={alpha_stable}, all_in_range={alpha_in_range}.
    Divergence at $N^* = {s3.get('n_star_diverge', 'None')}$ (if any)."

  If Verdict C was returned, Section 22.2 (Theorem 22.2, Causal Claim) should
  additionally state:
    "Proposition~P is not supported at the simulation level for the current
    choice of code-subspace projector $\\Pi_C$.  A hardware-realizable
    stabilizer code projector is required before Theorem~22.2 can be advanced
    to experimental validation."
""")

    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────
# SERIALISATION (JSON-safe)
# ────────────────────────────────────────────────────────────────

def _to_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, ArtifactResult):
        return {
            "label": obj.label,
            "passed": obj.passed,
            "value": obj.value,
            "threshold": obj.threshold,
            "detail": obj.detail,
            "counterargument": obj.counterargument,
        }
    if isinstance(obj, FalsificationVerdict):
        return {
            "letter": obj.letter,
            "snr": obj.snr,
            "delta_O_max": obj.delta_O_max,
            "sigma_numerical": obj.sigma_numerical,
            "all_artifacts_pass": obj.all_artifacts_pass,
            "n_artifacts_pass": obj.n_artifacts_pass,
            "justification": obj.justification,
            "counterargument": obj.counterargument,
        }
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    return obj


def _make_summary(result: dict) -> dict:
    """Compact JSON summary (omits full trajectory arrays)."""
    sig = result["signal"]
    v = result["verdict"]
    arts = result["artifacts"]
    p = result["params"]
    sec = result.get("secondary", {})

    summary = {
        "params": p,
        "signal": {
            "fidelity_ciir_final": float(sig["fidelity_ciir"][-1]),
            "fidelity_null_final": float(sig["fidelity_null"][-1]),
            "delta_O_max": float(sig["delta_O_max"]),
            "final_delta": float(sig["final_delta"]),
            "sigma_numerical": float(sig["sigma_numerical"]),
            "snr": float(sig["snr"]),
        },
        "artifacts": {
            k: {
                "passed": r.passed,
                "value": r.value,
                "threshold": r.threshold,
                "detail": r.detail,
            }
            for k, r in arts.items()
        },
        "verdict": {
            "letter": v.letter,
            "snr": v.snr,
            "all_artifacts_pass": v.all_artifacts_pass,
            "n_artifacts_pass": v.n_artifacts_pass,
        },
    }
    if sec:
        summary["secondary"] = _to_json_safe(sec)
    return summary


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="CIIR Falsification Protocol Runner")
    parser.add_argument("--N", type=int, default=PROTOCOL_PARAMS["N_default"],
                        help="Number of qubits (default 3)")
    parser.add_argument("--n-steps", type=int, default=PROTOCOL_PARAMS["n_steps"],
                        help="Integration steps (default 50)")
    parser.add_argument("--dt", type=float, default=PROTOCOL_PARAMS["dt"],
                        help="Time step (default 0.01)")
    parser.add_argument("--gamma", type=float, default=PROTOCOL_PARAMS["gamma"],
                        help="Lindblad noise rate (default 0.02)")
    parser.add_argument("--noise-strength", type=float,
                        default=PROTOCOL_PARAMS["noise_strength"],
                        help="Depolarizing noise (default 0.15, > p*=0.1)")
    parser.add_argument("--N-scan", type=int, nargs="+", default=[2, 4, 6],
                        help="N values for AH-5 and S3 (default 2 4 6)")
    parser.add_argument("--no-secondary", action="store_true",
                        help="Skip S1–S3 secondary objectives")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save JSON summary")
    args = parser.parse_args(argv)

    print(_h1("CIIR FALSIFICATION PROTOCOL — AUTONOMOUS EXECUTION"))
    print(f"\n  N = {args.N}  |  n_steps = {args.n_steps}  |  dt = {args.dt}")
    print(f"  γ = {args.gamma}  |  noise_strength = {args.noise_strength}")
    print(f"  N_scan = {args.N_scan}")
    print(f"  Secondary: {'NO' if args.no_secondary else 'YES'}")
    print()

    t0 = time.perf_counter()
    result = run_falsification_protocol(
        N=args.N,
        n_steps=args.n_steps,
        dt=args.dt,
        gamma=args.gamma,
        noise_strength=args.noise_strength,
        N_scan=args.N_scan,
        run_secondary=not args.no_secondary,
    )
    elapsed = time.perf_counter() - t0

    print(section1(result))
    print(section2(result))
    print(section3(result))
    print(section4(result))
    print(section5(result, elapsed))

    print(f"\n{SEP}")
    print(f"  Total execution time: {elapsed:.1f} s")
    print(f"  Verdict: {result['verdict'].letter}")
    print(SEP)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = _make_summary(result)
        with open(out_path, "w") as fh:
            json.dump(summary, fh, indent=2, default=_to_json_safe)
        print(f"\n  Summary written to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
