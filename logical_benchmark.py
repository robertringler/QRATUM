"""
Logical Qubit Benchmark Driver

Runs end-to-end metrics for the spacetime/logical qubit encoding using existing
primitives in logical_encoding.py and simulator_noise_assisted.py.

Outputs:
- <prefix>_logical_benchmark.json: structured metrics
- <prefix>_summary.md: human-readable summary
- auto_logical_benchmark.tex: LaTeX appendix snippet (optional include)

Printed metric tags (for downstream parsing):
- F_PROC_BEST=...
- F_PROC_MEAN=...
- GAMMA_CENTER=...
- GAMMA_OPT_EST=...
- LOGICAL_GAMMA_L=...  (proxy via Ramsey visibility)
- LOGICAL_GAMMA_L_HZ=... (if --freq-hz provided)
- GATE_TIME_NS=... (if --freq-hz provided)
- MAX_SAFE_CLOCK_MHZ=... (if --freq-hz provided)
- D2_AT_GAMMA_CENTER=... (if computed)
- OTOC_AT_GAMMA_CENTER=... (if computed)

Assumptions:
- Single-excitation subspace as in the codebase.
- Phase gate modeled by phase_gate_event on region B.
"""

from __future__ import annotations

import argparse
import json
import math
import os

# Project-local imports
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure project root is on sys.path when running from automation/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from logical_encoding import (
    build_channel_evolver,
    choose_regions_block,
    logical_channel_pauli_action,
    phase_gate_event,
    process_fidelity,
    reconstruct_process_matrix,
    run_ramsey,
)

try:
    # Optional: D2/OTOC sweep
    from simulator_noise_assisted import run_analysis as run_d2_otoc
except Exception:
    run_d2_otoc = None


@dataclass
class BenchmarkConfig:
    Lx: int = 4
    Ly: int = 4
    block_size: int = 2
    disorder_list: List[float] = None
    gamma_center: Optional[float] = None
    gamma_C: float = 1.0
    gamma_alpha: float = 0.8
    rel_variations: List[float] = None
    T: float = 5.0
    dt: float = 0.02
    phase_duration: float = 0.2
    desired_phi: float = math.pi / 2
    ramsey_points: int = 17
    compute_d2_otoc: bool = True
    force_d2_otoc: bool = False
    d2_otoc_gamma_points: int = 50
    out_prefix: str = "auto_logical"
    fast: bool = False
    # Physical mapping: if provided, 1 model time unit corresponds to 1/(2π f) seconds
    phys_freq_hz: Optional[float] = None
    clock_safety_factor: float = 5.0

    def finalize(self):
        if self.disorder_list is None:
            self.disorder_list = [0.2, 0.25, 0.3]
        if self.rel_variations is None:
            self.rel_variations = [-0.2, -0.1, 0.0, 0.1, 0.2]
        # Fast mode trims workloads
        if self.fast:
            self.disorder_list = [self.disorder_list[0]]
            self.rel_variations = [0.0]
            self.ramsey_points = min(self.ramsey_points, 9)
            self.compute_d2_otoc = False


def unitary_z(phi: float) -> np.ndarray:
    return np.diag([1.0 + 0j, np.exp(1j * phi)])


def estimate_gamma_center(cfg: BenchmarkConfig) -> float:
    # Heuristic: gamma_opt ≈ C * sigma^alpha, use first disorder as representative
    sigma = float(cfg.disorder_list[0]) if cfg.disorder_list else 0.2
    return float(cfg.gamma_C * (sigma**cfg.gamma_alpha))


def benchmark(cfg: BenchmarkConfig) -> Dict[str, Any]:
    cfg.finalize()

    # Regions A,B selection
    A, B = choose_regions_block(cfg.Lx, cfg.Ly, block_size=cfg.block_size)

    # Phase gate event to enact Z_phi on logical |1̃> (B region)
    delta = cfg.desired_phi / cfg.phase_duration
    events = [phase_gate_event(B, t_center=cfg.T / 2, duration=cfg.phase_duration, delta=delta)]

    # Gamma center
    gamma_center = cfg.gamma_center if cfg.gamma_center is not None else estimate_gamma_center(cfg)

    # Target unitary and single-point F_proc at gamma_center, mean across disorders
    U_target = unitary_z(cfg.desired_phi)
    fproc_vals = []
    for sigma in cfg.disorder_list:
        evolver = build_channel_evolver(
            cfg.Lx, cfg.Ly, A, B, gamma_center, cfg.T, cfg.dt, sigma, events
        )
        outputs = logical_channel_pauli_action(evolver)
        chi = reconstruct_process_matrix(outputs)
        fproc_vals.append(process_fidelity(chi, U_target))
    fproc_vals = np.array(fproc_vals)
    fproc_mean = float(np.mean(fproc_vals))
    fproc_best = float(np.max(fproc_vals))

    # Robustness vs rel gamma variations and disorders
    robust_records = []
    for rel in cfg.rel_variations:
        gamma = gamma_center * (1.0 + rel)
        for sigma in cfg.disorder_list:
            evolver = build_channel_evolver(
                cfg.Lx, cfg.Ly, A, B, gamma, cfg.T, cfg.dt, sigma, events
            )
            outputs = logical_channel_pauli_action(evolver)
            chi = reconstruct_process_matrix(outputs)
            F = process_fidelity(chi, U_target)
            robust_records.append((gamma, sigma, float(F)))
    robust_arr = np.array(robust_records)

    # Ramsey coherence proxy at gamma_center: visibility at time T
    phase_values = np.linspace(0.0, 2 * math.pi, cfg.ramsey_points)
    contrasts, visibility = run_ramsey(
        cfg.Lx,
        cfg.Ly,
        A,
        B,
        gamma_center,
        cfg.T,
        cfg.dt,
        cfg.disorder_list[0],
        phase_values=phase_values,
        t_phase=cfg.T / 2,
        phase_duration=cfg.phase_duration,
        delta_energy=delta,
    )
    # Proxy logical dephasing rate Γ_L ~ -ln(visibility)/T (bounded below by 0)
    vis_clamped = max(1e-8, float(visibility))
    gamma_L = max(0.0, float(-math.log(vis_clamped) / cfg.T))

    # Optional D2/OTOC context around gamma_center
    d2_at_center = None
    otoc_at_center = None
    gamma_opt_est = None
    gamma_list = None
    # Only attempt D2/OTOC for small systems by default (N<=16), or if forced.
    N = cfg.Lx * cfg.Ly
    do_d2_otoc = bool(
        cfg.compute_d2_otoc and (cfg.force_d2_otoc or N <= 16) and (run_d2_otoc is not None)
    )
    if do_d2_otoc:
        # sample logspace around center within 2 decades
        gmin = max(1e-6, gamma_center / 100.0)
        gmax = gamma_center * 100.0
        gamma_list = np.logspace(math.log10(gmin), math.log10(gmax), cfg.d2_otoc_gamma_points)
        res = run_d2_otoc(
            Lx=cfg.Lx,
            Ly=cfg.Ly,
            disorder=cfg.disorder_list[0],
            T=cfg.T,
            gamma_list=gamma_list,
            compute_otoc_flag=True,
        )
        # pick nearest gamma to center
        idx = int(np.argmin(np.abs(np.array(res["gamma"]) - gamma_center)))
        d2_at_center = float(res["D2"][idx]) if res.get("D2") is not None else None
        if res.get("otocs") is not None and res["otocs"] is not None:
            otoc_at_center = float(res["otocs"][idx])
        gamma_opt_est = float(res.get("gamma_opt", gamma_center))

    # Physical mapping (optional): convert model durations/rates to SI using f (Hz)
    physical: Dict[str, Any] = {"enabled": False}
    if cfg.phys_freq_hz is not None and cfg.phys_freq_hz > 0:
        time_unit_sec = 1.0 / (
            2.0 * math.pi * float(cfg.phys_freq_hz)
        )  # 1 model time unit in seconds
        gate_time_s = cfg.phase_duration * time_unit_sec
        gate_time_ns = gate_time_s * 1e9
        gamma_L_phys_hz = gamma_L / time_unit_sec  # s^-1
        coherence_time_s = (1.0 / gamma_L_phys_hz) if gamma_L > 0 else None
        # Clock limited by both gate length and dephasing (with safety factor)
        gate_rate_hz = (1.0 / gate_time_s) if gate_time_s > 0 else None
        dephase_limited_hz = (
            (gamma_L_phys_hz / max(1e-12, cfg.clock_safety_factor)) if gamma_L > 0 else None
        )
        if gate_rate_hz is not None and dephase_limited_hz is not None:
            max_safe_clock_hz = min(gate_rate_hz, dephase_limited_hz)
        else:
            max_safe_clock_hz = gate_rate_hz or dephase_limited_hz
        physical = {
            "enabled": True,
            "freq_hz": float(cfg.phys_freq_hz),
            "time_unit_seconds": time_unit_sec,
            "gate_time_seconds": gate_time_s,
            "gate_time_ns": gate_time_ns,
            "gamma_L_phys_hz": gamma_L_phys_hz,
            "coherence_time_seconds": (coherence_time_s if coherence_time_s is not None else None),
            "max_safe_clock_hz": (max_safe_clock_hz if max_safe_clock_hz is not None else None),
            "max_safe_clock_mhz": (
                max_safe_clock_hz / 1e6 if max_safe_clock_hz is not None else None
            ),
            "clock_safety_factor": float(cfg.clock_safety_factor),
        }

    # Aggregate results
    results: Dict[str, Any] = {
        "config": asdict(cfg),
        "A": A,
        "B": B,
        "gamma_center": gamma_center,
        "fproc_per_sigma": fproc_vals.tolist(),
        "fproc_mean": fproc_mean,
        "fproc_best": fproc_best,
        "robustness": {
            "gamma": robust_arr[:, 0].tolist(),
            "disorder": robust_arr[:, 1].tolist(),
            "F_proc": robust_arr[:, 2].tolist(),
        },
        "ramsey": {
            "phase_values": phase_values.tolist(),
            "contrasts": np.array(contrasts).tolist(),
            "visibility": float(visibility),
            "gamma_L_proxy": gamma_L,
        },
        "d2_otoc": {
            "enabled": bool(do_d2_otoc),
            "gamma_list": (gamma_list.tolist() if gamma_list is not None else None),
            "D2_at_gamma_center": d2_at_center,
            "OTOC_at_gamma_center": otoc_at_center,
            "gamma_opt_est": gamma_opt_est,
        },
        "physical": physical,
    }

    return results


def write_artifacts(results: Dict[str, Any], out_prefix: str) -> None:
    # JSON
    with open(f"{out_prefix}_logical_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Summary Markdown
    lines = []
    lines.append(f"# Logical Qubit Benchmark Summary ({out_prefix})\n")
    lines.append(
        f"- Lattice: {results['config']['Lx']}x{results['config']['Ly']}  A={len(results['A'])} sites, B={len(results['B'])} sites"
    )
    lines.append(f"- gamma_center: {results['gamma_center']:.5g}")
    lines.append(
        f"- F_proc (mean over sigma): {results['fproc_mean']:.4f}  | best: {results['fproc_best']:.4f}"
    )
    lines.append(
        f"- Ramsey visibility (T={results['config']['T']}): {results['ramsey']['visibility']:.4f}  -> gamma_L proxy: {results['ramsey']['gamma_L_proxy']:.4g}"
    )
    if results.get("physical", {}).get("enabled"):
        phys = results["physical"]
        gt_ns = phys.get("gate_time_ns")
        max_clk_mhz = phys.get("max_safe_clock_mhz")
        coh_us = (
            (phys.get("coherence_time_seconds") * 1e6)
            if phys.get("coherence_time_seconds")
            else None
        )
        lines.append(
            f"- Physical mapping: f={phys['freq_hz']:.4g} Hz; gate_time={gt_ns:.3g} ns; gamma_L={phys['gamma_L_phys_hz']:.3g} Hz; coherence≈{(coh_us if coh_us is not None else 'n/a')} μs; max_safe_clock≈{(max_clk_mhz if max_clk_mhz is not None else 'n/a'):.3g} MHz (κ={phys['clock_safety_factor']:.3g})."
        )
    if results["d2_otoc"]["enabled"]:
        d2 = results["d2_otoc"]["D2_at_gamma_center"]
        oc = results["d2_otoc"]["OTOC_at_gamma_center"]
        go = results["d2_otoc"]["gamma_opt_est"]
        lines.append(
            f"- D2(gamma_center): {d2 if d2 is not None else 'n/a'}  | OTOC(gamma_center): {oc if oc is not None else 'n/a'}  | gamma_opt_est: {go if go is not None else 'n/a'}"
        )
    with open(f"{out_prefix}_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # LaTeX appendix snippet (idempotent)
    tex_lines = [
        "% Auto-generated logical benchmark appendix (do not edit by hand)",
        "\\section*{Logical Qubit Benchmark}",
        f"Lattice {results['config']['Lx']}x{results['config']['Ly']}, regions $|A|={len(results['A'])}$, $|B|={len(results['B'])}$.\\",
        f"Center $\\gamma_c={results['gamma_center']:.4g}$; target gate $Z_{{\\pi/2}}$.\\",
        "\\begin{itemize}",
        f"  \\item Process fidelity (mean over $\\sigma$): {results['fproc_mean']:.4f} (best {results['fproc_best']:.4f}).",
        f"  \\item Ramsey visibility at $T={results['config']['T']}$: {results['ramsey']['visibility']:.4f} $\\Rightarrow$ proxy $\\Gamma_L={results['ramsey']['gamma_L_proxy']:.4g}$.",
        (
            f"  \\item $D_2(\\gamma_c)={results['d2_otoc']['D2_at_gamma_center']:.4f}$, OTOC$ (\\gamma_c)={results['d2_otoc']['OTOC_at_gamma_center']:.4f}$, $\\gamma_\\mathrm{{opt}}\\approx{results['d2_otoc']['gamma_opt_est']:.4g}$."
            if results["d2_otoc"]["enabled"]
            and results["d2_otoc"]["D2_at_gamma_center"] is not None
            else None
        ),
        (
            lambda: (
                (
                    f"  \\item Physical mapping $f={results['physical']['freq_hz']:.4g}$ Hz: gate time $\u200b{results['physical']['gate_time_ns']:.3g}$ ns; $\\Gamma_L\\approx{results['physical']['gamma_L_phys_hz']:.3g}$ Hz; max safe clock $\\approx{results['physical']['max_safe_clock_mhz']:.3g}$ MHz ($\\kappa={results['physical']['clock_safety_factor']:.3g}$)."
                )
                if results.get("physical", {}).get("enabled")
                else None
            )
        )(),
        "\\end{itemize}",
    ]
    tex_lines = [ln for ln in tex_lines if ln is not None]
    with open("auto_logical_benchmark.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")


def print_tags(results: Dict[str, Any]) -> None:
    # Console metric tags (single-line, uppercase KEY=VALUE)
    print(f"F_PROC_BEST={results['fproc_best']:.6f}")
    print(f"F_PROC_MEAN={results['fproc_mean']:.6f}")
    print(f"GAMMA_CENTER={results['gamma_center']:.6g}")
    if results["d2_otoc"]["enabled"]:
        go = results["d2_otoc"]["gamma_opt_est"]
        if go is not None:
            print(f"GAMMA_OPT_EST={go:.6g}")
        d2 = results["d2_otoc"]["D2_at_gamma_center"]
        if d2 is not None:
            print(f"D2_AT_GAMMA_CENTER={d2:.6g}")
        oc = (
            results["d2_otoc"]["OTOC_at_GAMMA_CENTER"]
            if "OTOC_at_GAMMA_CENTER" in results["d2_otoc"]
            else results["d2_otoc"].get("OTOC_at_gamma_center")
        )
        if oc is not None:
            print(f"OTOC_AT_GAMMA_CENTER={oc:.6g}")
    print(f"LOGICAL_GAMMA_L={results['ramsey']['gamma_L_proxy']:.6g}")
    # Physical tags (if enabled)
    phys = results.get("physical") or {}
    if phys.get("enabled"):
        if phys.get("gamma_L_phys_hz") is not None:
            print(f"LOGICAL_GAMMA_L_HZ={phys['gamma_L_phys_hz']:.6g}")
        if phys.get("gate_time_ns") is not None:
            print(f"GATE_TIME_NS={phys['gate_time_ns']:.6g}")
        if phys.get("max_safe_clock_mhz") is not None:
            print(f"MAX_SAFE_CLOCK_MHZ={phys['max_safe_clock_mhz']:.6g}")


def parse_args() -> BenchmarkConfig:
    p = argparse.ArgumentParser(description="Logical qubit benchmark driver")
    p.add_argument("--Lx", type=int, default=4)
    p.add_argument("--Ly", type=int, default=4)
    p.add_argument("--block-size", type=int, default=2)
    p.add_argument("--disorder", type=float, nargs="*", default=[0.2, 0.25, 0.3])
    p.add_argument("--gamma-center", type=float, default=None)
    p.add_argument("--gamma-C", type=float, default=1.0)
    p.add_argument("--gamma-alpha", type=float, default=0.8)
    p.add_argument("--rel", type=float, nargs="*", default=[-0.2, -0.1, 0.0, 0.1, 0.2])
    p.add_argument("--T", type=float, default=5.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--phase-duration", type=float, default=0.2)
    p.add_argument("--phi", type=float, default=math.pi / 2)
    p.add_argument("--no-d2-otoc", action="store_true")
    p.add_argument("--d2-otoc-gamma-points", type=int, default=50)
    p.add_argument("--force-d2-otoc", action="store_true")
    p.add_argument("--ramsey-points", type=int, default=17)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--out-prefix", type=str, default="auto_logical")
    # Physical mapping options
    p.add_argument(
        "--freq-hz",
        type=float,
        default=None,
        help="Characteristic frequency f (Hz); 1 model time unit = 1/(2π f) seconds",
    )
    p.add_argument(
        "--clock-safety",
        type=float,
        default=5.0,
        help="Safety factor κ for dephasing-limited clock",
    )
    a = p.parse_args()
    cfg = BenchmarkConfig(
        Lx=a.Lx,
        Ly=a.Ly,
        block_size=a.block_size,
        disorder_list=a.disorder,
        gamma_center=a.gamma_center,
        gamma_C=a.gamma_C,
        gamma_alpha=a.gamma_alpha,
        rel_variations=a.rel,
        T=a.T,
        dt=a.dt,
        phase_duration=a.phase_duration,
        desired_phi=a.phi,
        ramsey_points=a.ramsey_points,
        compute_d2_otoc=(not a.no_d2_otoc),
        d2_otoc_gamma_points=a.d2_otoc_gamma_points,
        force_d2_otoc=a.force_d2_otoc,
        out_prefix=a.out_prefix,
        fast=a.fast,
        phys_freq_hz=a.freq_hz,
        clock_safety_factor=a.clock_safety,
    )
    return cfg


def main():
    cfg = parse_args()
    results = benchmark(cfg)
    write_artifacts(results, cfg.out_prefix)
    print_tags(results)


if __name__ == "__main__":
    main()
