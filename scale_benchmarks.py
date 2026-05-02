"""
Scale Benchmarks Runner

Sweeps lattice sizes and runs the logical benchmark to produce a consolidated
report:

Artifacts:
- scale_summary.json
- scale_summary.md
- auto_scale_table.tex  (LaTeX table for inclusion)

Depends on automation.logical_benchmark for the per-size metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from automation.logical_benchmark import BenchmarkConfig, benchmark  # type: ignore


def parse_sizes(arg: List[str]) -> List[Tuple[int, int]]:
    sizes: List[Tuple[int, int]] = []
    for token in arg:
        if "x" in token.lower():
            Lx, Ly = token.lower().split("x")
            sizes.append((int(Lx), int(Ly)))
        else:
            L = int(token)
            sizes.append((L, L))
    return sizes


def write_table(rows: List[Dict[str, Any]], path: str) -> None:
    header = [
        "Lx",
        "Ly",
        "$\\gamma_c$",
        "$F_\\mathrm{proc}$ (mean)",
        "$\\Gamma_L$",
        "gate (ns)",
        "max clk (MHz)",
        "$D_2(\\gamma_c)$",
        "OTOC$(\\gamma_c)$",
        "$\\gamma_\\mathrm{opt}$",
    ]

    def fmt(x: Any) -> str:
        return "-" if x is None else (f"{x:.4g}" if isinstance(x, (int, float)) else str(x))

    lines: List[str] = []
    lines.append("% Auto-generated scaling table")
    lines.append("\\begin{table}[h]")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append("  \\begin{tabular}{r r r r r r r r r r}")
    lines.append("    \\toprule")
    lines.append("    " + " & ".join(header) + " \\\\")
    lines.append("    \\midrule")
    for r in rows:
        fields = [
            str(r["Lx"]),
            str(r["Ly"]),
            fmt(r["gamma_center"]),
            fmt(r["fproc_mean"]),
            fmt(r["gamma_L"]),
            fmt(r.get("gate_time_ns")),
            fmt(r.get("max_safe_clock_mhz")),
            fmt(r["D2_center"]),
            fmt(r["OTOC_center"]),
            fmt(r["gamma_opt_est"]),
        ]
        lines.append("    " + " & ".join(fields) + " \\\\")
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("  \\caption{Logical qubit scaling metrics across lattice sizes.}")
    lines.append("  \\label{tab:logical_scaling}")
    lines.append("\\end{table}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Run scaling benchmarks over lattice sizes")
    ap.add_argument(
        "--sizes",
        nargs="*",
        default=["6", "8", "12"],
        help="Sizes as L or LxLy tokens, e.g. 8 12 or 8x8 12x12",
    )
    ap.add_argument("--disorder", nargs="*", type=float, default=[0.2, 0.25, 0.3])
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--phi", type=float, default=math.pi / 2)
    ap.add_argument("--block-size", type=int, default=2)
    ap.add_argument("--out-prefix", type=str, default="scale")
    ap.add_argument("--no-d2-otoc", action="store_true")
    ap.add_argument("--fast", action="store_true")
    # Physical mapping passthrough
    ap.add_argument(
        "--freq-hz",
        type=float,
        default=None,
        help="Characteristic frequency f (Hz); 1 model time unit = 1/(2π f) seconds",
    )
    ap.add_argument(
        "--clock-safety",
        type=float,
        default=5.0,
        help="Safety factor κ for dephasing-limited clock",
    )
    args = ap.parse_args()

    sizes = parse_sizes(args.sizes)
    rows: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []

    for Lx, Ly in sizes:
        cfg = BenchmarkConfig(
            Lx=Lx,
            Ly=Ly,
            block_size=args.block_size,
            disorder_list=args.disorder,
            T=args.T,
            dt=args.dt,
            desired_phi=args.phi,
            compute_d2_otoc=(not args.no_d2_otoc),
            out_prefix=f"{args.out_prefix}_L{Lx}x{Ly}",
            fast=args.fast,
            phys_freq_hz=args.freq_hz,
            clock_safety_factor=args.clock_safety,
        )
        res = benchmark(cfg)
        all_results.append(res)
        rows.append(
            {
                "Lx": Lx,
                "Ly": Ly,
                "gamma_center": res["gamma_center"],
                "fproc_mean": res["fproc_mean"],
                "gamma_L": res["ramsey"]["gamma_L_proxy"],
                "gate_time_ns": (
                    res.get("physical", {}).get("gate_time_ns")
                    if res.get("physical", {}).get("enabled")
                    else None
                ),
                "max_safe_clock_mhz": (
                    res.get("physical", {}).get("max_safe_clock_mhz")
                    if res.get("physical", {}).get("enabled")
                    else None
                ),
                "D2_center": (
                    res["d2_otoc"]["D2_at_gamma_center"] if res["d2_otoc"]["enabled"] else None
                ),
                "OTOC_center": (
                    res["d2_otoc"]["OTOC_at_gamma_center"] if res["d2_otoc"]["enabled"] else None
                ),
                "gamma_opt_est": (
                    res["d2_otoc"]["gamma_opt_est"] if res["d2_otoc"]["enabled"] else None
                ),
            }
        )

    # Write JSON and Markdown summary
    with open("scale_summary.json", "w", encoding="utf-8") as f:
        json.dump({"sizes": sizes, "rows": rows, "results": all_results}, f, indent=2)

    md = [
        "# Logical Qubit Scaling Summary",
        "",
        "| Lx | Ly | gamma_c | F_proc (mean) | Gamma_L | gate (ns) | max clk (MHz) | D2(gamma_c) | OTOC(gamma_c) | gamma_opt |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt_md(x: Any) -> str:
        return "-" if x is None else (f"{x:.4g}" if isinstance(x, (int, float)) else str(x))

    for r in rows:
        md.append(
            f"| {r['Lx']} | {r['Ly']} | {fmt_md(r['gamma_center'])} | {fmt_md(r['fproc_mean'])} | {fmt_md(r['gamma_L'])} | {fmt_md(r.get('gate_time_ns'))} | {fmt_md(r.get('max_safe_clock_mhz'))} | {fmt_md(r['D2_center'])} | {fmt_md(r['OTOC_center'])} | {fmt_md(r['gamma_opt_est'])} |"
        )
    with open("scale_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    # LaTeX table
    write_table(rows, "auto_scale_table.tex")

    # Emit minimal tags for logs
    best = max((r["fproc_mean"] for r in rows if r["fproc_mean"] is not None), default=None)
    if best is not None:
        print(f"F_PROC_MEAN_BEST={best:.6f}")


if __name__ == "__main__":
    main()
