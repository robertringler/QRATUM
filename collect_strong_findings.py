from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "automation_runs"


def find_latest_index() -> Path | None:
    if not RUNS.exists():
        return None
    # choose most recent directory with index.json
    candidates = []
    for p in RUNS.iterdir():
        if p.is_dir():
            idx = p / "index.json"
            if idx.exists():
                candidates.append(idx)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def summarize_index(idx_path: Path) -> str:
    data = json.loads(idx_path.read_text(encoding="utf-8"))
    recs = data.get("records", [])
    lines = []
    lines.append(f"index: {idx_path}")
    lines.append(
        f"config_hash: {data.get('config_hash')} | quick_mode: {data.get('quick_mode')} | tasks: {data.get('task_count')} | any_failure: {data.get('any_failure')}"
    )
    lines.append(
        f"aggregate_wall_time: {data.get('aggregate_wall_time'):.3f}s | aggregate_cpu_time: {data.get('aggregate_cpu_time'):.3f}s | peak_rss: {data.get('peak_rss_bytes')} bytes"
    )
    # iterate records
    for i, r in enumerate(recs, 1):
        Lx, Ly = r.get("lattice", [None, None])
        dis = r.get("disorder")
        m = r.get("outputs", {}).get("metrics", {})
        q = r.get("quality", {})
        lines.append(f"-- record {i}: L={Lx}x{Ly}, disorder={dis}, gamma_opt={r.get('gamma_opt')}")
        lines.append(
            f"   D2={m.get('d2')} (forced={m.get('d2_forced')}) | alpha_d={m.get('alpha_d')} | d_s≈D2*alpha_d => gap={m.get('fractal_transport_gap')}"
        )
        lines.append(f"   Fisher: gamma={m.get('fisher_gamma')} sqrt={m.get('fisher_gamma_sqrt')}")
        lines.append(
            f"   Logical: F_logical={m.get('logical_fidelity')} | F_ent={m.get('entanglement_fidelity')} | I_coh={m.get('coherent_information')}"
        )
        lines.append(
            f"   Stability: trace_max={m.get('trace_max')} | dt_order={m.get('dt_order')} | boot_hw={m.get('bootstrap_halfwidth')} | min_eig={m.get('min_eig')}"
        )
        lines.append(
            f"   Perf: wall={m.get('wall_time_seconds')}s | cpu={m.get('cpu_time')}s | rss={m.get('rss')}"
        )
        lines.append(
            f"   Quality: trace_ok={q.get('trace_ok')} | positivity_ok={q.get('positivity_ok')} | dt_order_ok={q.get('dt_order_ok')} | bootstrap_ok={q.get('bootstrap_ok')} | fractal_transport_ok={q.get('fractal_transport_ok')}"
        )
    return "\n".join(lines)


def main():
    idx = None
    if len(sys.argv) > 1:
        idx = Path(sys.argv[1])
        if not idx.exists():
            print(f"Provided index.json not found: {idx}", file=sys.stderr)
            sys.exit(2)
    else:
        idx = find_latest_index()
        if idx is None:
            print("No index.json found under automation_runs", file=sys.stderr)
            sys.exit(1)
    print(summarize_index(idx))


if __name__ == "__main__":
    main()
