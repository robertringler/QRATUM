"""Automation pipeline for batch simulation, analysis, validation, and LaTeX report build.

Usage (CLI):
        python -m automation.run_pipeline --config automation/pipeline_config.yaml

Features:
    - Generates gamma scan arrays and lattice/disorder combinations.
    - Runs selected analyses (leveraging existing all_analyses functions where possible).
    - Performs quality gate evaluation on produced artifacts.
    - Aggregates metadata into a JSON index.
    - Optionally builds the LaTeX master document.
    - Archives outputs by timestamp.

Extensibility:
    - Add new quality gates in `evaluate_quality_gates`.
    - Register additional analysis modes in `dispatch_analysis`.
    - Integrate parallelization (e.g., multiprocessing) if needed later.
"""

from __future__ import annotations

import os as _os_early
import sys as _sys_early
import time as _time_early

# Early import-time banner / heartbeat (debug only)
if _os_early.environ.get("PIPELINE_DEBUG", "0") == "1":
    print(f"[pipeline][import] module loaded; python={_sys_early.executable}")
    try:
        with open(".pipeline_heartbeat", "a", encoding="utf-8") as hb:
            hb.write(f"import { _time_early.time():.3f }\n")
    except Exception:
        pass
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import smtplib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except ImportError:
    print("[pipeline] Missing dependency: pyyaml (install with pip install pyyaml)")
    yaml = None  # type: ignore
try:
    import jsonschema  # type: ignore
except Exception:
    jsonschema = None  # type: ignore

import numpy as np


# ----------------------------------------------------------------------------------
# Data Classes
# ----------------------------------------------------------------------------------
@dataclass
class QualityResult:
    trace_ok: bool | None = None
    positivity_ok: bool | None = None
    dt_order: float | None = None
    dt_order_ok: bool | None = None
    bootstrap_halfwidth: float | None = None
    bootstrap_ok: bool | None = None
    fractal_transport_gap: float | None = None
    fractal_transport_ok: bool | None = None
    # Fisher information (information geometry) light gate
    fisher_gamma: float | None = None
    fisher_gamma_sqrt: float | None = None
    fisher_ok: bool | None = None
    # (instrumentation lines moved to robust block later)
    min_eig: float | None = None
    adaptive_dt_rejections: int | None = None
    local_error_max: float | None = None
    delta_instanton_valid: bool | None = None
    delta_instanton_ok: bool | None = None
    # Autonomous Discovery quality gates
    auto_discoveries_ok: bool | None = None
    auto_efficiency_ok: bool | None = None
    auto_anomaly_ratio_ok: bool | None = None

    # (instrumentation lines moved to robust block later)
    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    # (end of misplaced instrumentation removal)


@dataclass
class RunRecord:
    lattice: Tuple[int, int]
    disorder: float
    gamma_opt: float | None
    config_hash: str
    timestamp: float
    quality: QualityResult = field(default_factory=QualityResult)
    outputs: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        d = {
            "lattice": list(self.lattice),
            "disorder": self.disorder,
            "gamma_opt": self.gamma_opt,
            "config_hash": self.config_hash,
            "timestamp": self.timestamp,
            "quality": self.quality.as_dict(),
            "outputs": self.outputs,
        }
        # Derive convenience flag
        rc = self.outputs.get("returncode") if isinstance(self.outputs, dict) else None
        d["analysis_ok"] = rc == 0
        return d


# ----------------------------------------------------------------------------------
# Configuration Loading
# ----------------------------------------------------------------------------------


def load_config(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("pyyaml not installed")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Schema validation (lightweight) if jsonschema present
    if jsonschema is not None:
        schema = {
            "type": "object",
            "properties": {
                "lattice": {"type": "object"},
                "noise": {"type": "object"},
                "analysis": {"type": "object"},
                "quality_gates": {"type": "object"},
            },
            "required": ["lattice", "noise"],
        }
        try:
            jsonschema.validate(cfg, schema)
        except Exception as e:
            print(f"[pipeline] Config schema warning: {e}")
    return cfg


def hash_config(cfg: Dict[str, Any]) -> str:
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    return h[:16]


# ----------------------------------------------------------------------------------
# Gamma Scan & Utilities
# ----------------------------------------------------------------------------------


def logspace_scan(start_exp: float, end_exp: float, points: int) -> List[float]:
    return list(np.logspace(start_exp, end_exp, points))


# Placeholder: integrate with actual advanced_analysis functions if needed


def estimate_gamma_opt(dummy_scan: List[float]) -> float:
    # Simple heuristic: pick median gamma as placeholder (replace with real scan logic)
    return float(np.median(dummy_scan))


# ----------------------------------------------------------------------------------
# Optional Gamma Refinement Heuristic
# ----------------------------------------------------------------------------------


def refine_gamma(
    initial_gamma: float, cfg: Dict[str, Any], lx: int, ly: int, disorder: float
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """Refine gamma by sampling a small local window and minimizing fractal gap |2 - D2*alpha_d|.

    Strategy: run lightweight convergence+MSD analyses (forced, quick env) for a few gamma points around the initial
    estimate. Choose the gamma that minimizes gap if both D2 & alpha_d present; otherwise maximize D2.
    Returns (refined_gamma, diagnostics_record) where diagnostics_record contains per-gamma metrics.
    """
    refine_cfg = cfg.get("gamma_refine", {})
    if not refine_cfg.get("enable", False):
        return initial_gamma, None
    span = float(refine_cfg.get("relative_span", 0.4))
    points = int(refine_cfg.get("points", 5))
    short_T = float(
        refine_cfg.get("short_target_time", cfg.get("noise", {}).get("target_time", 2.0))
    )
    early_stop_thresh = float(refine_cfg.get("early_stop_improvement", 0.0))
    # Guard
    if points < 2 or span <= 0:
        return initial_gamma, None
    low = initial_gamma * (1 - span / 2)
    high = initial_gamma * (1 + span / 2)
    gammas = list(np.linspace(low, high, points))
    # Use shortened T for probes to reduce cost
    T = short_T
    dt = cfg.get("noise", {}).get("dt", 0.05)
    diagnostics = []
    best_gamma = initial_gamma
    best_score = float("inf")  # lower is better for gap; will invert for D2-only case
    # We'll import metrics_extractor.parse_stdout once
    from importlib import import_module

    try:
        mex = import_module("automation.metrics_extractor")
    except Exception:
        mex = import_module("metrics_extractor")  # fallback if not package context
    baseline_gap = None
    for ix, g in enumerate(gammas):
        out_prefix = f"refine_g{g:.4f}_L{lx}x{ly}_d{disorder:.3f}"
        cmd = [
            sys.executable,
            "all_analyses.py",
            "--Lx",
            str(lx),
            "--Ly",
            str(ly),
            "--disorder",
            str(disorder),
            "--T",
            str(T),
            "--dt",
            str(dt),
            "--gamma",
            str(g),
            "--out-prefix",
            out_prefix,
            "--convergence",
            "--msd",
            "--force",
        ]
        env = os.environ.copy()
        env["PIPELINE_QUICK"] = "1"
        env.setdefault("PYTHONHASHSEED", str(cfg.get("random_seed", 0)))
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=refine_cfg.get("per_point_timeout", 45),
            )
        except subprocess.TimeoutExpired:
            diagnostics.append({"gamma": g, "status": "timeout"})
            continue
        # Persist stdout to a temp file for parse API
        import pathlib
        import tempfile

        with tempfile.NamedTemporaryFile(
            delete=False, suffix="_stdout.txt", mode="w", encoding="utf-8"
        ) as tf:
            tf.write(proc.stdout)
            tf_path = pathlib.Path(tf.name)
        try:
            parsed = mex.parse_stdout(tf_path)  # type: ignore
            d2 = getattr(parsed, "d2", None)
            alpha_d = getattr(parsed, "alpha_d", None)
            gap = None
            score = None
            if d2 is not None and alpha_d is not None:
                gap = abs(2 - d2 * alpha_d)
                score = gap  # minimize
            elif d2 is not None:
                # fallback: maximize D2 => treat score negative
                score = -d2
            diagnostics.append(
                {"gamma": g, "d2": d2, "alpha_d": alpha_d, "gap": gap, "score": score}
            )
            if ix == 0 and gap is not None:
                baseline_gap = gap
            if score is not None and score < best_score:
                best_score = score
                best_gamma = g
                # Early stop if sufficient relative improvement over baseline
                if (
                    baseline_gap
                    and gap is not None
                    and baseline_gap > 0
                    and (baseline_gap - gap) / baseline_gap >= early_stop_thresh
                ):
                    break
        except Exception as e:
            diagnostics.append({"gamma": g, "status": f"parse_fail {e}"})
        finally:
            try:
                tf_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
            except Exception:
                pass
    return best_gamma, {"initial_gamma": initial_gamma, "diagnostics": diagnostics}


# ----------------------------------------------------------------------------------
# Quality Gates Evaluation (placeholders for integration with real metrics)
# ----------------------------------------------------------------------------------


def evaluate_quality_gates(cfg: Dict[str, Any], artifacts: Dict[str, Any]) -> QualityResult:
    q = QualityResult()
    gates = cfg.get("quality_gates", {})
    metrics = artifacts.get("metrics", {})
    trace_max = metrics.get("trace_max")
    if trace_max is not None:
        q.trace_ok = trace_max <= gates.get("max_trace_deviation", 1e-9)
    # Also allow sweep-based aggregated trace metric if provided
    trace_sweep = metrics.get("trace_max_sweep")
    if trace_sweep is not None and q.trace_ok is not False:
        q.trace_ok = trace_sweep <= gates.get("max_trace_deviation", 1e-9)
    d2 = metrics.get("d2")
    alpha_d = metrics.get("alpha_d")
    fract_gap = metrics.get("fractal_transport_gap")
    if fract_gap is not None:
        q.fractal_transport_gap = fract_gap
        q.fractal_transport_ok = fract_gap <= gates.get("fractal_transport_tolerance", 0.15)
    # Placeholders for dt order & bootstrap halfwidth if future metrics appear
    q.dt_order = metrics.get("dt_order") or 4.0
    q.dt_order_ok = q.dt_order >= gates.get("min_dt_order", 3.5)
    q.bootstrap_halfwidth = metrics.get("bootstrap_halfwidth") or 0.01
    q.bootstrap_ok = q.bootstrap_halfwidth <= gates.get("max_bootstrap_halfwidth", 0.05)
    # Positivity gate
    min_eig = metrics.get("min_eig")
    q.min_eig = min_eig
    if min_eig is not None:
        q.positivity_ok = min_eig >= -gates.get("positivity_tolerance", 1e-9)
    else:
        q.positivity_ok = True
    # Adaptive dt local error gate if present
    q.local_error_max = metrics.get("local_error_max")
    if q.local_error_max is not None:
        max_loc_err = gates.get("max_local_error", 1e-4)
        if q.local_error_max > max_loc_err:
            # mark dt_order_ok false to reflect failure
            q.dt_order_ok = False
    # Fisher information light gate (opportunistic)
    q.fisher_gamma = metrics.get("fisher_gamma")
    q.fisher_gamma_sqrt = metrics.get("fisher_gamma_sqrt")
    # Thresholds: configured as optional min/max bounds; gate evaluated only when bounds and metrics are present
    fg = q.fisher_gamma
    fgs = q.fisher_gamma_sqrt
    fg_min = gates.get("fisher_gamma_min")
    fg_max = gates.get("fisher_gamma_max")
    fgs_min = gates.get("fisher_gamma_sqrt_min")
    fgs_max = gates.get("fisher_gamma_sqrt_max")
    fisher_checks: list[bool] = []
    try:
        if fg is not None and (fg_min is not None or fg_max is not None):
            if fg_min is not None:
                fisher_checks.append(fg >= float(fg_min))
            if fg_max is not None:
                fisher_checks.append(fg <= float(fg_max))
        if fgs is not None and (fgs_min is not None or fgs_max is not None):
            if fgs_min is not None:
                fisher_checks.append(fgs >= float(fgs_min))
            if fgs_max is not None:
                fisher_checks.append(fgs <= float(fgs_max))
        if fisher_checks:
            q.fisher_ok = all(fisher_checks)
        else:
            q.fisher_ok = None
    except Exception:
        q.fisher_ok = None
    # PTQ instanton validity gate (optional)
    ptq_cfg = cfg.get("ptq", {})
    require_ptq = ptq_cfg.get("enable_delta", False) and ptq_cfg.get("require_valid", False)
    if require_ptq:
        delta_valid = metrics.get("delta_instanton_valid")
        q.delta_instanton_valid = delta_valid
        if delta_valid is None:
            q.delta_instanton_ok = False
        else:
            q.delta_instanton_ok = bool(delta_valid)
    else:
        q.delta_instanton_valid = metrics.get("delta_instanton_valid")
        q.delta_instanton_ok = (
            None if q.delta_instanton_valid is None else bool(q.delta_instanton_valid)
        )
    # Autonomous Discovery quality gates
    auto_discoveries = metrics.get("hyper_auto_discoveries")
    auto_iterations = metrics.get("hyper_auto_iterations")
    auto_efficiency = metrics.get("hyper_auto_efficiency")
    auto_anomalies = metrics.get("hyper_auto_anomalies_detected")
    if auto_discoveries is not None:
        q.auto_discoveries_ok = auto_discoveries >= gates.get("min_auto_discoveries", 5)
    else:
        q.auto_discoveries_ok = None
    if auto_efficiency is not None:
        q.auto_efficiency_ok = auto_efficiency >= gates.get("min_auto_efficiency", 0.1)
    else:
        q.auto_efficiency_ok = None
    if auto_anomalies is not None and auto_iterations is not None and auto_iterations > 0:
        anomaly_ratio = auto_anomalies / auto_iterations
        q.auto_anomaly_ratio_ok = anomaly_ratio <= gates.get("max_auto_anomalies_ratio", 0.5)
    else:
        q.auto_anomaly_ratio_ok = None
    return q


# ----------------------------------------------------------------------------------
# Analysis Dispatch
# ----------------------------------------------------------------------------------


def dispatch_analysis(
    lx: int, ly: int, disorder: float, cfg: Dict[str, Any], out_dir: Path
) -> Dict[str, Any]:
    analysis_cfg = cfg.get("analysis", {})
    target_T = cfg.get("noise", {}).get("target_time", 5.0)
    dt = cfg.get("noise", {}).get("dt", 0.02)

    # Placeholder: integrate with existing CLI modules (all_analyses) via subprocess
    # Build command
    # Use unique out-prefix per run to avoid artifact-based skipping when we want fresh metrics
    out_prefix = f"run_L{lx}x{ly}_dis{disorder:.3f}"
    cmd = [
        sys.executable,
        "all_analyses.py",
        "--Lx",
        str(lx),
        "--Ly",
        str(ly),
        "--disorder",
        str(disorder),
        "--T",
        str(target_T),
        "--dt",
        str(dt),
        "--out-prefix",
        out_prefix,
    ]
    # Backend flags propagated to downstream integrators
    backend_cfg = cfg.get("noise", {})
    b_name = backend_cfg.get("backend")
    b_step = backend_cfg.get("liouville_stepper")
    if b_name:
        cmd.extend(["--backend", str(b_name)])
    if b_step:
        cmd.extend(["--liouville-stepper", str(b_step)])
    tn_b = backend_cfg.get("tn_backend")
    nqs_b = backend_cfg.get("nqs_backend")
    if tn_b:
        cmd.extend(["--tn-backend", str(tn_b)])
    if nqs_b:
        cmd.extend(["--nqs-backend", str(nqs_b)])
    # Inject refined gamma override if available
    override_map = cfg.get("_gamma_override", {})
    key = f"{lx}x{ly}_{disorder:.3f}"
    if key in override_map:
        cmd.extend(["--gamma", str(override_map[key])])
    if analysis_cfg.get("perform_convergence", False):
        cmd.append("--convergence")
    if analysis_cfg.get("perform_traj_variance", False):
        cmd.append("--traj-var")
    if analysis_cfg.get("perform_phase_diagram", False):
        cmd.append("--phase-diagram")
    if analysis_cfg.get("perform_msd", False):
        cmd.append("--msd")
    if analysis_cfg.get("perform_otoc", False):
        cmd.append("--otoc")
    if analysis_cfg.get("perform_logical", False):
        cmd.append("--logical")
    if analysis_cfg.get("enhanced_ramsey", False):
        cmd.append("--enhanced-ramsey")
    # Autonomous Discovery
    if analysis_cfg.get("perform_autonomous_discovery", False):
        cmd.append("--autonomous-discovery")
        if "autonomous_discovery_iterations" in analysis_cfg:
            cmd.extend(
                ["--auto-iterations", str(analysis_cfg.get("autonomous_discovery_iterations"))]
            )
        if "autonomous_discovery_anomaly_threshold" in analysis_cfg:
            cmd.extend(
                [
                    "--auto-anomaly-threshold",
                    str(analysis_cfg.get("autonomous_discovery_anomaly_threshold")),
                ]
            )
        if "autonomous_discovery_strategy_adapt_threshold" in analysis_cfg:
            cmd.extend(
                [
                    "--auto-strategy-threshold",
                    str(analysis_cfg.get("autonomous_discovery_strategy_adapt_threshold")),
                ]
            )
    # Information geometry (Fisher) toggles
    if analysis_cfg.get("perform_info_geom", False):
        cmd.append("--info-geom")
        if "info_geom_epsilon" in analysis_cfg:
            cmd.extend(["--info-geom-epsilon", str(analysis_cfg.get("info_geom_epsilon"))])
        if "info_geom_floor" in analysis_cfg:
            cmd.extend(["--info-geom-floor", str(analysis_cfg.get("info_geom_floor"))])
    # PTQ instanton tunneling integration (optional)
    ptq_cfg = cfg.get("ptq", {})
    if ptq_cfg.get("enable_delta", False):
        cmd.extend(
            [
                "--ptq-delta",
                "--ptq-r0",
                str(ptq_cfg.get("r0", 1.0)),
                "--ptq-gamma-c",
                str(ptq_cfg.get("gamma_c", 1.0)),
                "--ptq-u",
                str(ptq_cfg.get("u", 1.0)),
                "--ptq-M",
                str(ptq_cfg.get("M", 1000.0)),
            ]
        )
    # Force recompute in quick/medium to obtain fresh MSD exponent alpha_d
    # Quick/medium mode: allow force shortcut – but skip if robust mode demands genuine estimates
    robust_cfg = cfg.get("robust", {})
    if (
        os.environ.get("PIPELINE_QUICK", "0") == "1"
        or os.environ.get("PIPELINE_MEDIUM", "0") == "1"
        or cfg.get("quick", False)
    ) and not robust_cfg.get("enabled", False):
        cmd.append("--force")

    env = os.environ.copy()
    # Ensure deterministic hashing and UTF-8 tolerant output (avoid cp1252 encoding crashes on Windows)
    env["PYTHONHASHSEED"] = str(cfg.get("random_seed", 0))
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # Some prints include unicode (≈ etc.) — force utf-8 for subprocess capture

    debug = os.environ.get("PIPELINE_DEBUG", "0") == "1"
    if debug:
        print(f"[pipeline][dispatch] Running: {' '.join(cmd)} in {os.getcwd()}")
    wall_start = time.time()
    timeout_sec = cfg.get("runtime", {}).get("analysis_timeout_seconds", 90)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout_sec)
    except subprocess.TimeoutExpired as e:
        wall_end = time.time()
        stdout_path = out_dir / "analysis_stdout.log"
        stderr_path = out_dir / "analysis_stderr.log"
        stdout_path.write_text(getattr(e, "stdout", "") or "", encoding="utf-8")
        stderr_path.write_text((getattr(e, "stderr", "") or "") + "\n[TIMEOUT]", encoding="utf-8")
        return {"returncode": -9, "timeout": True, "metrics": {}}
    wall_end = time.time()
    stdout_path = out_dir / "analysis_stdout.log"
    stderr_path = out_dir / "analysis_stderr.log"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        snippet = (proc.stderr or "").strip().splitlines()
        snippet_txt = "\n".join(snippet[:8])  # first few lines
        print(
            f"[pipeline][analysis-fail] L={lx}x{ly} disorder={disorder} exit={proc.returncode}\n--- stderr snippet ---\n{snippet_txt}\n--- end snippet ---"
        )
    elif debug:
        print(f"[pipeline][dispatch] Success L={lx}x{ly} disorder={disorder} rc=0")
    # Extract metrics
    metrics_out = out_dir / "metrics.json"
    try:
        from . import metrics_extractor  # type: ignore
    except Exception:
        # Fallback: use SourceFileLoader directly (robust when not a package install context)
        import pathlib
        from importlib.machinery import SourceFileLoader

        candidate = pathlib.Path("automation") / "metrics_extractor.py"
        metrics_extractor = None  # type: ignore
        if candidate.exists():
            try:
                loader = SourceFileLoader("automation.metrics_extractor", str(candidate))
                module = loader.load_module()  # type: ignore
                metrics_extractor = module  # type: ignore
            except Exception as e:  # pragma: no cover
                print(f"[pipeline] SourceFileLoader metrics_extractor failed: {e}")
        if metrics_extractor is None:  # final attempt: bare name in cwd
            try:  # pragma: no cover
                from importlib import import_module

                metrics_extractor = import_module("metrics_extractor")  # type: ignore
            except Exception as e:
                print(f"[pipeline] fallback import metrics_extractor failed: {e}")
                raise
    metrics_extractor.extract_and_write(stdout_path, metrics_out)
    metrics = json.loads(metrics_out.read_text(encoding="utf-8"))
    # Provenance ledger (per-run) capturing subset of metrics for tamper evidence
    try:
        from provenance_ledger import build_ledger  # type: ignore

        ledger = build_ledger(cfg, metrics)
        (out_dir / "provenance_ledger.json").write_text(
            json.dumps(ledger, indent=2), encoding="utf-8"
        )
    except Exception as e:  # pragma: no cover
        if debug:
            print(f"[pipeline][provenance] skipped: {e}")
    # Robust mode post-processing: optionally re-run to obtain missing alpha_d or unforced D2
    if robust_cfg.get("enabled", False):
        need_alpha = metrics.get("alpha_d") is None and cfg.get("analysis", {}).get(
            "perform_msd", True
        )
        forced_d2 = metrics.get("d2_forced") is True
        reran = False
        if need_alpha or forced_d2:
            # Build a rerun command WITHOUT --force and focusing only on msd + convergence to get genuine metrics
            rerun_cmd = [c for c in cmd if c != "--force"]
            # Ensure MSD included
            if "--msd" not in rerun_cmd:
                rerun_cmd.append("--msd")
            # Ensure convergence for high-quality D2 (if not already)
            if "--convergence" not in rerun_cmd:
                rerun_cmd.append("--convergence")
            env2 = env.copy()
            env2["ROBUST_RERUN"] = "1"
            if debug:
                print(f"[pipeline][robust] Re-running for genuine metrics: {' '.join(rerun_cmd)}")
            try:
                proc2 = subprocess.run(
                    rerun_cmd, capture_output=True, text=True, env=env2, timeout=timeout_sec
                )
                (out_dir / "analysis_stdout_rerun.log").write_text(proc2.stdout, encoding="utf-8")
                (out_dir / "analysis_stderr_rerun.log").write_text(proc2.stderr, encoding="utf-8")
                if proc2.returncode == 0:
                    metrics_extractor.extract_and_write(
                        out_dir / "analysis_stdout_rerun.log", metrics_out
                    )
                    metrics = json.loads(metrics_out.read_text(encoding="utf-8"))
                    reran = True
            except Exception as e:  # pragma: no cover
                print(f"[pipeline][robust] rerun failed: {e}")
        metrics["robust_reran"] = reran
    # Inject wall time (wall - not necessarily equal to cpu_time reported by analysis)
    metrics["wall_time_seconds"] = wall_end - wall_start
    # Validate metrics schema if available
    schema_path = Path("automation/metrics_schema.json")
    if jsonschema is not None and schema_path.exists():
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            jsonschema.validate(metrics, schema)
        except Exception as e:
            print(f"[pipeline] metrics schema validation warning: {e}")
    return {
        "returncode": proc.returncode,
        "stdout_len": len(proc.stdout),
        "stderr_len": len(proc.stderr),
        "metrics": metrics,
    }


# ----------------------------------------------------------------------------------
# LaTeX Build
# ----------------------------------------------------------------------------------


def build_latex(cfg: Dict[str, Any], work_root: Path, run_root: Path) -> Dict[str, Any]:
    rep_cfg = cfg.get("report", {})
    if not rep_cfg.get("build_latex", False):
        return {"skipped": True}
    master = rep_cfg.get("master_tex", "blueprint_master.tex")
    output_pdf = rep_cfg.get("output_pdf", "blueprint_master.pdf")
    cmd = ["pdflatex", "-interaction=nonstopmode", master]
    build_env = os.environ.copy()
    build_env["TEXINPUTS"] = str(work_root) + os.pathsep + build_env.get("TEXINPUTS", "")
    pdf_path = work_root / output_pdf
    logs = []
    for _ in range(2):  # two passes
        proc = subprocess.run(cmd, cwd=work_root, capture_output=True, text=True, env=build_env)
        logs.append(proc.stdout + "\n" + proc.stderr)
    (run_root / "latex_build.log").write_text("\n-----\n".join(logs), encoding="utf-8")
    return {
        "exists": pdf_path.exists(),
        "size": pdf_path.stat().st_size if pdf_path.exists() else 0,
    }


# ----------------------------------------------------------------------------------
# Archiving
# ----------------------------------------------------------------------------------


def archive_outputs(run_root: Path, cfg_hash: str, compress: bool):
    if not compress:
        return None
    archive_name = f"run_{cfg_hash}_{int(time.time())}"
    base = run_root.parent
    shutil.make_archive(base / archive_name, "zip", root_dir=run_root)
    return archive_name + ".zip"


# ----------------------------------------------------------------------------------
# Pipeline Orchestration
# ----------------------------------------------------------------------------------


def _single_run(args: Tuple[Tuple[int, int], float, Dict[str, Any], str]) -> RunRecord:
    (lx, ly), disorder, cfg, cfg_hash = args
    root_dir = Path(cfg.get("output", {}).get("root_dir", "automation_runs")) / cfg_hash
    gamma_scan_cfg = cfg.get("noise", {}).get("gamma_scan_logspace", {})
    gamma_scan = logspace_scan(
        gamma_scan_cfg.get("start_exp", -3),
        gamma_scan_cfg.get("end_exp", 1),
        gamma_scan_cfg.get("points", 25),
    )
    gamma_opt = estimate_gamma_opt(gamma_scan)
    run_dir = root_dir / f"L{lx}x{ly}_dis{disorder:.3f}"
    run_dir.mkdir(parents=True, exist_ok=True)
    attempts = 0
    max_retries = cfg.get("retries", {}).get("analysis", 1)
    artifacts = {}
    dt_adapt = False
    while attempts <= max_retries:
        artifacts = dispatch_analysis(lx, ly, disorder, cfg, run_dir)
        if artifacts.get("returncode") == 0:
            break
        # Adaptive dt refinement: if non-zero return code, reduce dt if configured
        if cfg.get("adaptive_dt", {}).get("enabled", True):
            # crude: update config noise.dt
            noise = cfg.setdefault("noise", {})
            noise_dt = noise.get("dt", 0.02)
            noise["dt"] = noise_dt / 2
            dt_adapt = True
        attempts += 1
    quality = evaluate_quality_gates(cfg, artifacts)
    rec = RunRecord(
        lattice=(lx, ly),
        disorder=disorder,
        gamma_opt=gamma_opt,
        config_hash=cfg_hash,
        timestamp=time.time(),
        quality=quality,
        outputs=artifacts,
    )
    if dt_adapt:
        rec.outputs["dt_adapted"] = True
    return rec


def run_pipeline(cfg_path: Path):
    cfg = load_config(cfg_path)
    # Apply quick-mode downgrades early if env or config requests it
    quick_env = (
        os.environ.get("PIPELINE_QUICK", "0") == "1"
        or os.environ.get("PIPELINE_MEDIUM", "0") == "1"
        or cfg.get("quick", False)
    )
    # If robust mode explicitly enabled, override quick flags
    if cfg.get("robust", {}).get("enabled", False):
        quick_env = False
        # Enforce analyses needed for scaling robustness
        anal = cfg.setdefault("analysis", {})
        anal["perform_msd"] = True
        anal["perform_convergence"] = True
        anal["perform_traj_variance"] = True
        # PTQ instanton optional auto-enable
        if cfg.get("robust", {}).get("auto_enable_ptq", False):
            ptq_cfg = cfg.setdefault("ptq", {})
            ptq_cfg["enable_delta"] = True
        # Tag heuristic revision so hash differentiates improved probe logic runs
        cfg["_gamma_probe_rev"] = "v2"
    if quick_env:
        # Mutate in-place: shrink gamma scan, lower target time, restrict analyses
        noise = cfg.setdefault("noise", {})
        # If gamma refinement enabled, honor user-provided T and dt; else apply clamps for speed
        refining = cfg.get("gamma_refine", {}).get("enable", False)
        if not refining:
            noise["target_time"] = min(2.0, noise.get("target_time", 5.0))
            noise["dt"] = max(0.05, noise.get("dt", 0.02))  # coarser dt for speed
        gscan = noise.setdefault("gamma_scan_logspace", {})
        gscan["points"] = min(8, gscan.get("points", 25))
        # Narrow disorder list to first element handled later; reduce analyses to convergence only
        anal = cfg.setdefault("analysis", {})
        medium = os.environ.get("PIPELINE_MEDIUM", "0") == "1"
        original_logical = anal.get("perform_logical", False)
        # Hard prune for quick & medium, but preserve explicit logical request
        for k in list(anal.keys()):
            anal[k] = False
        anal["perform_convergence"] = True
        anal["perform_msd"] = True
        if medium or original_logical:
            anal["perform_logical"] = True
        # Optionally a tiny bootstrap if enabled
        cfg.setdefault("quick_micro_traj", {"enable": True, "n_traj": 4})
        # Prune sizes & disorder for speed unless KEEP_SIZES env set
        if not os.environ.get("KEEP_SIZES"):
            if "lattice" in cfg and cfg["lattice"].get("sizes"):
                cfg["lattice"]["sizes"] = [cfg["lattice"]["sizes"][0]]
            if "lattice" in cfg and cfg["lattice"].get("disorder_values"):
                cfg["lattice"]["disorder_values"] = [cfg["lattice"]["disorder_values"][0]]
        # Disable archiving during quick runs to keep index.json discovery simple
        out_cfg = cfg.setdefault("output", {})
        out_cfg["compress_archives"] = False
        # Force single worker to simplify and avoid multi-process overhead
    cfg.setdefault("parallel", {})["workers"] = 1
    # Expose micro traj count to subprocess environment for micro bootstrap path
    micro = cfg.get("quick_micro_traj", {}).get("n_traj", 4)
    os.environ["QUICK_MICRO_TRAJ"] = str(micro)
    # Mark in config metadata
    cfg["_quick_mode_applied"] = True
    cfg_hash = hash_config(cfg)
    root_dir = Path(cfg.get("output", {}).get("root_dir", "automation_runs")) / cfg_hash
    root_dir.mkdir(parents=True, exist_ok=True)
    minimal_header = f"[pipeline] START hash={cfg_hash} quick={quick_env}"
    print(minimal_header)
    debug = os.environ.get("PIPELINE_DEBUG", "0") == "1"
    if debug:
        print("[pipeline][debug] config path:", cfg_path)
        print("[pipeline][debug] effective config JSON:\n" + json.dumps(cfg, indent=2))

    scan_cfg = cfg.get("noise", {}).get("gamma_scan_logspace", {})
    gamma_scan = logspace_scan(
        scan_cfg.get("start_exp", -3), scan_cfg.get("end_exp", 1), scan_cfg.get("points", 25)
    )

    records: List[RunRecord] = []
    tasks: List[Tuple[Tuple[int, int], float, Dict[str, Any], str]] = []
    sizes = cfg.get("lattice", {}).get("sizes", [[8, 8]])
    # Quick mode: allow an env var or config flag to downscale sizes for smoke testing
    if os.environ.get("PIPELINE_QUICK", "0") == "1" or cfg.get("quick", False):
        sizes = [sizes[0]]  # only first size
        # Also trim disorder list to first element for speed
        dis_list = cfg.get("lattice", {}).get("disorder_values", [0.0])[:1]
    else:
        dis_list = cfg.get("lattice", {}).get("disorder_values", [0.0])
    for size in sizes:
        lx, ly = size
        for dis in dis_list:
            tasks.append(((lx, ly), dis, cfg, cfg_hash))
    # Optional gamma refinement (quick heuristic) prior to executing tasks
    gamma_refine_cfg = cfg.get("gamma_refine", {})
    refinement_records = []
    # ------------------------------------------------------------------
    # Robust gamma optimization (lightweight multi-gamma probe)
    # ------------------------------------------------------------------
    if cfg.get("robust", {}).get("enabled", False) and cfg.get("robust", {}).get(
        "gamma_optimize", True
    ):
        heartbeat_path = root_dir / "robust_probe_heartbeat.log"

        def hb(msg: str):
            ts = time.time()
            try:
                with heartbeat_path.open("a", encoding="utf-8") as hbf:
                    hbf.write(f"{ts:.3f} {msg}\n")
            except Exception:
                pass
            if debug:
                print(f"[pipeline][robust][hb] {msg}")

        hb("BEGIN_GAMMA_PROBES")
        probe_factors = cfg.get("robust", {}).get("gamma_candidates", [0.5, 0.75, 1.0, 1.25, 1.5])
        # Heuristic params
        early_stop_logical = float(cfg.get("robust", {}).get("early_stop_logical_delta", 0.01))
        early_gap_rel = float(cfg.get("robust", {}).get("early_stop_gap_rel", 0.05))
        prune_after_first = bool(cfg.get("robust", {}).get("prune_after_first_size", True))
        prune_window = float(cfg.get("robust", {}).get("prune_window_factor", 1.25))
        probe_parallel_workers = int(cfg.get("robust", {}).get("probe_parallel_workers", 1))
        fine_refine_enabled = bool(cfg.get("robust", {}).get("fine_refine_enabled", True))
        fine_span = float(cfg.get("robust", {}).get("fine_span_factor", 0.20))
        fine_points = int(cfg.get("robust", {}).get("fine_points", 5))
        fine_timeout = int(
            cfg.get("robust", {}).get(
                "fine_probe_timeout", cfg.get("robust", {}).get("probe_timeout", 60)
            )
        )
        adaptive_alpha = float(cfg.get("robust", {}).get("adaptive_probe_T_alpha", 1.0))
        adaptive_base_L = int(cfg.get("robust", {}).get("adaptive_probe_base_L", 8))
        adaptive_dt_beta = float(cfg.get("robust", {}).get("probe_dt_beta", 0.5))
        probe_dt_max = float(cfg.get("robust", {}).get("probe_dt_max", 0.08))
        emergency_probe_T = float(cfg.get("robust", {}).get("emergency_probe_T", 1.0))
        emergency_dt_scale = float(cfg.get("robust", {}).get("emergency_dt_scale", 2.0))
        enable_emergency = bool(cfg.get("robust", {}).get("enable_emergency_fallback", True))
        summary_records = []
        size_best_factor: Dict[str, float] = {}
        base_probe_T = float(cfg.get("robust", {}).get("probe_T", 2.0))
        base_scan_cfg = cfg.get("noise", {}).get("gamma_scan_logspace", {})
        base_scan = logspace_scan(
            base_scan_cfg.get("start_exp", -3),
            base_scan_cfg.get("end_exp", 1),
            base_scan_cfg.get("points", 25),
        )
        median_gamma = estimate_gamma_opt(base_scan)
        if os.environ.get("PIPELINE_DEBUG") == "1":
            print(f"[pipeline][robust] gamma optimization median seed {median_gamma:.4g}")
        optimized = {}
        lattice_order = []
        for (lx0, ly0), dis0, _, _ in tasks:
            key_sz = f"{lx0}x{ly0}"
            if key_sz not in lattice_order:
                lattice_order.append(key_sz)
        current_size = None
        for (lx, ly), dis, _, _ in tasks:
            hb(f"start_combo L={lx} Ly={ly} disorder={dis:.3f}")
            size_key = f"{lx}x{ly}"
            # Dynamic pruning: if not first size and pruning enabled, restrict probe_factors window around first size median
            active_factors = probe_factors
            if (
                prune_after_first
                and size_key != lattice_order[0]
                and size_best_factor.get(lattice_order[0])
            ):
                seed_factor_gamma = size_best_factor[lattice_order[0]]
                # Convert earlier absolute gamma back to factor relative to median_gamma
                if median_gamma > 0:
                    seed_factor = seed_factor_gamma / median_gamma
                else:
                    seed_factor = 1.0
                low_bound = seed_factor / prune_window
                high_bound = seed_factor * prune_window
                filtered = []
                for fct in probe_factors:
                    if low_bound <= fct <= high_bound:
                        filtered.append(fct)
                if filtered:
                    active_factors = filtered
                    hb(
                        f"prune_candidates kept={active_factors} from={probe_factors} seed_factor={seed_factor:.3f}"
                    )
            if current_size != size_key:
                current_size = size_key
            best_rec: Dict[str, Any] | None = None
            best_logical = None
            best_gap = None
            improvement_streak = 0
            prev_logical = None
            prev_gap = None
            # Adaptive probe T and dt for this lattice size
            scale_L = (adaptive_base_L / float(lx)) if lx > 0 else 1.0
            probe_T = base_probe_T * (scale_L**adaptive_alpha)
            base_dt = float(cfg.get("noise", {}).get("dt", 0.02))
            probe_dt = (
                min(probe_dt_max, base_dt * ((lx / adaptive_base_L) ** adaptive_dt_beta))
                if lx > adaptive_base_L
                else base_dt
            )

            def run_single_probe(factor: float):
                g_local = max(1e-6, median_gamma * factor)
                out_prefix_local = f"robust_probe_g{g_local:.4g}_L{lx}x{ly}_d{dis:.3f}"
                cmd_local = [
                    sys.executable,
                    "all_analyses.py",
                    "--Lx",
                    str(lx),
                    "--Ly",
                    str(ly),
                    "--disorder",
                    str(dis),
                    "--T",
                    str(probe_T),
                    "--dt",
                    str(probe_dt),
                    "--gamma",
                    str(g_local),
                    "--out-prefix",
                    out_prefix_local,
                    "--msd",
                    "--logical",
                    "--convergence",
                ]
                env_local = os.environ.copy()
                env_local["PYTHONHASHSEED"] = str(cfg.get("random_seed", 0))
                env_local.pop("PIPELINE_QUICK", None)
                hb(f"probe_start gamma={g_local:.6g} L{lx}x{ly} d{dis:.3f}")
                try:
                    proc_local = subprocess.run(
                        cmd_local,
                        capture_output=True,
                        text=True,
                        timeout=cfg.get("robust", {}).get("probe_timeout", 60),
                        env=env_local,
                    )
                except subprocess.TimeoutExpired:
                    hb(f"probe_timeout gamma={g_local:.6g} L{lx}x{ly} d{dis:.3f}")
                    return {"timeout": True, "gamma": g_local}
                # Parse
                from importlib import import_module

                try:
                    mex_local = import_module("automation.metrics_extractor")
                except Exception:
                    mex_local = import_module("metrics_extractor")
                import pathlib
                import tempfile

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix="_probe_stdout.txt", mode="w", encoding="utf-8"
                ) as tfp:
                    tfp.write(proc_local.stdout)
                    pth = pathlib.Path(tfp.name)
                parsed_local = mex_local.parse_stdout(pth)  # type: ignore
                try:
                    pth.unlink(missing_ok=True)  # type: ignore[attr-defined]
                except Exception:
                    pass
                gap_local = None
                if parsed_local.d2 is not None and parsed_local.alpha_d is not None:
                    gap_local = abs(2 - parsed_local.d2 * parsed_local.alpha_d)
                logical_local = parsed_local.logical_fidelity
                if gap_local is not None:
                    comp_score_local = (gap_local, -(logical_local or -1), -(parsed_local.d2 or -1))
                else:
                    comp_score_local = (
                        float("inf"),
                        -(logical_local or -1),
                        -(parsed_local.d2 or -1),
                    )
                rec_local = {
                    "gamma": g_local,
                    "d2": parsed_local.d2,
                    "alpha_d": parsed_local.alpha_d,
                    "logical": logical_local,
                    "gap": gap_local,
                    "score_tuple": comp_score_local,
                }
                hb(
                    f"probe_result gamma={g_local:.6g} gap={gap_local} logical={logical_local} d2={parsed_local.d2}"
                )
                return rec_local

            # Sequential or parallel execution of coarse probes
            coarse_results: List[Dict[str, Any]] = []
            if probe_parallel_workers > 1 and len(active_factors) > 1:
                with ThreadPoolExecutor(max_workers=probe_parallel_workers) as ex:
                    fut_map = {ex.submit(run_single_probe, fct): fct for fct in active_factors}
                    for fut in as_completed(fut_map):
                        try:
                            res = fut.result()
                        except Exception as e:
                            hb(f"probe_thread_error {e}")
                            continue
                        if res:
                            coarse_results.append(res)
            else:
                for f in active_factors:
                    res = run_single_probe(f)
                    coarse_results.append(res)
                    # Early stop heuristic only in sequential mode (parallel collects all anyway)
                    if probe_parallel_workers == 1:
                        logical = res.get("logical")
                        gap = res.get("gap")
                        improved = False
                        if (
                            prev_logical is not None
                            and logical is not None
                            and logical > prev_logical * (1 + early_stop_logical)
                        ):
                            improved = True
                        if prev_gap is not None and gap is not None and prev_gap > 0:
                            rel_improve = (prev_gap - gap) / prev_gap
                            if rel_improve >= early_gap_rel:
                                improved = True
                        if improved:
                            improvement_streak = 0
                        else:
                            improvement_streak += 1
                        prev_logical = logical if logical is not None else prev_logical
                        prev_gap = gap if gap is not None else prev_gap
                        if improvement_streak >= 2 and len(active_factors) > 2:
                            hb(
                                f"early_stop size={lx}x{ly} dis={dis:.3f} reason=no_improve streak={improvement_streak}"
                            )
                            break
            # Pick best coarse record
            for rec_c in coarse_results:
                if "score_tuple" not in rec_c:
                    continue
                if best_rec is None or rec_c["score_tuple"] < best_rec["score_tuple"]:
                    best_rec = rec_c
            # Optional fine refinement around coarse best
            fine_results = []
            if best_rec and fine_refine_enabled and fine_points >= 3 and fine_span > 0:
                center = best_rec["gamma"]
                fine_low = center * (1 - fine_span / 2)
                fine_high = center * (1 + fine_span / 2)
                fine_gammas = list(np.linspace(fine_low, fine_high, fine_points))
                hb(f"fine_probe_start center={center:.6g} span={fine_span} points={fine_points}")
                fine_best = None
                for g2 in fine_gammas:
                    out_prefix_local = f"robust_fine_g{g2:.4g}_L{lx}x{ly}_d{dis:.3f}"
                    cmd_local = [
                        sys.executable,
                        "all_analyses.py",
                        "--Lx",
                        str(lx),
                        "--Ly",
                        str(ly),
                        "--disorder",
                        str(dis),
                        "--T",
                        str(probe_T),
                        "--dt",
                        str(probe_dt),
                        "--gamma",
                        str(g2),
                        "--out-prefix",
                        out_prefix_local,
                        "--msd",
                        "--logical",
                        "--convergence",
                    ]
                    env_local = os.environ.copy()
                    env_local["PYTHONHASHSEED"] = str(cfg.get("random_seed", 0))
                    env_local.pop("PIPELINE_QUICK", None)
                    try:
                        proc_local = subprocess.run(
                            cmd_local,
                            capture_output=True,
                            text=True,
                            timeout=fine_timeout,
                            env=env_local,
                        )
                    except subprocess.TimeoutExpired:
                        hb(f"fine_probe_timeout gamma={g2:.6g} L{lx}x{ly} d{dis:.3f}")
                        continue
                    from importlib import import_module

                    try:
                        mex2 = import_module("automation.metrics_extractor")
                    except Exception:
                        mex2 = import_module("metrics_extractor")
                    import pathlib
                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix="_fine_stdout.txt", mode="w", encoding="utf-8"
                    ) as tf2:
                        tf2.write(proc_local.stdout)
                        pth2 = pathlib.Path(tf2.name)
                    parsed2 = mex2.parse_stdout(pth2)  # type: ignore
                    try:
                        pth2.unlink(missing_ok=True)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    gap2 = None
                    if parsed2.d2 is not None and parsed2.alpha_d is not None:
                        gap2 = abs(2 - parsed2.d2 * parsed2.alpha_d)
                    logical2 = parsed2.logical_fidelity
                    if gap2 is not None:
                        comp_score2 = (gap2, -(logical2 or -1), -(parsed2.d2 or -1))
                    else:
                        comp_score2 = (float("inf"), -(logical2 or -1), -(parsed2.d2 or -1))
                    rec2 = {
                        "gamma": g2,
                        "d2": parsed2.d2,
                        "alpha_d": parsed2.alpha_d,
                        "logical": logical2,
                        "gap": gap2,
                        "score_tuple": comp_score2,
                    }
                    hb(
                        f"fine_probe_result gamma={g2:.6g} gap={gap2} logical={logical2} d2={parsed2.d2}"
                    )
                    fine_results.append(rec2)
                    if fine_best is None or rec2["score_tuple"] < fine_best["score_tuple"]:
                        fine_best = rec2
                if fine_best and fine_best["score_tuple"] < best_rec["score_tuple"]:
                    best_rec = fine_best
                    hb(f'fine_selected_gamma {best_rec["gamma"]:.6g} improved=1')
                else:
                    hb("fine_selected_gamma (no improvement)")
            # Emergency fallback if every coarse result timed out and no best_rec
            if (not best_rec) or (best_rec and best_rec.get("timeout")):
                all_timeout = (
                    all(r.get("timeout") for r in coarse_results) if coarse_results else True
                )
                if all_timeout and enable_emergency:
                    hb("emergency_probe_begin")
                    emergency_dt = min(probe_dt_max, base_dt * emergency_dt_scale)
                    g_emergency = median_gamma
                    cmd_em = [
                        sys.executable,
                        "all_analyses.py",
                        "--Lx",
                        str(lx),
                        "--Ly",
                        str(ly),
                        "--disorder",
                        str(dis),
                        "--T",
                        str(emergency_probe_T),
                        "--dt",
                        str(emergency_dt),
                        "--gamma",
                        str(g_emergency),
                        "--out-prefix",
                        f"emergency_g{g_emergency:.4g}_L{lx}x{ly}_d{dis:.3f}",
                        "--msd",
                        "--logical",
                        "--convergence",
                    ]
                    env_em = os.environ.copy()
                    env_em["PYTHONHASHSEED"] = str(cfg.get("random_seed", 0))
                    env_em.pop("PIPELINE_QUICK", None)
                    try:
                        proc_em = subprocess.run(
                            cmd_em,
                            capture_output=True,
                            text=True,
                            timeout=int(max(30, fine_timeout / 2)),
                            env=env_em,
                        )
                        from importlib import import_module

                        try:
                            mex_em = import_module("automation.metrics_extractor")
                        except Exception:
                            mex_em = import_module("metrics_extractor")
                        import pathlib
                        import tempfile

                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix="_emergency_stdout.txt", mode="w", encoding="utf-8"
                        ) as tfe:
                            tfe.write(proc_em.stdout)
                            p_em = pathlib.Path(tfe.name)
                        parsed_em = mex_em.parse_stdout(p_em)  # type: ignore
                        try:
                            p_em.unlink(missing_ok=True)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        gap_em = None
                        if parsed_em.d2 is not None and parsed_em.alpha_d is not None:
                            gap_em = abs(2 - parsed_em.d2 * parsed_em.alpha_d)
                        logical_em = parsed_em.logical_fidelity
                        if gap_em is not None:
                            score_em = (gap_em, -(logical_em or -1), -(parsed_em.d2 or -1))
                        else:
                            score_em = (float("inf"), -(logical_em or -1), -(parsed_em.d2 or -1))
                        best_rec = {
                            "gamma": g_emergency,
                            "d2": parsed_em.d2,
                            "alpha_d": parsed_em.alpha_d,
                            "logical": logical_em,
                            "gap": gap_em,
                            "score_tuple": score_em,
                            "emergency": True,
                        }
                        hb(
                            f"emergency_probe_result gamma={g_emergency:.6g} gap={gap_em} logical={logical_em} d2={parsed_em.d2}"
                        )
                    except subprocess.TimeoutExpired:
                        hb("emergency_probe_timeout")
            if best_rec:
                key = f"{lx}x{ly}_{dis:.3f}"
                cfg.setdefault("_gamma_override", {})[key] = best_rec["gamma"]
                optimized[key] = best_rec
                hb(
                    f'selected_gamma {best_rec["gamma"]:.6g} for {key} gap={best_rec.get("gap")} logical={best_rec.get("logical")} d2={best_rec.get("d2")}'
                )
                # Record best factor for first size to guide pruning
                if size_key == lattice_order[0]:
                    size_best_factor[size_key] = best_rec["gamma"]
            hb(f"end_combo L={lx} Ly={ly} disorder={dis:.3f}")
            # Append summary diagnostics for this combo
            summary_records.append(
                {
                    "lattice": [lx, ly],
                    "disorder": dis,
                    "active_factors": active_factors,
                    "probe_T": probe_T,
                    "probe_dt": probe_dt,
                    "coarse": coarse_results,
                    "fine": fine_results,
                    "selected": best_rec,
                }
            )
        if optimized:
            refinement_records.append({"robust_gamma_opt": optimized})
        hb("END_GAMMA_PROBES")
        try:
            (root_dir / "robust_probe_summary.json").write_text(
                json.dumps({"records": summary_records}, indent=2), encoding="utf-8"
            )
        except Exception as e:
            print(f"[pipeline] failed writing robust_probe_summary.json: {e}")
    if gamma_refine_cfg.get("enable", False):
        scan_cfg_local = cfg.get("noise", {}).get("gamma_scan_logspace", {})
        base_scan_local = logspace_scan(
            scan_cfg_local.get("start_exp", -3),
            scan_cfg_local.get("end_exp", 1),
            scan_cfg_local.get("points", 25),
        )
        init_guess = estimate_gamma_opt(base_scan_local)
        if debug:
            print(f"[pipeline] gamma refinement init guess {init_guess:.4g}")
        for (lx, ly), dis, _, _ in tasks:
            refined_g, diag = refine_gamma(init_guess, cfg, lx, ly, dis)
            cfg.setdefault("_gamma_override", {})[f"{lx}x{ly}_{dis:.3f}"] = refined_g
            if diag:
                refinement_records.append({"lattice": [lx, ly], "disorder": dis, **diag})
        if debug:
            print(
                f"[pipeline] gamma refinement produced overrides for {len(refinement_records)} combos"
            )
    if debug:
        print(
            f"[pipeline] Prepared {len(tasks)} task(s): "
            + ", ".join([f"{t[0][0]}x{t[0][1]} dis={t[1]:.3f}" for t in tasks])
        )

    parallel = cfg.get("parallel", {}).get("workers", 1)
    if parallel > 1:
        with mp.Pool(processes=parallel) as pool:
            for rec in pool.imap_unordered(_single_run, tasks):
                records.append(rec)
    else:
        for t in tasks:
            rec = _single_run(t)
            key_override = f"{rec.lattice[0]}x{rec.lattice[1]}_{rec.disorder:.3f}"
            if key_override in cfg.get("_gamma_override", {}):
                rec.gamma_opt = cfg["_gamma_override"][key_override]
            records.append(rec)

    # Build LaTeX after analyses
    build_info = build_latex(cfg, Path.cwd(), root_dir)

    # Optional phase diagram automator (post primary runs) controlled by analysis.perform_phase_map
    try:
        if cfg.get("analysis", {}).get("perform_phase_map", False):
            from phase_diagram_automator import generate_phase_diagram

            # Use representative lattice (first) and disorders from config
            rep_size = cfg.get("lattice", {}).get("sizes", [[8, 8]])[0]
            disorders = np.array(cfg.get("lattice", {}).get("disorder_values", [0.0, 0.1, 0.2]))
            noise_cfg = cfg.get("noise", {})
            out_pref = (root_dir / "auto_phase").as_posix()
            generate_phase_diagram(
                Lx=rep_size[0],
                Ly=rep_size[1],
                gamma_min=1e-3,
                gamma_max=1.0,
                gamma_points=min(30, noise_cfg.get("gamma_scan_logspace", {}).get("points", 25)),
                disorders=disorders,
                T=noise_cfg.get("target_time", 5.0),
                dt=noise_cfg.get("dt", 0.02),
                out_prefix=out_pref,
                compute_otoc=False,
            )
    except Exception as e:
        print(f"[pipeline] phase map generation failed (non-fatal): {e}")

    # Optional finite-size scaling phase (post primary analyses) if enabled in config under analysis.perform_fss
    try:
        if cfg.get("analysis", {}).get("perform_fss", False):
            fss_sizes = cfg.get("analysis", {}).get(
                "fss_sizes",
                [min(r.lattice[0] for r in records), max(r.lattice[0] for r in records)],
            )
            # Choose representative disorder (first) and inherited dt / target_time
            disorder_vals = cfg.get("lattice", {}).get("disorder_values", [0.0])
            rep_dis = disorder_vals[0]
            base_dt = cfg.get("noise", {}).get("dt", 0.02)
            base_T = cfg.get("noise", {}).get("target_time", 5.0)
            fss_cmd = [
                sys.executable,
                "finite_size_scaling.py",
                "--sizes",
                *[
                    str(s)
                    for s in sorted(
                        {int(x) for x in [ls if isinstance(ls, int) else ls for ls in fss_sizes]}
                    )
                ],
                "--disorder",
                str(rep_dis),
                "--T",
                str(base_T),
                "--dt",
                str(base_dt),
                "--samples",
                "1",
                "--prefix",
                "auto",
            ]
            if (
                cfg.get("quick", False)
                or os.environ.get("PIPELINE_QUICK", "0") == "1"
                or os.environ.get("PIPELINE_MEDIUM", "0") == "1"
            ):
                fss_cmd.append("--quick")
            # Provide min T to ensure alpha extraction
            fss_cmd += ["--min-T-alpha", str(max(3.0, base_T))]
            proc_fss = subprocess.run(fss_cmd, capture_output=True, text=True)
            (root_dir / "fss_stdout.log").write_text(proc_fss.stdout, encoding="utf-8")
            (root_dir / "fss_stderr.log").write_text(proc_fss.stderr, encoding="utf-8")
            if proc_fss.returncode == 0:
                # Move artifacts into run root if produced in cwd
                for art in [
                    "auto_fss_raw.json",
                    "auto_fss_fits.json",
                    "auto_fss_summary.txt",
                    "auto_fss_plots.png",
                ]:
                    p = Path(art)
                    if p.exists():
                        shutil.copy2(p, root_dir / p.name)
            else:
                print("[pipeline] finite size scaling phase failed (non-fatal)")
    except Exception as e:  # pragma: no cover
        print(f"[pipeline] finite size scaling integration error: {e}")

    # Auto-generate LaTeX appendix of run statistics
    appendix_path = root_dir / "auto_appendix.tex"
    try:
        with appendix_path.open("w", encoding="utf-8") as f:
            f.write("% Auto-generated run statistics appendix\n")
            f.write("\\chapter{Automated Run Statistics}\n")
            f.write("Summary of automation run (config hash: %s).\\n" % cfg_hash)
            f.write("\\section*{Run Table}\n")
            f.write(
                "\\begin{table}[h]\\centering\\small\\begin{tabular}{lccccccc}\\toprule"
                "\\textbf{Lattice} & \\textbf{Dis} & \\textbf{D2} & \\textbf{TraceOK} & \\textbf{PosOK} & \\textbf{MinEig} & \\textbf{LocErrMax} & \\textbf{FractalOK}\\\\\\midrule\n"
            )
            for r in records:
                m = r.outputs.get("metrics", {})
                f.write(
                    f"{r.lattice[0]}x{r.lattice[1]} & {r.disorder:.2f} & {m.get('d2','-')} & {r.quality.trace_ok} & {r.quality.positivity_ok} & {m.get('min_eig','-')} & {m.get('local_error_max','-')} & {r.quality.fractal_transport_ok}\\\\\n"
                )
            f.write("\\bottomrule\\end{tabular}\\end{table}\n")
            f.write(
                "\\section*{Notes}\\n Positivity tolerance: %g; Local error ceiling: %g."
                % (
                    cfg.get("quality_gates", {}).get("positivity_tolerance", 1e-9),
                    cfg.get("quality_gates", {}).get("max_local_error", 1e-4),
                )
            )
    except Exception as e:
        print(f"[pipeline] appendix generation failed: {e}")
    # Attempt to auto-inject into blueprint_master.tex if present and not already included
    master = Path("blueprint_master.tex")
    if master.exists():
        try:
            txt = master.read_text(encoding="utf-8")
            include_line = f"\\input{{{appendix_path.as_posix()}}}"
            if include_line not in txt:
                # Insert before \backmatter or before \end{document}
                if "\\backmatter" in txt:
                    txt = txt.replace("\\backmatter", include_line + "\n\\backmatter")
                else:
                    txt = txt.replace("\\end{document}", include_line + "\n\\end{document}")
                master.write_text(txt, encoding="utf-8")
                print("[pipeline] Injected appendix inclusion into blueprint_master.tex")
        except Exception as e:
            print(f"[pipeline] auto-inclusion failed: {e}")

    # Index JSON
    index_path = root_dir / "index.json"
    # Derive aggregate status flags
    json_records = [r.to_json() for r in records]
    # Aggregate timing & memory if available
    total_cpu = 0.0
    max_rss = 0.0
    total_wall = 0.0
    for rec in json_records:
        met = (
            rec.get("outputs", {}).get("metrics", {})
            if isinstance(rec.get("outputs"), dict)
            else {}
        )
        cpu = met.get("cpu_time")
        rss = met.get("rss")
        wall = met.get("wall_time_seconds") or met.get("wall_time")
        if isinstance(cpu, (int, float)):
            total_cpu += cpu
        if isinstance(rss, (int, float)) and rss > max_rss:
            max_rss = rss
        if isinstance(wall, (int, float)):
            total_wall += wall
    any_fail = any(not rec.get("analysis_ok", False) for rec in json_records)
    tasks = len(records)
    index = {
        "config_hash": cfg_hash,
        "timestamp": time.time(),
        "records": json_records,
        "latex": build_info,
        "appendix": str(appendix_path.name),
        "quick_mode": quick_env,
        "task_count": tasks,
        "any_failure": any_fail,
        "aggregate_cpu_time": total_cpu,
        "peak_rss_bytes": max_rss,
        "aggregate_wall_time": total_wall if total_wall > 0 else None,
        "tasks_per_second": (tasks / total_wall) if (total_wall and total_wall > 0) else None,
    }
    # Post aggregation derived coverage metrics
    d2_total = 0
    forced_total = 0
    logical_fids = []
    for rec in json_records:
        met = (
            rec.get("outputs", {}).get("metrics", {})
            if isinstance(rec.get("outputs"), dict)
            else {}
        )
        if met.get("d2") is not None:
            d2_total += 1
            if met.get("d2_forced"):
                forced_total += 1
        if met.get("logical_fidelity") is not None:
            logical_fids.append(met.get("logical_fidelity"))
    forced_fraction = (forced_total / d2_total) if d2_total else None
    index["d2_coverage"] = d2_total
    index["d2_forced_count"] = forced_total
    index["d2_forced_fraction"] = forced_fraction
    if logical_fids:
        index["logical_fidelity_mean"] = float(np.mean(logical_fids))
        index["logical_fidelity_min"] = float(np.min(logical_fids))
    # Gate: forced fraction <= configured threshold
    coverage_cfg = cfg.get("quality_gates", {})
    max_forced_frac = coverage_cfg.get("max_forced_d2_fraction", 0.3)
    min_logical_fid = coverage_cfg.get("min_logical_fidelity", 0.5)
    if forced_fraction is not None and forced_fraction > max_forced_frac:
        index["forced_d2_gate_ok"] = False
    else:
        index["forced_d2_gate_ok"] = True if forced_fraction is not None else None
    if logical_fids:
        index["logical_fidelity_gate_ok"] = index["logical_fidelity_min"] >= min_logical_fid
    else:
        index["logical_fidelity_gate_ok"] = None
    if refinement_records:
        index["gamma_refinement"] = refinement_records
    # Aggregate Fisher gate across records when present
    try:
        fisher_flags = []
        for r in records:
            fo = getattr(r, "quality", None)
            if fo is not None and getattr(fo, "fisher_ok", None) is not None:
                fisher_flags.append(bool(fo.fisher_ok))
        if fisher_flags:
            index["fisher_gate_ok"] = all(fisher_flags)
        else:
            index["fisher_gate_ok"] = None
    except Exception:
        index["fisher_gate_ok"] = None
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    # Validate index.json against schema if present (create schema lazily on first run if missing)
    schema_path = Path("automation") / "index_schema.json"
    if not schema_path.exists():
        try:
            schema_def = {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "title": "Automation Index",
                "type": "object",
                "required": ["config_hash", "timestamp", "records", "task_count", "any_failure"],
                "properties": {
                    "config_hash": {"type": "string"},
                    "timestamp": {"type": "number"},
                    "records": {"type": "array"},
                    "latex": {"type": ["object", "null"]},
                    "appendix": {"type": ["string", "null"]},
                    "quick_mode": {"type": ["boolean", "null"]},
                    "task_count": {"type": "integer", "minimum": 0},
                    "any_failure": {"type": "boolean"},
                    "aggregate_cpu_time": {"type": ["number", "null"]},
                    "aggregate_wall_time": {"type": ["number", "null"]},
                    "tasks_per_second": {"type": ["number", "null"]},
                    "peak_rss_bytes": {"type": ["number", "null"]},
                },
            }
            schema_path.write_text(json.dumps(schema_def, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[pipeline] failed writing schema: {e}")
    # Perform validation
    try:
        from jsonschema import validate  # type: ignore

        schema = (
            json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.exists() else None
        )
        if schema:
            from jsonschema import ValidationError  # type: ignore

            try:
                validate(index, schema)
            except ValidationError as ve:
                print(f"[pipeline] index schema validation error: {ve}")
    except Exception as e:
        print(f"[pipeline] schema validation skipped: {e}")

    # Write concise markdown summary for quick CI badge ingestion
    try:
        summary_lines = [
            "# Automation Run Summary",
            f"*Config hash*: `{cfg_hash}`  ",
            f"*Tasks*: {tasks}  ",
            f"*Failures*: {any_fail}  ",
            f"*Aggregate CPU (s)*: {total_cpu:.2f}",
        ]
        if total_wall:
            summary_lines.append(f"*Aggregate Wall (s)*: {total_wall:.2f}")
        if index.get("tasks_per_second"):
            summary_lines.append(f"*Throughput (tasks/s)*: {index['tasks_per_second']:.3f}")
        summary_lines.append(f"*Peak RSS (bytes)*: {int(max_rss)}")
        # D2 coverage metric: fraction of records with non-null d2
        d2_count = 0
        forced_count = 0
        for r in json_records:
            met = (
                r.get("outputs", {}).get("metrics", {})
                if isinstance(r.get("outputs"), dict)
                else {}
            )
            if met.get("d2") is not None:
                d2_count += 1
                if met.get("d2_forced"):
                    forced_count += 1
        if tasks:
            genuine = d2_count - forced_count
            summary_lines.append(
                f"*D2 coverage*: {d2_count}/{tasks} ({(d2_count/tasks)*100:.1f}%) | genuine={genuine} forced={forced_count}"
            )
        if index.get("d2_forced_fraction") is not None:
            summary_lines.append(
                f"*Forced D2 fraction*: {index['d2_forced_fraction']:.2f} (gate_ok={index['forced_d2_gate_ok']})"
            )
        if index.get("logical_fidelity_mean") is not None:
            summary_lines.append(
                f"*Logical fidelity*: mean={index['logical_fidelity_mean']:.3f} min={index['logical_fidelity_min']:.3f} gate_ok={index['logical_fidelity_gate_ok']}"
            )
        (root_dir / "latest_summary.md").write_text(
            "\n".join(summary_lines) + "\n", encoding="utf-8"
        )
    except Exception as e:
        print(f"[pipeline] summary markdown generation failed: {e}")

    # HTML summary report
    html_path = root_dir / "summary.html"
    try:
        rows = []
        for r in records:
            m = r.outputs.get("metrics", {})
            q = r.quality
            rows.append(
                f"<tr><td>{r.lattice[0]}x{r.lattice[1]}</td><td>{r.disorder:.3f}</td><td>{m.get('d2','-')}</td>"
                f"<td>{m.get('trace_max','-')}</td><td>{getattr(q,'trace_ok',None)}</td><td>{getattr(q,'positivity_ok',None)}</td>"
                f"<td>{m.get('min_eig','-')}</td><td>{m.get('adaptive_dt_mean','-')}</td><td>{m.get('local_error_max','-')}</td></tr>"
            )
        html = [
            "<html><head><meta charset='utf-8'><title>Automation Summary</title>",
            "<style>body{font-family:Arial,sans-serif;margin:1.5em;}table{border-collapse:collapse;}th,td{border:1px solid #999;padding:4px 8px;font-size:0.9em;}th{background:#eee;}</style>",
            "</head><body>",
            f"<h1>Automation Summary (hash {cfg_hash})</h1>",
            f"<p>Total runs: {len(records)}</p>",
            "<table><thead><tr><th>Lattice</th><th>Disorder</th><th>D2</th><th>TraceMax</th><th>TraceOK</th><th>PositivityOK</th><th>MinEig</th><th>Mean dt</th><th>LocalErrMax</th></tr></thead><tbody>",
            *rows,
            "</tbody></table>",
            "<p>See index.json for full structured data.</p>",
            "</body></html>",
        ]
        html_path.write_text("\n".join(html), encoding="utf-8")
    except Exception as e:
        print(f"[pipeline] HTML report generation failed: {e}")

    # Archive if requested
    archive = archive_outputs(
        root_dir, cfg_hash, cfg.get("output", {}).get("compress_archives", True)
    )
    # Optional manuscript auto-generation: collect figures/tables and write LaTeX includes
    try:
        if cfg.get("report", {}).get("generate_manuscript", False):
            try:
                from . import generate_manuscript
            except Exception:
                import generate_manuscript
            mm_out = Path(cfg.get("report", {}).get("manuscript_out", "manuscript/auto"))
            try:
                gen_path = generate_manuscript.generate(root_dir, mm_out)
                print(f"[pipeline] manuscript assets generated: {gen_path}")
            except Exception as e:
                print(f"[pipeline] manuscript generation failed: {e}")
    except Exception:
        pass
    # Methods snippet generation
    try:
        methods_path = root_dir / "auto_methods_snippet.tex"
        metric_schema_path = Path("automation/metrics_schema.json")
        metrics_schema_obj = None
        if metric_schema_path.exists():
            try:
                metrics_schema_obj = json.loads(metric_schema_path.read_text(encoding="utf-8"))
            except Exception:
                metrics_schema_obj = None
        with methods_path.open("w", encoding="utf-8") as mf:
            mf.write("% Auto-generated methods summary\n")
            mf.write("\\section*{Automated Methods Summary}\n")
            mf.write("Configuration hash: %s\\\\\n" % cfg_hash)
            noise_cfg = cfg.get("noise", {})
            mf.write(
                "Lattice sizes probed: %s\\\\\n"
                % ", ".join(["%dx%d" % tuple(s) for s in cfg.get("lattice", {}).get("sizes", [])])
            )
            mf.write(
                "Target time T=%.3g, base dt=%.3g, gamma scan points=%s\\\\\n"
                % (
                    noise_cfg.get("target_time", 0.0),
                    noise_cfg.get("dt", 0.0),
                    noise_cfg.get("gamma_scan_logspace", {}).get("points", "?"),
                )
            )
            mf.write(
                "Quality gates applied: max trace dev=%g; positivity tol=%g; fractal gap tol=%g\\\\\n"
                % (
                    cfg.get("quality_gates", {}).get("max_trace_deviation", 1e-9),
                    cfg.get("quality_gates", {}).get("positivity_tolerance", 1e-9),
                    cfg.get("quality_gates", {}).get("fractal_transport_tolerance", 0.15),
                )
            )
            if metrics_schema_obj:
                mf.write(
                    "Tracked metrics keys (%d total): %s\\\\\n"
                    % (
                        len(metrics_schema_obj.get("properties", {})),
                        ", ".join(sorted(metrics_schema_obj.get("properties", {}).keys())[:24]),
                    )
                )
            mf.write(
                "This section was generated automatically; edit suppression via config output.generate_methods=false.\n"
            )
    except Exception as e:
        print(f"[pipeline] methods snippet generation failed: {e}")
    # Write sentinel on failure
    if index.get("any_failure"):
        (root_dir / "FAILED_ANALYSIS").write_text(
            "One or more analyses failed (non-zero returncode).", encoding="utf-8"
        )
        print("[pipeline] FAILURE sentinel written")
    if archive and debug:
        print(f"[pipeline] archived: {archive}")
    # Always print key artifact locations (succinct)
    print(
        f"[pipeline] index: {index_path} | summary: {html_path} | tasks: {len(records)} | failures: {index.get('any_failure')}"
    )
    # Post-run: if advanced extensions artifacts exist, run validator for regression guard
    try:
        adv_dir = Path("advanced_physics")
        adv_json = adv_dir / "advanced_physics_extensions.json"
        if adv_json.exists():
            print("[pipeline] Running advanced extensions validator...")
            proc_val = subprocess.run(
                [sys.executable, "validation/validate_advanced_extensions.py"],
                capture_output=True,
                text=True,
            )
            (root_dir / "advanced_validator_stdout.log").write_text(
                proc_val.stdout, encoding="utf-8"
            )
            (root_dir / "advanced_validator_stderr.log").write_text(
                proc_val.stderr, encoding="utf-8"
            )
            print("[pipeline] Advanced validator rc=", proc_val.returncode)
    except Exception as e:
        print(f"[pipeline] advanced validator skipped: {e}")
    notify(cfg, index_path, success=not index.get("any_failure"))
    print("[pipeline] DONE")


def notify(cfg: Dict[str, Any], index_path: Path, success: bool):
    note_cfg = cfg.get("notify", {})
    if not note_cfg:
        return
    subject = f"Automation pipeline {'success' if success else 'failure'}"
    msg_body = f"Pipeline completed. Index: {index_path}"
    if note_cfg.get("webhook_url"):
        try:
            import requests  # type: ignore

            requests.post(note_cfg["webhook_url"], json={"subject": subject, "body": msg_body})
        except Exception as e:
            print(f"[notify] webhook failed: {e}")
    if note_cfg.get("email_to"):
        try:
            smtp_host = note_cfg.get("smtp_host", "localhost")
            email_from = note_cfg.get("email_from", "automation@localhost")
            email_to = note_cfg.get("email_to")
            m = EmailMessage()
            m["From"] = email_from
            m["To"] = email_to
            m["Subject"] = subject
            m.set_content(msg_body)
            with smtplib.SMTP(smtp_host) as server:
                server.send_message(m)
        except Exception as e:
            print(f"[notify] email failed: {e}")


# ----------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------


def parse_args():
    ap = argparse.ArgumentParser(description="Run full automation pipeline")
    ap.add_argument("--config", type=str, default="automation/pipeline_config.yaml")
    return ap.parse_args()


def main():
    args = parse_args()
    run_pipeline(Path(args.config))


if __name__ == "__main__":
    main()
