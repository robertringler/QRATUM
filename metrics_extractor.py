"""Metrics extraction module.
Parses log files and generated artifacts to compute:
 - Trace deviation (from a stored diagnostic file or stdout markers)
 - Estimated D2 (from a CSV or pattern in stdout)
 - Time step convergence (if multiple dt runs present)
 - Fractal-transport consistency gap
Produces a JSON metrics dict consumed by the pipeline quality gate evaluation.

Integration assumptions:
  - all_analyses.py prints lines like:
        TRACE_MAX=...\n
        D2_ESTIMATE=...\n
        MSD_EXPONENT=...\n
  - Future: create dedicated artifact files (e.g., metrics_run.json)
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

TRACE_RE = re.compile(r"TRACE_MAX=([0-9.eE+-]+)")
TRACE_SWEEP_RE = re.compile(r"TRACE_MAX_SWEEP=([0-9.eE+-]+)")
MIN_EIG_RE = re.compile(r"MIN_EIG=([0-9.eE+-]+)")
NEG_EIG_COUNT_RE = re.compile(r"NEG_EIG_COUNT=([0-9.eE+-]+)")
ADAPT_REJECT_RE = re.compile(r"ADAPTIVE_DT_REJECTIONS=([0-9.eE+-]+)")
ADAPT_MEAN_RE = re.compile(r"ADAPTIVE_DT_MEAN=([0-9.eE+-]+)")
LOCAL_ERR_MAX_RE = re.compile(r"LOCAL_ERROR_MAX=([0-9.eE+-]+)")
LOGICAL_FID_RE = re.compile(r"LOGICAL_FIDELITY=([0-9.eE+-]+)")
ENT_FID_RE = re.compile(r"ENTANGLEMENT_FIDELITY=([0-9.eE+-]+)")
COHERENT_INFO_RE = re.compile(r"COHERENT_INFORMATION=([0-9.eE+-]+)")
D2_RE = re.compile(r"D2_ESTIMATE=([0-9.eE+-]+)")
D2_FORCED_FLAG_RE = re.compile(r"D2_FORCED=([01])")
FISHER_GAMMA_RE = re.compile(r"FISHER_GAMMA=([0-9.eE+-]+)")
FISHER_GAMMA_SQRT_RE = re.compile(r"FISHER_GAMMA_SQRT=([0-9.eE+-]+)")
# PTQ instanton tunneling metrics
DELTA_INST_RE = re.compile(r"DELTA_INSTANTON=([0-9.eE+-]+)")
DELTA_INST_S0_RE = re.compile(r"DELTA_INSTANTON_S0=([0-9.eE+-]+)")
DELTA_INST_VALID_RE = re.compile(r"DELTA_INSTANTON_VALID=([01])")
DELTA_INST_OMEGA0_RE = re.compile(r"DELTA_INSTANTON_OMEGA0=([0-9.eE+-]+)")
# Convergence line pattern example: [convergence] dt=0.02 steps=100 D2=0.678129 OTOC=...
CONV_LINE_RE = re.compile(r"D2=([0-9.]+)")
MSD_EXP_RE = re.compile(r"MSD_EXPONENT=([0-9.eE+-]+)")
DT_ORDER_RE = re.compile(r"DT_ORDER=([0-9.eE+-]+)")
BOOT_HW_RE = re.compile(r"BOOTSTRAP_HALFWIDTH=([0-9.eE+-]+)")
CPU_TIME_RE = re.compile(r"CPU_TIME_SECONDS=([0-9.eE+-]+)")
RSS_RE = re.compile(r"RSS_BYTES=([0-9.eE+-]+)")
LOGICAL_GAMMA_L_HZ_RE = re.compile(r"LOGICAL_GAMMA_L_HZ=([0-9.eE+-]+)")
GATE_TIME_NS_RE = re.compile(r"GATE_TIME_NS=([0-9.eE+-]+)")
MAX_SAFE_CLOCK_MHZ_RE = re.compile(r"MAX_SAFE_CLOCK_MHZ=([0-9.eE+-]+)")

# Ultra-advanced metrics
ULTRA_QFT_RE = re.compile(r"ULTRA_QFT_SCALING_EXPONENT=([0-9.eE+-]+)")
ULTRA_HOLO_VIOL_RE = re.compile(r"ULTRA_MAX_HOLO_VIOLATION=([0-9.eE+-]+)")
ULTRA_MAX_CHERN_RE = re.compile(r"ULTRA_MAX_CHERN_NUMBER=([0-9.eE+-]+)")
ULTRA_FISHER_RE = re.compile(r"ULTRA_FISHER_SCALING=([0-9.eE+-]+)")

# Hyper-advanced metrics
HYPER_GAUGE_EMERGENCE_RE = re.compile(r"HYPER_GAUGE_EMERGENCE_EXPONENT=([0-9.eE+-]+)")
HYPER_DIFFEO_SCALING_RE = re.compile(r"HYPER_DIFFEO_SCALING=([0-9.eE+-]+)")
HYPER_NONCOMM_THETA_RE = re.compile(r"HYPER_NONCOMM_THETA=([0-9.eE+-]+)")
HYPER_SPECTRAL_DIM_RE = re.compile(r"HYPER_SPECTRAL_DIMENSION=([0-9.eE+-]+)")
HYPER_CHAOS_R_RE = re.compile(r"HYPER_CHAOS_R_PARAMETER=([0-9.eE+-]+)")
HYPER_PAGE_ENTROPY_RE = re.compile(r"HYPER_PAGE_ENTROPY=([0-9.eE+-]+)")
HYPER_ENTROPY_PROD_RE = re.compile(r"HYPER_ENTROPY_PRODUCTION_RATE=([0-9.eE+-]+)")

# Autonomous Discovery metrics
HYPER_AUTO_DISCOVERIES_RE = re.compile(r"HYPER_AUTO_DISCOVERIES=([0-9.eE+-]+)")
HYPER_AUTO_ITERATIONS_RE = re.compile(r"HYPER_AUTO_ITERATIONS=([0-9.eE+-]+)")
HYPER_AUTO_OPTIMAL_GAMMA_RE = re.compile(r"HYPER_AUTO_OPTIMAL_GAMMA=([0-9.eE+-]+)")
HYPER_AUTO_OPTIMAL_DISORDER_RE = re.compile(r"HYPER_AUTO_OPTIMAL_DISORDER=([0-9.eE+-]+)")
HYPER_AUTO_EFFICIENCY_RE = re.compile(r"HYPER_AUTO_EFFICIENCY=([0-9.eE+-]+)")
HYPER_AUTO_ANOMALIES_DETECTED_RE = re.compile(r"HYPER_AUTO_ANOMALIES_DETECTED=([0-9.eE+-]+)")
HYPER_AUTO_HYPOTHESES_GENERATED_RE = re.compile(r"HYPER_AUTO_HYPOTHESES_GENERATED=([0-9.eE+-]+)")
HYPER_AUTO_STRATEGY_CHANGES_RE = re.compile(r"HYPER_AUTO_STRATEGY_CHANGES=([0-9.eE+-]+)")
HYPER_AUTO_KNOWLEDGE_SIZE_RE = re.compile(r"HYPER_AUTO_KNOWLEDGE_SIZE=([0-9.eE+-]+)")


@dataclass
class Metrics:
    trace_max: Optional[float] = None
    trace_max_sweep: Optional[float] = None
    d2: Optional[float] = None
    d2_forced: Optional[bool] = None
    alpha_d: Optional[float] = None
    fractal_transport_gap: Optional[float] = None  # |d_s - D2*alpha_D|
    dt_order: Optional[float] = None
    bootstrap_halfwidth: Optional[float] = None
    cpu_time: Optional[float] = None
    rss: Optional[float] = None
    min_eig: Optional[float] = None
    neg_eig_count: Optional[int] = None
    adaptive_dt_rejections: Optional[int] = None
    adaptive_dt_mean: Optional[float] = None
    local_error_max: Optional[float] = None
    logical_fidelity: Optional[float] = None
    entanglement_fidelity: Optional[float] = None
    coherent_information: Optional[float] = None
    fisher_gamma: Optional[float] = None
    fisher_gamma_sqrt: Optional[float] = None
    delta_instanton: Optional[float] = None
    delta_instanton_S0: Optional[float] = None
    delta_instanton_valid: Optional[bool] = None
    delta_instanton_omega0: Optional[float] = None
    # Physical mapping derived tags
    logical_gamma_L_hz: Optional[float] = None
    gate_time_ns: Optional[float] = None
    max_safe_clock_mhz: Optional[float] = None
    # Ultra-advanced
    ultra_qft_scaling_exponent: Optional[float] = None
    ultra_max_holo_violation: Optional[float] = None
    ultra_max_chern_number: Optional[float] = None
    ultra_fisher_scaling: Optional[float] = None
    # Hyper-advanced
    hyper_gauge_emergence_exponent: Optional[float] = None
    hyper_diffeo_scaling: Optional[float] = None
    hyper_noncomm_theta: Optional[float] = None
    hyper_spectral_dimension: Optional[float] = None
    hyper_chaos_r_parameter: Optional[float] = None
    hyper_page_entropy: Optional[float] = None
    hyper_entropy_production_rate: Optional[float] = None
    # Autonomous Discovery
    hyper_auto_discoveries: Optional[int] = None
    hyper_auto_iterations: Optional[int] = None
    hyper_auto_optimal_gamma: Optional[float] = None
    hyper_auto_optimal_disorder: Optional[float] = None
    hyper_auto_efficiency: Optional[float] = None
    hyper_auto_anomalies_detected: Optional[int] = None
    hyper_auto_hypotheses_generated: Optional[int] = None
    hyper_auto_strategy_changes: Optional[int] = None
    hyper_auto_knowledge_size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_stdout(path: Path) -> Metrics:
    m = Metrics()
    if not path.exists():
        return m
    text = path.read_text(encoding="utf-8", errors="ignore")
    if match := TRACE_RE.search(text):
        m.trace_max = float(match.group(1))
    if match := TRACE_SWEEP_RE.search(text):
        m.trace_max_sweep = float(match.group(1))
    if match := D2_RE.search(text):
        m.d2 = float(match.group(1))
    else:
        # Fallback: parse all convergence D2 values and take the last (finest dt) or median
        conv_vals = [float(x) for x in CONV_LINE_RE.findall(text)]
        if conv_vals:
            # Heuristic: choose last (assuming sorted by refining dt). If spread small use median.
            last = conv_vals[-1]
            if len(conv_vals) >= 3 and (max(conv_vals) - min(conv_vals)) < 1e-3:
                last = statistics.median(conv_vals)
            m.d2 = last
    if match := MSD_EXP_RE.search(text):
        m.alpha_d = float(match.group(1))
    if match := DT_ORDER_RE.search(text):
        m.dt_order = float(match.group(1))
    if match := BOOT_HW_RE.search(text):
        m.bootstrap_halfwidth = float(match.group(1))
    if match := CPU_TIME_RE.search(text):
        m.cpu_time = float(match.group(1))
    if match := RSS_RE.search(text):
        m.rss = float(match.group(1))
    if match := LOGICAL_GAMMA_L_HZ_RE.search(text):
        m.logical_gamma_L_hz = float(match.group(1))
    if match := GATE_TIME_NS_RE.search(text):
        m.gate_time_ns = float(match.group(1))
    if match := MAX_SAFE_CLOCK_MHZ_RE.search(text):
        m.max_safe_clock_mhz = float(match.group(1))
    # Ultra-advanced
    if match := ULTRA_QFT_RE.search(text):
        m.ultra_qft_scaling_exponent = float(match.group(1))
    if match := ULTRA_HOLO_VIOL_RE.search(text):
        m.ultra_max_holo_violation = float(match.group(1))
    if match := ULTRA_MAX_CHERN_RE.search(text):
        m.ultra_max_chern_number = float(match.group(1))
    if match := ULTRA_FISHER_RE.search(text):
        m.ultra_fisher_scaling = float(match.group(1))
    # Hyper-advanced
    if match := HYPER_GAUGE_EMERGENCE_RE.search(text):
        m.hyper_gauge_emergence_exponent = float(match.group(1))
    if match := HYPER_DIFFEO_SCALING_RE.search(text):
        m.hyper_diffeo_scaling = float(match.group(1))
    if match := HYPER_NONCOMM_THETA_RE.search(text):
        m.hyper_noncomm_theta = float(match.group(1))
    if match := HYPER_SPECTRAL_DIM_RE.search(text):
        m.hyper_spectral_dimension = float(match.group(1))
    if match := HYPER_CHAOS_R_RE.search(text):
        m.hyper_chaos_r_parameter = float(match.group(1))
    if match := HYPER_PAGE_ENTROPY_RE.search(text):
        m.hyper_page_entropy = float(match.group(1))
    if match := HYPER_ENTROPY_PROD_RE.search(text):
        m.hyper_entropy_production_rate = float(match.group(1))
    # Autonomous Discovery metrics
    if match := HYPER_AUTO_DISCOVERIES_RE.search(text):
        m.hyper_auto_discoveries = int(float(match.group(1)))
    if match := HYPER_AUTO_ITERATIONS_RE.search(text):
        m.hyper_auto_iterations = int(float(match.group(1)))
    if match := HYPER_AUTO_OPTIMAL_GAMMA_RE.search(text):
        m.hyper_auto_optimal_gamma = float(match.group(1))
    if match := HYPER_AUTO_OPTIMAL_DISORDER_RE.search(text):
        m.hyper_auto_optimal_disorder = float(match.group(1))
    if match := HYPER_AUTO_EFFICIENCY_RE.search(text):
        m.hyper_auto_efficiency = float(match.group(1))
    if match := HYPER_AUTO_ANOMALIES_DETECTED_RE.search(text):
        m.hyper_auto_anomalies_detected = int(float(match.group(1)))
    if match := HYPER_AUTO_HYPOTHESES_GENERATED_RE.search(text):
        m.hyper_auto_hypotheses_generated = int(float(match.group(1)))
    if match := HYPER_AUTO_STRATEGY_CHANGES_RE.search(text):
        m.hyper_auto_strategy_changes = int(float(match.group(1)))
    if match := HYPER_AUTO_KNOWLEDGE_SIZE_RE.search(text):
        m.hyper_auto_knowledge_size = int(float(match.group(1)))
    if match := MIN_EIG_RE.search(text):
        m.min_eig = float(match.group(1))
    if match := NEG_EIG_COUNT_RE.search(text):
        m.neg_eig_count = int(float(match.group(1)))
    if match := ADAPT_REJECT_RE.search(text):
        m.adaptive_dt_rejections = int(float(match.group(1)))
    if match := ADAPT_MEAN_RE.search(text):
        m.adaptive_dt_mean = float(match.group(1))
    if match := LOCAL_ERR_MAX_RE.search(text):
        m.local_error_max = float(match.group(1))
    if match := LOGICAL_FID_RE.search(text):
        m.logical_fidelity = float(match.group(1))
    if match := ENT_FID_RE.search(text):
        m.entanglement_fidelity = float(match.group(1))
    if match := COHERENT_INFO_RE.search(text):
        m.coherent_information = float(match.group(1))
    if match := FISHER_GAMMA_RE.search(text):
        m.fisher_gamma = float(match.group(1))
    if match := FISHER_GAMMA_SQRT_RE.search(text):
        m.fisher_gamma_sqrt = float(match.group(1))
    if match := D2_FORCED_FLAG_RE.search(text):
        m.d2_forced = match.group(1) == "1"
    # Provide default false for forced flag if D2 present but no explicit tag
    if m.d2 is not None and m.d2_forced is None:
        m.d2_forced = False
    # PTQ instanton metrics
    if match := DELTA_INST_RE.search(text):
        m.delta_instanton = float(match.group(1))
    if match := DELTA_INST_S0_RE.search(text):
        m.delta_instanton_S0 = float(match.group(1))
    if match := DELTA_INST_VALID_RE.search(text):
        m.delta_instanton_valid = match.group(1) == "1"
    if match := DELTA_INST_OMEGA0_RE.search(text):
        m.delta_instanton_omega0 = float(match.group(1))
    # Fractal transport gap derivation:
    # Let MSD exponent alpha_d ~ 2/d_w (walk dimension d_w) and effective spectral/fractal transport dimension d_s ≈ D2 * alpha_d.
    # Compare d_s_est to embedding dimension (2 for 2D lattice) -> gap = |2 - d_s_est|.
    if m.d2 is not None and m.alpha_d is not None:
        d_s_est = m.d2 * m.alpha_d
        m.fractal_transport_gap = abs(2.0 - d_s_est)
    return m


def write_metrics(metrics: Metrics, out_path: Path) -> None:
    data = metrics.to_dict()
    # Guarantee presence of d2_forced field for downstream tests even if None
    if "d2_forced" not in data or data["d2_forced"] is None:
        # Default False if D2 exists, else None
        if data.get("d2") is not None and data.get("d2_forced") is None:
            data["d2_forced"] = False
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def extract_and_write(analysis_stdout: Path, metrics_path: Path) -> Path:
    metrics = parse_stdout(analysis_stdout)
    write_metrics(metrics, metrics_path)
    return metrics_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    extract_and_write(Path(args.stdout), Path(args.out))
