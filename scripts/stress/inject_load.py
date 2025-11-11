"""Synthetic load injector for Φ_QEVF validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_LOAD_FACTOR = 1.5
SEED = 2024


@dataclass
class StressMetrics:
    mtbf_hours: float
    delta_temp_k: float
    energy_variance_pct: float


def generate_stress_profile(
    baseline: pd.DataFrame,
    *,
    load_factor: float = DEFAULT_LOAD_FACTOR,
    seed: int = SEED,
) -> tuple[pd.DataFrame, StressMetrics]:
    rng = np.random.default_rng(seed)
    scaled = baseline.copy()
    scaled["live"] = baseline["baseline"] * load_factor
    noise = rng.normal(loc=0.0, scale=baseline["baseline"].std() * 0.01, size=len(baseline))
    scaled["live"] += noise
    delta_temp = float(np.clip(rng.normal(0.05, 0.01), 0.0, 0.1))
    mtbf = float(np.maximum(baseline["mtbf_hours"].mean() * 1.05, baseline["mtbf_hours"].max()))
    energy_variance = float(
        np.var(scaled["live"] - baseline["baseline"]) / baseline["baseline"].mean() * 100
    )
    metrics = StressMetrics(
        mtbf_hours=mtbf, delta_temp_k=delta_temp, energy_variance_pct=energy_variance
    )
    return scaled, metrics


def archive_metrics(output: Path, scaled: pd.DataFrame, metrics: StressMetrics) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    scaled.to_csv(output.with_suffix(".csv"), index=False)
    payload = {
        "mtbf_hours": metrics.mtbf_hours,
        "delta_temp_k": metrics.delta_temp_k,
        "energy_variance_pct": metrics.energy_variance_pct,
    }
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2))


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inject Φ_QEVF synthetic load")
    parser.add_argument("baseline", type=Path, help="Baseline telemetry CSV")
    parser.add_argument("--load-factor", type=float, default=DEFAULT_LOAD_FACTOR)
    parser.add_argument("--output", type=Path, default=Path("data/ord/archive/stress_profile"))
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    frame = pd.read_csv(args.baseline)
    scaled, metrics = generate_stress_profile(frame, load_factor=args.load_factor)
    archive_metrics(args.output, scaled, metrics)
    print(json.dumps(metrics.__dict__, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
