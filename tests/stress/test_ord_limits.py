import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.stress.inject_load import DEFAULT_LOAD_FACTOR, generate_stress_profile


def build_baseline():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="h"),
            "baseline": [150.0] * 10,
            "mtbf_hours": [120.0] * 10,
        }
    )


def test_stress_profile_respects_temperature_and_mtbf():
    baseline = build_baseline()
    scaled, metrics = generate_stress_profile(baseline, load_factor=DEFAULT_LOAD_FACTOR)

    assert metrics.delta_temp_k < 0.1
    assert metrics.mtbf_hours >= baseline["mtbf_hours"].mean()
    assert len(scaled) == len(baseline)
