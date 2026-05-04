import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from services.telemetry.verifier import TelemetryVerifier, VerificationConfig


def build_frame(values):
    return pd.DataFrame(values)


def test_verifier_detects_within_tolerance(tmp_path):
    baseline = build_frame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=6, freq="h"),
            "metric": ["ops_per_kw" for _ in range(6)],
            "baseline": [100.0, 101.0, 99.5, 100.5, 100.2, 99.8],
            "live": [100.2, 100.8, 99.7, 100.6, 100.1, 99.9],
        }
    )
    live = baseline.copy()
    live["live"] += [0.2, -0.1, 0.3, -0.2, 0.0, 0.1]

    verifier = TelemetryVerifier(VerificationConfig(tolerance_pct=5.0, archive_dir=tmp_path))
    result = verifier.evaluate(live, baseline)

    assert result.passed is True
    assert result.rmse < 1.0
    assert result.threshold_breaches <= 3

    summary = json.loads(Path(result.summary_path).read_text())
    assert summary["record_count"] == 6
    manifest_lines = Path(result.manifest_path).read_text().strip().splitlines()
    assert len(manifest_lines) == 2
    for line in manifest_lines:
        digest, filename = line.split()
        file_path = tmp_path / filename
        assert file_path.exists()
        assert len(digest) == 64


def test_verifier_flags_excessive_variance(tmp_path):
    baseline = build_frame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="h"),
            "metric": ["variance" for _ in range(5)],
            "baseline": [90, 90, 90, 90, 90],
            "live": [90, 90, 90, 90, 90],
        }
    )
    live = baseline.copy()
    live["live"] = [110, 112, 109, 111, 110]

    verifier = TelemetryVerifier(VerificationConfig(tolerance_pct=5.0, archive_dir=tmp_path))
    result = verifier.evaluate(live, baseline)

    assert result.passed is False
    assert result.threshold_breaches > 3
