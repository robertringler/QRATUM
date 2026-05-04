"""Closed-loop Φ_QEVF telemetry verification utilities.

This module compares live telemetry feeds against deterministic baseline
simulation outputs and produces audit-ready archives for ORD (Operational
Reliability Drift) campaigns.  It also feeds verification summaries into the
Quantum Market Protocol (QMP) sandbox so economic projections stay aligned
with the latest physics-backed evidence.

The verifier computes RMSE, relative variance, and threshold breach counts for
any number of telemetry dimensions provided as pandas DataFrames.  Results are
persisted to ``data/ord/archive`` along with SHA256 manifests for downstream
compliance automation.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping

import numpy as np
import pandas as pd


def _load_qmp_pricing_model():
    """Dynamically load the QuantumPricingModel from the QMP sandbox.

    The sandbox lives in ``qmp/qex-sandbox`` which is not a valid Python
    package name, so we resolve the module manually to keep imports explicit
    and deterministic.
    """

    qmp_root = Path(__file__).resolve().parents[2] / "qmp" / "qex-sandbox" / "pricing" / "model.py"
    spec = importlib.util.spec_from_file_location("qmp.qex_sandbox.pricing.model", qmp_root)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, "Unable to resolve QMP pricing model"
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)  # type: ignore[call-arg]
    return module.QuantumPricingModel


QuantumPricingModel = _load_qmp_pricing_model()


@dataclass(slots=True)
class VerificationConfig:
    """Configuration for :class:`TelemetryVerifier`.

    Attributes
    ----------
    tolerance_pct:
        Maximum acceptable variance percentage before triggering CI failure
        gates.  The default is 5% which matches Phase VI.1 tolerances.
    rolling_window_hours:
        Size of the rolling evaluation window.  The verifier still works if the
        data does not cover the full window; the value is persisted as
        metadata for compliance reports.
    archive_dir:
        Filesystem directory where ORD archives are written.
    provenance:
        Additional metadata injected into summary reports.  Commit SHAs and
        pipeline identifiers can be supplied here.
    """

    tolerance_pct: float = 5.0
    rolling_window_hours: int = 6
    archive_dir: Path = Path("data/ord/archive")
    provenance: MutableMapping[str, str] = dataclasses.field(default_factory=dict)


@dataclass(slots=True)
class VerificationResult:
    """Computed statistics for a telemetry verification run."""

    rmse: float
    variance_pct: float
    threshold_breaches: int
    passed: bool
    summary_path: Path
    manifest_path: Path
    archive_csv: Path


class TelemetryVerifier:
    """Compare live telemetry frames with deterministic baseline references."""

    def __init__(self, config: VerificationConfig | None = None) -> None:
        self.config = config or VerificationConfig()
        self.config.archive_dir.mkdir(parents=True, exist_ok=True)
        self.pricing_model = QuantumPricingModel()

    @staticmethod
    def _to_frame(dataset: Iterable[Mapping[str, float]] | pd.DataFrame) -> pd.DataFrame:
        frame = dataset.copy() if isinstance(dataset, pd.DataFrame) else pd.DataFrame(list(dataset))
        required = {"timestamp", "metric", "baseline", "live"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Telemetry frame missing columns: {sorted(missing)}")
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        return frame

    def evaluate(
        self,
        live_dataset: Iterable[Mapping[str, float]] | pd.DataFrame,
        baseline_dataset: Iterable[Mapping[str, float]] | pd.DataFrame,
        metadata: dict[str, str] | None = None,
    ) -> VerificationResult:
        """Compute verification statistics and persist archival artefacts.

        Parameters
        ----------
        live_dataset, baseline_dataset:
            Iterables or DataFrames containing ``timestamp``, ``metric``,
            ``live`` and ``baseline`` columns.  ``baseline_dataset`` values are
            used solely for provenance in the merged archive.
        metadata:
            Optional metadata merged with provenance when emitting the summary
            JSON.
        """

        live_frame = self._to_frame(live_dataset)
        baseline_frame = self._to_frame(baseline_dataset)
        merged = pd.merge(
            baseline_frame,
            live_frame[["timestamp", "metric", "live"]],
            on=["timestamp", "metric"],
            suffixes=("_baseline", ""),
        )
        merged["delta"] = merged["live"] - merged["baseline"]
        rmse = float(np.sqrt(np.mean(np.square(merged["delta"].to_numpy()))))
        baseline_reference = np.maximum(np.abs(merged["baseline"].to_numpy()), 1e-9)
        variance_pct = float(
            np.var(merged["delta"].to_numpy()) / np.mean(baseline_reference) * 100.0
        )
        relative_delta = np.abs(merged["delta"].to_numpy()) / baseline_reference
        threshold_breaches = int(np.sum(relative_delta * 100.0 > self.config.tolerance_pct))
        breach_cap = max(3, int(0.2 * len(merged)))
        fail_due_to_variance = variance_pct > self.config.tolerance_pct and threshold_breaches > 3
        fail_due_to_frequency = threshold_breaches > breach_cap
        passed = not (fail_due_to_variance or fail_due_to_frequency)

        summary_path, csv_path, manifest_path = self._emit_archives(
            merged,
            rmse=rmse,
            variance_pct=variance_pct,
            threshold_breaches=threshold_breaches,
            metadata=metadata or {},
            passed=passed,
        )

        # Feed the verification outcome into the Quantum Market Protocol sandbox
        self.pricing_model.update_from_verification(
            rmse=rmse,
            variance_pct=variance_pct,
            passed=passed,
            breach_count=threshold_breaches,
            telemetry=merged,
        )

        return VerificationResult(
            rmse=rmse,
            variance_pct=variance_pct,
            threshold_breaches=threshold_breaches,
            passed=passed,
            summary_path=summary_path,
            manifest_path=manifest_path,
            archive_csv=csv_path,
        )

    def _emit_archives(
        self,
        merged: pd.DataFrame,
        *,
        rmse: float,
        variance_pct: float,
        threshold_breaches: int,
        metadata: Mapping[str, str],
        passed: bool,
    ) -> tuple[Path, Path, Path]:
        timestamp = _dt.datetime.utcnow().replace(microsecond=0).isoformat().replace(":", "") + "Z"
        archive_csv = self.config.archive_dir / f"ord_metrics_{timestamp}.csv"
        summary_json = self.config.archive_dir / f"ord_summary_{timestamp}.json"
        manifest_path = self.config.archive_dir / f"ord_manifest_{timestamp}.sha256"

        enriched = merged.copy()
        enriched["variance_pct"] = variance_pct
        enriched["rmse"] = rmse
        enriched["threshold_breaches"] = threshold_breaches
        enriched["passed"] = passed
        enriched.to_csv(archive_csv, index=False)

        provenance = dict(self.config.provenance)
        provenance.setdefault("rolling_window_hours", str(self.config.rolling_window_hours))
        provenance.setdefault("tolerance_pct", f"{self.config.tolerance_pct:.2f}")
        provenance.setdefault("commit", os.getenv("GITHUB_SHA", "UNKNOWN"))
        summary_payload: dict[str, object] = {
            "timestamp": timestamp,
            "rmse": rmse,
            "variance_pct": variance_pct,
            "threshold_breaches": threshold_breaches,
            "passed": passed,
            "metadata": {**provenance, **metadata},
            "record_count": int(len(enriched)),
        }
        summary_json.write_text(json.dumps(summary_payload, indent=2))

        hashes = {
            archive_csv.name: self._sha256_file(archive_csv),
            summary_json.name: self._sha256_file(summary_json),
        }
        manifest_lines = [f"{digest}  {filename}" for filename, digest in hashes.items()]
        manifest_path.write_text("\n".join(manifest_lines) + "\n")
        return summary_json, archive_csv, manifest_path

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()


__all__ = [
    "VerificationConfig",
    "VerificationResult",
    "TelemetryVerifier",
]
