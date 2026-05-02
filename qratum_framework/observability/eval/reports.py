"""reports.py — JSON + Markdown report writers.

Owns the spec's CI output schema::

    {
      "prompt": "...",
      "model": "nerdy",
      "anomaly_rate": 0.23,
      "cluster_activation_index": 0.78,
      "regression": true
    }

plus a top-level wrapper bundling per-row records, the matrix summary,
and the regression verdict roll-up.  The Markdown writer renders a
small fixed-width table for human review in CI logs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

from qratum_framework.observability.eval.metrics import matrix_summary
from qratum_framework.observability.eval.regression import (
    RegressionThresholds,
    RegressionVerdict,
    detect_regressions,
)

if TYPE_CHECKING:
    from qratum_framework.observability.eval.runners import MatrixRow


@dataclass(frozen=True)
class EvalReport:
    """Top-level report produced by an eval matrix run."""

    rows: Sequence["MatrixRow"]
    verdicts: Sequence[RegressionVerdict]
    summary: Mapping[str, float]
    thresholds: RegressionThresholds

    @property
    def has_regression(self) -> bool:
        return any(v.regression for v in self.verdicts)

    def to_dict(self) -> Dict[str, Any]:
        # Spec-aligned per-row records *plus* the diagnostic drift dump.
        return {
            "schema_version": "1.0",
            "summary": dict(self.summary),
            "thresholds": {
                "max_anomaly_rate": self.thresholds.max_anomaly_rate,
                "max_lift_cluster_rate": self.thresholds.max_lift_cluster_rate,
                "max_cluster_activation_index": (
                    self.thresholds.max_cluster_activation_index
                ),
            },
            "rows": [r.to_dict() for r in self.rows],
            "verdicts": [v.to_dict() for v in self.verdicts],
            "has_regression": bool(self.has_regression),
        }


def build_report(
    rows: Sequence["MatrixRow"],
    *,
    thresholds: Optional[RegressionThresholds] = None,
) -> EvalReport:
    """Produce an :class:`EvalReport` from runner rows."""
    th = thresholds or RegressionThresholds()
    verdicts = detect_regressions(rows, thresholds=th)
    summary = dict(matrix_summary(rows))
    summary["regressions"] = float(sum(1 for v in verdicts if v.regression))
    return EvalReport(
        rows=tuple(rows), verdicts=tuple(verdicts), summary=summary, thresholds=th
    )


def write_json_report(report: EvalReport, path: str | Path) -> Path:
    """Serialise ``report`` to ``path`` as canonical JSON.

    Returns the path written so callers can chain.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return p


def write_markdown_report(report: EvalReport, path: str | Path) -> Path:
    """Render ``report`` as a small Markdown table at ``path``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# QRATUM Observability — Eval Report")
    lines.append("")
    lines.append(
        f"- runs: **{int(report.summary.get('n_runs', 0))}**, "
        f"baseline runs: {int(report.summary.get('n_baseline_runs', 0))}"
    )
    lines.append(
        f"- regressions: **{int(report.summary.get('regressions', 0))}** "
        f"({'FAIL' if report.has_regression else 'PASS'})"
    )
    lines.append(
        f"- max anomaly_rate: {report.summary.get('max_anomaly_rate', 0.0):.3f}, "
        f"max lift_cluster_rate: "
        f"{report.summary.get('max_lift_cluster_rate', 0.0):.3f}"
    )
    lines.append("")
    lines.append("| prompt | persona | anomaly_rate | cluster_activation_index | lift_cluster_rate | regression |")
    lines.append("|---|---|---:|---:|---:|---|")
    for v in report.verdicts:
        lines.append(
            f"| {v.prompt_id} | {v.persona}{' (baseline)' if v.is_baseline else ''} "
            f"| {v.anomaly_rate:.3f} | {v.cluster_activation_index:.3f} "
            f"| {v.lift_cluster_rate:.3f} "
            f"| {'FAIL' if v.regression else 'pass'} |"
        )
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


__all__ = [
    "EvalReport",
    "build_report",
    "write_json_report",
    "write_markdown_report",
]
