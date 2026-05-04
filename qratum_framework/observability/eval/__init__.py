"""Layer C — Evaluation Harness.

Prompt × persona × decoding matrix runner with regression detection.
See :mod:`qratum_framework.observability` for the layer overview.
"""

from qratum_framework.observability.eval.metrics import (
    MatrixMetrics,
    aggregate_run,
    matrix_summary,
)
from qratum_framework.observability.eval.personas import Persona, default_personas
from qratum_framework.observability.eval.prompts import PromptSpec, default_prompts
from qratum_framework.observability.eval.regression import (
    RegressionThresholds,
    RegressionVerdict,
    detect_regressions,
)
from qratum_framework.observability.eval.reports import (
    EvalReport,
    write_json_report,
    write_markdown_report,
)
from qratum_framework.observability.eval.runners import (
    EvalRunner,
    MatrixRow,
    ModelCallable,
    SimpleEmissionModel,
)

__all__ = [
    "EvalReport",
    "EvalRunner",
    "MatrixMetrics",
    "MatrixRow",
    "ModelCallable",
    "Persona",
    "PromptSpec",
    "RegressionThresholds",
    "RegressionVerdict",
    "SimpleEmissionModel",
    "aggregate_run",
    "default_personas",
    "default_prompts",
    "detect_regressions",
    "matrix_summary",
    "write_json_report",
    "write_markdown_report",
]
