from __future__ import annotations

from qstack.alignment.constitution import DEFAULT_CONSTITUTION
from qstack.alignment.evaluator import AlignmentEvaluator
from qstack.config import QStackConfig


def test_alignment_precheck_passes_with_defaults():
    config = QStackConfig()
    evaluator = AlignmentEvaluator(config, DEFAULT_CONSTITUTION)

    violations = evaluator.pre_operation_check("qnx.simulation", {})

    assert not evaluator.has_fatal(violations)


def test_alignment_precheck_catches_invalid_timesteps():
    config = QStackConfig()
    config.qnx.timesteps = -1
    evaluator = AlignmentEvaluator(config, DEFAULT_CONSTITUTION)

    violations = evaluator.pre_operation_check("qnx.simulation", {})

    assert evaluator.has_fatal(violations)
    assert any(v.article_id == "ARTICLE_QNX_SAFETY" for v in violations)
