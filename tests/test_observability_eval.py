"""Tests for Layer C — Evaluation Harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qratum_framework.observability.eval import (
    EvalRunner,
    Persona,
    SimpleEmissionModel,
    default_personas,
    default_prompts,
    detect_regressions,
    matrix_summary,
    write_json_report,
    write_markdown_report,
)
from qratum_framework.observability.eval.regression import RegressionThresholds
from qratum_framework.observability.eval.reports import build_report
from qratum_framework.observability.eval.runners import BaselineMissingError
from qratum_framework.trace import MerkleLedger

# ---------------------------------------------------------------------------
# Fixtures and emission model
# ---------------------------------------------------------------------------


def test_default_prompts_has_three_spec_prompts():
    pids = [p.id for p in default_prompts()]
    assert pids == ["postgres_pooling", "sorting_algorithm", "quantum_tunneling"]


def test_default_personas_includes_one_baseline_le_03():
    personas = default_personas()
    bs = [p for p in personas if p.is_baseline]
    assert len(bs) == 1
    assert bs[0].temperature <= 0.3


def test_simple_emission_model_deterministic():
    m = SimpleEmissionModel(n_tokens=32)
    p, persona = default_prompts()[0], default_personas()[1]
    a = list(m(p, persona))
    b = list(m(p, persona))
    assert a == b


def test_simple_emission_model_baseline_emits_no_stylistic():
    m = SimpleEmissionModel(n_tokens=32)
    prompt = default_prompts()[0]
    baseline = default_personas()[0]  # neutral, drift_rate=0
    out = list(m(prompt, baseline))
    # Baseline persona has empty stylistic_tokens by construction.
    assert all(tok not in baseline.stylistic_tokens for tok in out)


def test_simple_emission_model_drifted_persona_injects_stylistic():
    m = SimpleEmissionModel(n_tokens=64)
    prompt = default_prompts()[0]
    nerdy = default_personas()[1]
    out = list(m(prompt, nerdy))
    assert any(tok in nerdy.stylistic_tokens for tok in out)


# ---------------------------------------------------------------------------
# Runner — baseline-pairing invariant
# ---------------------------------------------------------------------------


def test_runner_requires_baseline_persona():
    only_personas = (Persona(name="nerdy", temperature=0.7),)
    with pytest.raises(BaselineMissingError):
        EvalRunner(personas=only_personas)


def test_runner_rejects_high_temperature_baseline():
    bad = (
        Persona(name="loose", temperature=0.9, is_baseline=True),
        Persona(name="nerdy", temperature=0.7),
    )
    with pytest.raises(BaselineMissingError):
        EvalRunner(personas=bad)


# ---------------------------------------------------------------------------
# End-to-end matrix run
# ---------------------------------------------------------------------------


def test_runner_emits_one_row_per_prompt_persona():
    runner = EvalRunner()
    rows = runner.run()
    expected = len(default_prompts()) * len(default_personas())
    assert len(rows) == expected


def test_runner_baseline_rows_are_clean():
    runner = EvalRunner()
    rows = runner.run()
    baseline_rows = [r for r in rows if r.is_baseline]
    assert baseline_rows
    for r in baseline_rows:
        assert r.drift.anomaly_detected is False
        assert r.metrics.anomaly_rate == 0.0


def test_runner_planted_drift_signal_is_detectable():
    """The deterministic SimpleEmissionModel injects fantasy tokens
    under the nerdy persona; some non-baseline row should have a
    non-zero anomaly_rate."""
    runner = EvalRunner()
    rows = runner.run()
    non_baseline = [r for r in rows if not r.is_baseline]
    assert any(r.metrics.anomaly_rate > 0.0 for r in non_baseline)


def test_runner_deterministic_across_runs():
    rows1 = EvalRunner().run()
    rows2 = EvalRunner().run()
    assert [r.to_dict() for r in rows1] == [r.to_dict() for r in rows2]


def test_runner_with_ledger_appends_one_entry_per_row():
    ledger = MerkleLedger()
    runner = EvalRunner(ledger=ledger)
    rows = runner.run()
    assert ledger.height() == len(rows)
    assert ledger.verify() is True


# ---------------------------------------------------------------------------
# Aggregation + regression detection
# ---------------------------------------------------------------------------


def test_aggregate_run_anomaly_rate_bounded():
    runner = EvalRunner()
    rows = runner.run()
    for r in rows:
        assert 0.0 <= r.metrics.anomaly_rate <= 1.0
        assert 0.0 <= r.metrics.cluster_activation_index <= 1.0


def test_detect_regressions_skips_baseline_by_default():
    runner = EvalRunner()
    rows = runner.run()
    verdicts = detect_regressions(rows)
    for v in verdicts:
        if v.is_baseline:
            assert v.regression is False


def test_detect_regressions_planted_signal_trips_strict_thresholds():
    """With aggressive thresholds, the planted nerdy-persona drift trips."""
    runner = EvalRunner()
    rows = runner.run()
    th = RegressionThresholds(
        max_anomaly_rate=0.0,
        max_lift_cluster_rate=0.0,
        max_cluster_activation_index=0.0,
    )
    verdicts = detect_regressions(rows, thresholds=th)
    assert any(v.regression for v in verdicts)


def test_matrix_summary_aggregates_safely_on_empty():
    summary = matrix_summary([])
    assert summary["n_runs"] == 0


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def test_eval_report_to_dict_has_spec_keys(tmp_path: Path):
    runner = EvalRunner()
    rows = runner.run()
    report = build_report(rows)
    payload = report.to_dict()
    # Each per-row record must contain the spec's CI output keys.
    for r in payload["rows"]:
        for key in ("prompt", "model", "anomaly_rate", "cluster_activation_index"):
            assert key in r


def test_write_json_and_markdown_report_round_trip(tmp_path: Path):
    runner = EvalRunner()
    report = build_report(runner.run())
    json_path = write_json_report(report, tmp_path / "report.json")
    md_path = write_markdown_report(report, tmp_path / "report.md")
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text())
    assert "rows" in payload and "verdicts" in payload
    md = md_path.read_text()
    assert "# QRATUM" in md


def test_eval_report_has_regression_flag_consistent():
    runner = EvalRunner()
    rows = runner.run()
    report = build_report(rows)
    expected = any(v.regression for v in report.verdicts)
    assert report.has_regression == expected


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_eval_subcommand_smoke(tmp_path: Path, capsys):
    from qratum_framework.cli import main

    rc = main(
        [
            "eval",
            "--report-json",
            str(tmp_path / "report.json"),
            "--report-md",
            str(tmp_path / "report.md"),
        ]
    )
    assert rc == 0
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()


def test_cli_clusters_subcommand_smoke(tmp_path: Path):
    from qratum_framework.cli import main

    p = tmp_path / "p.txt"
    b = tmp_path / "b.txt"
    p.write_text(" ".join(["goblin", "troll", "filler"] * 8))
    b.write_text(" ".join(["the", "and", "of"] * 8))
    rc = main(
        [
            "clusters",
            "--persona-tokens",
            str(p),
            "--baseline-tokens",
            str(b),
            "--window",
            "4",
        ]
    )
    assert rc == 0
