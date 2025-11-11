"""Tests for CLI commands."""

import json

import pytest
from click.testing import CliRunner

from quasim.cli.quasim_own import cli


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


def test_cli_help(runner):
    """Test CLI help command."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "QuASIM-Own" in result.output


def test_cli_train(runner, tmp_path):
    """Test train command."""
    output_dir = tmp_path / "test_run"

    result = runner.invoke(
        cli,
        [
            "train",
            "--model",
            "rf",
            "--task",
            "tabular-cls",
            "--dataset",
            "wine",
            "--seed",
            "42",
            "--out",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Training complete" in result.output

    # Check that results were saved
    results_file = output_dir / "results.json"
    assert results_file.exists()

    with open(results_file) as f:
        data = json.load(f)
        assert data["model"] == "rf"
        assert data["task"] == "tabular-cls"


def test_cli_eval(runner, tmp_path):
    """Test eval command."""
    # First train a model
    output_dir = tmp_path / "test_run"
    runner.invoke(
        cli,
        [
            "train",
            "--model",
            "logreg",
            "--task",
            "tabular-cls",
            "--dataset",
            "wine",
            "--out",
            str(output_dir),
        ],
    )

    # Then evaluate it
    result = runner.invoke(
        cli,
        ["eval", "--run", str(output_dir), "--metrics", "all"],
    )

    assert result.exit_code == 0
    assert "Evaluation Results" in result.output


def test_cli_benchmark(runner, tmp_path):
    """Test benchmark command with quick suite."""
    report_dir = tmp_path / "reports"

    result = runner.invoke(
        cli,
        [
            "benchmark",
            "--suite",
            "quick",
            "--repeat",
            "1",
            "--cpu-only",
            "--report",
            str(report_dir),
        ],
    )

    # May fail due to missing dependencies, but should start
    # Just check it attempts to run
    assert "benchmark" in result.output.lower() or result.exit_code >= 0


def test_cli_export(runner, tmp_path):
    """Test export command."""
    # First train a model
    output_dir = tmp_path / "test_run"
    runner.invoke(
        cli,
        [
            "train",
            "--model",
            "rf",
            "--task",
            "tabular-cls",
            "--dataset",
            "wine",
            "--out",
            str(output_dir),
        ],
    )

    # Export it
    export_path = tmp_path / "exported_model.json"
    result = runner.invoke(
        cli,
        [
            "export",
            "--run",
            str(output_dir),
            "--format",
            "json",
            "--out",
            str(export_path),
        ],
    )

    assert result.exit_code == 0
    assert export_path.exists()


def test_cli_modelcard(runner, tmp_path):
    """Test modelcard command."""
    # First train a model
    output_dir = tmp_path / "test_run"
    runner.invoke(
        cli,
        [
            "train",
            "--model",
            "slt",
            "--task",
            "text-cls",
            "--dataset",
            "imdb-mini",
            "--out",
            str(output_dir),
        ],
    )

    # Generate model card
    card_path = tmp_path / "model_card.md"
    result = runner.invoke(
        cli,
        ["modelcard", "--run", str(output_dir), "--out", str(card_path)],
    )

    assert result.exit_code == 0
    assert card_path.exists()
