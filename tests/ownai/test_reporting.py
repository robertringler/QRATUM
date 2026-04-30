"""Tests for reporting utilities."""

import pytest

from quasim.ownai.eval.benchmark import BenchmarkResult
from quasim.ownai.eval.reporting import (generate_ascii_chart,
                                         generate_markdown_report,
                                         generate_summary_table,
                                         save_results_csv, save_results_json)


@pytest.fixture
def sample_results():
    """Create sample benchmark results."""

    return [
        BenchmarkResult(
            task="tabular-cls",
            model_name="slt",
            dataset="wine",
            seed=42,
            primary_metric=0.85,
            secondary_metric=0.83,
            latency_p50=10.5,
            latency_p95=15.2,
            throughput=95.0,
            model_size_mb=2.5,
            energy_proxy=0.68,
            prediction_hash="abc123" * 10 + "abcd",
        ),
        BenchmarkResult(
            task="tabular-cls",
            model_name="slt",
            dataset="wine",
            seed=43,
            primary_metric=0.84,
            secondary_metric=0.82,
            latency_p50=10.8,
            latency_p95=15.5,
            throughput=93.0,
            model_size_mb=2.5,
            energy_proxy=0.70,
            prediction_hash="abc123" * 10 + "abcd",
        ),
        BenchmarkResult(
            task="tabular-cls",
            model_name="rf",
            dataset="wine",
            seed=42,
            primary_metric=0.82,
            secondary_metric=0.80,
            latency_p50=12.0,
            latency_p95=16.0,
            throughput=83.0,
            model_size_mb=3.0,
            energy_proxy=0.78,
            prediction_hash="def456" * 10 + "defg",
        ),
    ]


def test_save_results_csv(sample_results, tmp_path):
    """Test saving results to CSV."""

    output_path = tmp_path / "results.csv"
    save_results_csv(sample_results, output_path)

    assert output_path.exists()

    # Read back and verify
    with open(output_path) as f:
        lines = f.readlines()
        assert len(lines) == 4  # Header + 3 results
        assert "model" in lines[0]
        assert "slt" in lines[1]


def test_save_results_json(sample_results, tmp_path):
    """Test saving results to JSON."""

    output_path = tmp_path / "results.json"
    save_results_json(sample_results, output_path)

    assert output_path.exists()

    import json

    with open(output_path) as f:
        data = json.load(f)
        assert len(data) == 3
        assert data[0]["model"] == "slt"


def test_generate_summary_table(sample_results):
    """Test generating summary table."""

    summary = generate_summary_table(sample_results)

    assert len(summary) == 2  # 2 unique (task, model) combinations
    assert "tabular-cls_slt" in summary
    assert "tabular-cls_rf" in summary

    slt_summary = summary["tabular-cls_slt"]
    assert slt_summary["n_runs"] == 2
    assert 0.83 <= slt_summary["primary_mean"] <= 0.86
    assert slt_summary["deterministic"] is True


def test_generate_markdown_report(sample_results, tmp_path):
    """Test generating Markdown report."""

    output_path = tmp_path / "report.md"
    generate_markdown_report(sample_results, output_path, title="Test Report")

    assert output_path.exists()

    with open(output_path) as f:
        content = f.read()
        assert "Test Report" in content
        assert "slt" in content
        assert "rf" in content
        assert "Reliability-per-Watt" in content


def test_generate_ascii_chart():
    """Test generating ASCII chart."""

    values = [0.8, 0.6, 0.9, 0.7]
    labels = ["model_a", "model_b", "model_c", "model_d"]

    chart = generate_ascii_chart(values, labels, max_width=30)

    assert "model_a" in chart
    assert "model_c" in chart
    assert "█" in chart
    assert "0.8" in chart
