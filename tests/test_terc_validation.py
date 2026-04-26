"""Tests for TERC Framework Validation Suite."""

import json

import numpy as np
import pytest

from quasim.terc_validation import ValidationRunner
from quasim.terc_validation.validation_runner import ValidationConfig


class TestValidationRunner:
    """Test suite for ValidationRunner."""

    def test_validation_config_defaults(self):
        """Test ValidationConfig default values."""

        config = ValidationConfig()
        assert config.tier is None
        assert config.full_suite is False
        assert config.log_level == "INFO"
        assert config.random_seed == 42

    def test_validation_runner_initialization(self, tmp_path):
        """Test ValidationRunner initialization."""

        config = ValidationConfig(output_dir=tmp_path)
        runner = ValidationRunner(config)
        assert runner.config == config
        assert runner.results == {}

    def test_run_tier_1(self, tmp_path):
        """Test Tier 1 validation."""

        config = ValidationConfig(tier=1, output_dir=tmp_path)
        runner = ValidationRunner(config)
        results = runner.run_tier_1()

        assert results["tier"] == 1
        assert results["name"] == "Computational Foundations"
        assert "experiments" in results
        assert len(results["experiments"]) > 0
        assert results["status"] in ["passed", "failed"]

    def test_run_tier_2(self, tmp_path):
        """Test Tier 2 validation."""

        config = ValidationConfig(tier=2, output_dir=tmp_path)
        runner = ValidationRunner(config)
        results = runner.run_tier_2()

        assert results["tier"] == 2
        assert results["name"] == "Neurobiological Correlation"
        assert "experiments" in results
        assert len(results["experiments"]) > 0

    def test_run_tier_3(self, tmp_path):
        """Test Tier 3 validation."""

        config = ValidationConfig(tier=3, output_dir=tmp_path)
        runner = ValidationRunner(config)
        results = runner.run_tier_3()

        assert results["tier"] == 3
        assert results["name"] == "Clinical Digital Twin"
        assert "experiments" in results

    def test_run_tier_4(self, tmp_path):
        """Test Tier 4 validation."""

        config = ValidationConfig(tier=4, output_dir=tmp_path)
        runner = ValidationRunner(config)
        results = runner.run_tier_4()

        assert results["tier"] == 4
        assert results["name"] == "Meta Validation"
        assert "experiments" in results

    def test_run_full_suite(self, tmp_path):
        """Test full validation suite."""

        config = ValidationConfig(full_suite=True, output_dir=tmp_path)
        runner = ValidationRunner(config)
        results = runner.run_full_suite()

        assert "tiers" in results
        assert len(results["tiers"]) == 4
        assert "summary" in results
        assert "total_tiers" in results["summary"]
        assert "total_experiments" in results["summary"]
        assert "success_rate" in results["summary"]

    def test_experiment_1_1_tda(self, tmp_path):
        """Test Experiment 1.1: TDA baseline."""

        config = ValidationConfig(tier=1, output_dir=tmp_path)
        runner = ValidationRunner(config)
        result = runner._run_experiment_1_1_tda()

        assert result["id"] == "1.1"
        assert result["name"] == "TDA Baseline"
        assert "metrics" in result
        assert "beta_0" in result["metrics"]
        assert "beta_1" in result["metrics"]
        assert "beta_2" in result["metrics"]

    def test_experiment_1_2_quotient(self, tmp_path):
        """Test Experiment 1.2: Quotient calibration."""

        config = ValidationConfig(tier=1, output_dir=tmp_path)
        runner = ValidationRunner(config)
        result = runner._run_experiment_1_2_quotient()

        assert result["id"] == "1.2"
        assert result["name"] == "Quotient Calibration"
        assert "metrics" in result
        assert "calibration_error" in result["metrics"]

    def test_save_results(self, tmp_path):
        """Test results saving functionality."""

        config = ValidationConfig(full_suite=True, output_dir=tmp_path)
        runner = ValidationRunner(config)
        runner.run()

        # Check JSON file exists
        json_file = tmp_path / "validation_results.json"
        assert json_file.exists()

        # Verify JSON content
        with open(json_file) as f:
            saved_results = json.load(f)

        assert "tiers" in saved_results
        assert "summary" in saved_results

    def test_generate_markdown_report(self, tmp_path):
        """Test markdown report generation."""

        config = ValidationConfig(full_suite=True, output_dir=tmp_path)
        runner = ValidationRunner(config)
        runner.run()

        # Check markdown file exists
        md_file = tmp_path.parent / "terc_validation_summary.md"
        assert md_file.exists()

        # Verify markdown content
        with open(md_file) as f:
            content = f.read()

        assert "TERC Framework Validation Summary" in content
        assert "Overall Results" in content
        assert "Tier Results" in content

    def test_deterministic_seeding(self, tmp_path):
        """Test that validation is deterministic with fixed seed."""

        config1 = ValidationConfig(tier=1, output_dir=tmp_path, random_seed=42)
        runner1 = ValidationRunner(config1)
        results1 = runner1.run_tier_1()

        config2 = ValidationConfig(tier=1, output_dir=tmp_path, random_seed=42)
        runner2 = ValidationRunner(config2)
        results2 = runner2.run_tier_1()

        # Results should be identical with same seed
        assert results1["status"] == results2["status"]
        assert len(results1["experiments"]) == len(results2["experiments"])


class TestExperiments:
    """Test individual experiment modules."""

    def test_experiment_1_1_tda_module(self):
        """Test standalone TDA experiment."""

        from quasim.terc_validation.experiment_1_1_tda import run_experiment

        result = run_experiment()
        assert result["status"] == "passed"
        assert "metrics" in result
        assert result["metrics"]["beta_0"] > 0

    def test_experiment_1_2_quotient_module(self):
        """Test standalone quotient experiment."""

        from quasim.terc_validation.experiment_1_2_quotient import run_experiment

        result = run_experiment()
        assert result["status"] == "passed"
        assert "metrics" in result
        assert result["metrics"]["calibration_error"] >= 0

    def test_experiment_2_1_eeg_module(self):
        """Test standalone EEG experiment."""

        from quasim.terc_validation.experiment_2_1_eeg import run_experiment

        result = run_experiment()
        assert result["status"] == "passed"
        assert "metrics" in result
        assert -1 <= result["metrics"]["correlation"] <= 1

    def test_experiment_3_1_pathology_module(self):
        """Test standalone pathology experiment."""

        from quasim.terc_validation.experiment_3_1_pathology import run_experiment

        result = run_experiment()
        assert result["status"] == "passed"
        assert "metrics" in result
        assert 0 <= result["metrics"]["accuracy"] <= 1

    def test_experiment_5_4_integration_module(self):
        """Test standalone integration experiment."""

        from quasim.terc_validation.experiment_5_4_integration import run_experiment

        result = run_experiment()
        assert result["status"] == "passed"
        assert "metrics" in result
        assert 0 <= result["metrics"]["overall_integration_score"] <= 1


class TestConsciousnessMetrics:
    """Test consciousness metric computations."""

    def test_persistent_homology_computation(self):
        """Test persistent homology computation."""

        from quasim.terc_validation.experiment_1_1_tda import compute_persistent_homology

        np.random.seed(42)
        point_cloud = np.random.randn(50, 3)
        result = compute_persistent_homology(point_cloud)

        assert "beta_0" in result
        assert "beta_1" in result
        assert "beta_2" in result
        assert result["beta_0"] > 0

    def test_quotient_calibration(self):
        """Test quotient space calibration."""

        from quasim.terc_validation.experiment_1_2_quotient import calibrate_quotient_space

        np.random.seed(42)
        state_space = np.random.randn(100, 5)
        result = calibrate_quotient_space(state_space, n_quotient_classes=5)

        assert "calibration_error" in result
        assert "n_classes" in result
        assert result["n_classes"] == 5

    def test_eeg_correlation(self):
        """Test EEG correlation computation."""

        from quasim.terc_validation.experiment_2_1_eeg import compute_eeg_correlation

        np.random.seed(42)
        eeg_signal = np.random.randn(100)
        consciousness_metric = eeg_signal + 0.1 * np.random.randn(100)
        result = compute_eeg_correlation(eeg_signal, consciousness_metric)

        assert "correlation" in result
        assert "p_value" in result
        assert -1 <= result["correlation"] <= 1

    def test_pathology_classification(self):
        """Test pathology classification."""

        from quasim.terc_validation.experiment_3_1_pathology import classify_pathology

        metrics = {"phi": 0.85, "icq": 0.90, "beta_1": 5}
        result = classify_pathology(metrics)

        assert "classification" in result
        assert "confidence" in result
        assert result["classification"] in ["healthy", "mild", "severe"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
