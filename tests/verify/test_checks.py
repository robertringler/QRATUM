"""Tests for verification tool."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from quasim_verify.checks import (
    audit_chain,
    comp_artifacts,
    comp_cmmc_map,
    doc_language_lint,
    econ_montecarlo,
    econ_phi_qevf,
    tech_benchmarks,
    tech_compression,
    tech_rl_convergence,
    tech_telemetry_rmse,
)
from quasim_verify.io import load_json, load_yaml, sha256_file, write_json
from quasim_verify.models import CheckResult


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""

    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def base_config(temp_dir):
    """Create base configuration for tests."""

    return {
        "inputs": {
            "brief_paths": [],
            "telemetry": {},
            "artifacts": {},
            "compliance": {},
            "economics": {},
        },
        "checks": [],
        "policy": {"tolerances": {}, "ban_phrases_unless_evidence": []},
        "outputs": {
            "report_json": str(temp_dir / "report.json"),
            "report_sarif": str(temp_dir / "report.sarif"),
            "audit_chain_file": str(temp_dir / "audit.sha256chain"),
        },
    }


class TestTelemetryRMSE:
    """Tests for TECH-002 telemetry RMSE check."""

    def test_rmse_calculation_passes(self, temp_dir, base_config):
        """Test RMSE check passes with good data."""

        # Create test CSV
        df = pd.DataFrame(
            {
                "time_s": [0, 1, 2, 3, 4],
                "altitude_km": [100.0, 101.0, 102.0, 103.0, 104.0],
                "altitude_km_ref": [100.0, 101.0, 102.0, 103.0, 104.0],
            }
        )
        spacex_csv = temp_dir / "f9.csv"
        nasa_csv = temp_dir / "orion.csv"
        df.to_csv(spacex_csv, index=False)
        df.to_csv(nasa_csv, index=False)

        config = base_config.copy()
        config["inputs"]["telemetry"] = {
            "spacex_csv": str(spacex_csv),
            "nasa_csv": str(nasa_csv),
        }
        config["policy"]["tolerances"]["rmse_max"] = 0.02

        result = tech_telemetry_rmse.run(config)

        assert isinstance(result, CheckResult)
        assert result.id == "TECH-002"
        assert result.passed is True
        assert "spacex_f9" in result.details
        assert result.details["spacex_f9"] < 0.02

    def test_rmse_calculation_fails(self, temp_dir, base_config):
        """Test RMSE check fails with poor data."""

        df = pd.DataFrame(
            {
                "time_s": [0, 1, 2, 3, 4],
                "altitude_km": [100.0, 101.0, 102.0, 103.0, 104.0],
                "altitude_km_ref": [110.0, 111.0, 112.0, 113.0, 114.0],
            }
        )
        spacex_csv = temp_dir / "f9.csv"
        nasa_csv = temp_dir / "orion.csv"
        df.to_csv(spacex_csv, index=False)
        df.to_csv(nasa_csv, index=False)

        config = base_config.copy()
        config["inputs"]["telemetry"] = {
            "spacex_csv": str(spacex_csv),
            "nasa_csv": str(nasa_csv),
        }
        config["policy"]["tolerances"]["rmse_max"] = 0.02

        result = tech_telemetry_rmse.run(config)

        assert result.passed is False


class TestCompression:
    """Tests for TECH-004 compression check."""

    def test_compression_passes(self, temp_dir, base_config):
        """Test compression check passes."""

        npz_file = temp_dir / "compression.npz"
        np.savez(npz_file, raw_flops=1e12, compressed_flops=5e10)

        config = base_config.copy()
        config["inputs"]["artifacts"]["compression_npz"] = str(npz_file)
        config["policy"]["tolerances"]["compression_min_ratio"] = 18.0

        result = tech_compression.run(config)

        assert result.id == "TECH-004"
        assert result.passed is True
        assert result.details["ratio"] == 20.0

    def test_compression_fails(self, temp_dir, base_config):
        """Test compression check fails with low ratio."""

        npz_file = temp_dir / "compression.npz"
        np.savez(npz_file, raw_flops=1e12, compressed_flops=1e11)

        config = base_config.copy()
        config["inputs"]["artifacts"]["compression_npz"] = str(npz_file)
        config["policy"]["tolerances"]["compression_min_ratio"] = 18.0

        result = tech_compression.run(config)

        assert result.passed is False
        assert result.details["ratio"] == 10.0


class TestRLConvergence:
    """Tests for TECH-003 RL convergence check."""

    def test_rl_convergence_passes(self, temp_dir, base_config):
        """Test RL convergence passes."""

        json_file = temp_dir / "rl.json"
        with open(json_file, "w") as f:
            json.dump({"final_convergence": 0.995}, f)

        config = base_config.copy()
        config["inputs"]["artifacts"]["rl_convergence_json"] = str(json_file)
        config["policy"]["tolerances"]["rl_convergence_min"] = 0.993

        result = tech_rl_convergence.run(config)

        assert result.id == "TECH-003"
        assert result.passed is True
        assert result.details["convergence"] == 0.995

    def test_rl_convergence_fails(self, temp_dir, base_config):
        """Test RL convergence fails."""

        json_file = temp_dir / "rl.json"
        with open(json_file, "w") as f:
            json.dump({"final_convergence": 0.990}, f)

        config = base_config.copy()
        config["inputs"]["artifacts"]["rl_convergence_json"] = str(json_file)
        config["policy"]["tolerances"]["rl_convergence_min"] = 0.993

        result = tech_rl_convergence.run(config)

        assert result.passed is False


class TestPhiQEVF:
    """Tests for ECON-201 Φ_QEVF check."""

    def test_phi_qevf_passes(self, temp_dir, base_config):
        """Test Φ_QEVF calculation passes."""

        yaml_file = temp_dir / "phi.yaml"
        data = {
            "base_value_per_eph": 0.0004,
            "eta_ent": 0.93,
            "eta_baseline": 1.0,
            "coherence_penalty": 0.95,
            "runtime_factor": 1.0,
            "market_multiplier": 1.0,
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        config = base_config.copy()
        config["inputs"]["economics"]["phi_inputs_yaml"] = str(yaml_file)

        result = econ_phi_qevf.run(config)

        assert result.id == "ECON-201"
        assert result.passed is True
        assert "phi_qevf" in result.details


class TestMonteCarloValuation:
    """Tests for ECON-202 Monte Carlo check."""

    def test_monte_carlo_passes(self, temp_dir, base_config):
        """Test Monte Carlo valuation passes."""

        yaml_file = temp_dir / "mc.yaml"
        data = {
            "trials": 1000,
            "seed": 42,
            "scenarios": [
                {"prob": 0.3, "value": 6.0e8},
                {"prob": 0.5, "value": 7.5e8},
                {"prob": 0.2, "value": 9.0e8},
            ],
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        config = base_config.copy()
        config["inputs"]["economics"]["montecarlo_params_yaml"] = str(yaml_file)
        config["policy"]["tolerances"]["valuation_p50_min"] = 5.0e8
        config["policy"]["tolerances"]["valuation_p50_max"] = 1.5e9

        result = econ_montecarlo.run(config)

        assert result.id == "ECON-202"
        assert result.passed is True
        assert "p50" in result.details


class TestComplianceArtifacts:
    """Tests for COMP-101 compliance artifacts check."""

    def test_compliance_artifacts_pass(self, temp_dir, base_config):
        """Test compliance artifacts check passes."""

        # Create dummy files
        psac = temp_dir / "PSAC.pdf"
        sas = temp_dir / "SAS.pdf"
        der = temp_dir / "DER.pdf"
        psac.write_text("dummy")
        sas.write_text("dummy")
        der.write_text("dummy")

        config = base_config.copy()
        config["inputs"]["compliance"] = {
            "psac_id": str(psac),
            "sas_id": str(sas),
            "der_letter": str(der),
        }
        config["policy"]["require_der_for_level_a"] = True

        result = comp_artifacts.run(config)

        assert result.id == "COMP-101"
        assert result.passed is True
        assert len(result.details["missing"]) == 0

    def test_compliance_artifacts_fail(self, temp_dir, base_config):
        """Test compliance artifacts check fails."""

        config = base_config.copy()
        config["inputs"]["compliance"] = {
            "psac_id": "/nonexistent/PSAC.pdf",
            "sas_id": "/nonexistent/SAS.pdf",
            "der_letter": "/nonexistent/DER.pdf",
        }
        config["policy"]["require_der_for_level_a"] = True

        result = comp_artifacts.run(config)

        assert result.passed is False
        assert len(result.details["missing"]) > 0


class TestCMMCMap:
    """Tests for COMP-102 CMMC mapping check."""

    def test_cmmc_map_passes(self, temp_dir, base_config):
        """Test CMMC map check passes."""

        yaml_file = temp_dir / "cmmc.yaml"
        data = {
            "practices": [
                {"id": f"AC.L2-3.1.{i}", "description": f"Practice {i}"} for i in range(115)
            ]
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        config = base_config.copy()
        config["inputs"]["compliance"]["cmmc_map"] = str(yaml_file)
        config["policy"]["tolerances"]["cmmc_practices_min"] = 110

        result = comp_cmmc_map.run(config)

        assert result.id == "COMP-102"
        assert result.passed is True
        assert result.details["count"] == 115


class TestLanguageLint:
    """Tests for GOV-301 language lint check."""

    def test_language_lint_no_hits(self, temp_dir, base_config):
        """Test language lint with no banned phrases."""

        md_file = temp_dir / "doc.md"
        md_file.write_text("This is a clean document with no banned phrases.")

        config = base_config.copy()
        config["inputs"]["brief_paths"] = [str(md_file)]
        config["policy"]["ban_phrases_unless_evidence"] = ["53B", "forbidden phrase"]

        result = doc_language_lint.run(config)

        assert result.id == "GOV-301"
        assert result.passed is True
        assert len(result.details["banned_hits"]) == 0

    def test_language_lint_with_hits(self, temp_dir, base_config):
        """Test language lint finds banned phrases."""

        md_file = temp_dir / "doc.md"
        md_file.write_text("This document claims 53B in value.")

        config = base_config.copy()
        config["inputs"]["brief_paths"] = [str(md_file)]
        config["policy"]["ban_phrases_unless_evidence"] = ["53B"]

        result = doc_language_lint.run(config)

        assert result.passed is False
        assert "53B" in result.details["banned_hits"]


class TestAuditChain:
    """Tests for DOC-401 audit chain check."""

    def test_audit_chain_creation(self, temp_dir, base_config):
        """Test audit chain generation."""

        # Create test files
        file1 = temp_dir / "file1.txt"
        file2 = temp_dir / "file2.txt"
        file1.write_text("content1")
        file2.write_text("content2")

        config = base_config.copy()
        config["inputs"]["brief_paths"] = [str(file1), str(file2)]
        config["inputs"]["economics"] = {}
        config["inputs"]["artifacts"] = {}

        result = audit_chain.run(config)

        assert result.id == "DOC-401"
        assert result.passed is True
        assert result.details["files_hashed"] == 2

        # Verify chain file exists
        chain_file = Path(config["outputs"]["audit_chain_file"])
        assert chain_file.exists()


class TestBenchmarks:
    """Tests for TECH-001 benchmark check."""

    def test_benchmarks_pass(self, temp_dir, base_config):
        """Test benchmark validation passes."""

        bench_dir = temp_dir / "benchmarks"
        bench_dir.mkdir()

        # Create benchmark files
        np.savez(bench_dir / "test1.npz", speedup=15.0)
        np.savez(bench_dir / "test2.npz", baseline_time=100.0, quasim_time=5.0)

        config = base_config.copy()
        config["inputs"]["artifacts"]["benchmarks_npz_dir"] = str(bench_dir)
        config["policy"]["tolerances"]["benchmark_speedup_min"] = 10.0

        result = tech_benchmarks.run(config)

        assert result.id == "TECH-001"
        assert result.passed is True
        assert result.details["files_checked"] == 2


class TestIOUtilities:
    """Tests for I/O utilities."""

    def test_sha256_file(self, temp_dir):
        """Test SHA256 file hashing."""

        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        hash1 = sha256_file(str(test_file))
        hash2 = sha256_file(str(test_file))

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_write_read_json(self, temp_dir):
        """Test JSON write and read."""

        data = {"key": "value", "number": 42}
        json_file = temp_dir / "test.json"

        write_json(str(json_file), data)
        loaded = load_json(str(json_file))

        assert loaded == data

    def test_load_yaml(self, temp_dir):
        """Test YAML loading."""

        yaml_file = temp_dir / "test.yaml"
        data = {"key": "value", "list": [1, 2, 3]}

        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        loaded = load_yaml(str(yaml_file))
        assert loaded == data
