"""Tests for causality audit."""

from compiled_realities.audit.causality_audit import CausalityAuditor
from compiled_realities.models.cptp_process_tensor import CPTPProcessTensorModel


def test_auditor_initialization():
    """Test auditor initialization."""
    config = {"test": "value"}
    auditor = CausalityAuditor(config)

    assert auditor.config == config
    assert auditor.audit_log == []


def test_no_future_data_access_check():
    """Test no future data access verification."""
    config = {}
    auditor = CausalityAuditor(config)

    # Should pass: all accessed data <= t1
    passed = auditor.verify_no_future_data_access(t1=10, data_accessed=[0, 5, 10])
    assert passed

    # Should fail: some accessed data > t1
    passed = auditor.verify_no_future_data_access(t1=10, data_accessed=[0, 5, 15])
    assert not passed


def test_twin_run_invariance_check():
    """Test twin run invariance verification."""
    config = {
        "dimension": 2,
        "environment_coupling": 0.1,
        "t1_t2_separation": 5,
        "n_timesteps": 30,
        "dt": 0.1,
    }

    model = CPTPProcessTensorModel(config, seed=42)
    results_I0, results_I1 = model.run_twin_simulations()

    auditor = CausalityAuditor({})
    t2 = results_I0["metadata"]["t2"]

    # Should pass: twin runs with same seed before intervention
    passed = auditor.verify_twin_run_invariance(results_I0, results_I1, t2)
    assert passed


def test_compute_hash():
    """Test hash computation."""
    config = {}
    auditor = CausalityAuditor(config)

    data1 = {"a": 1, "b": 2}
    data2 = {"a": 1, "b": 2}
    data3 = {"a": 1, "b": 3}

    hash1 = auditor.compute_hash(data1)
    hash2 = auditor.compute_hash(data2)
    hash3 = auditor.compute_hash(data3)

    assert hash1 == hash2  # Same data should have same hash
    assert hash1 != hash3  # Different data should have different hash


def test_create_audit_report():
    """Test audit report creation."""
    config = {"test": "config"}
    auditor = CausalityAuditor(config)

    # Add some audit entries
    auditor.verify_no_future_data_access(t1=10, data_accessed=[0, 5, 10])

    results = {"test": "results"}
    report = auditor.create_audit_report(results)

    assert "config_hash" in report
    assert "results_hash" in report
    assert "audit_log" in report
    assert "all_checks_passed" in report
    assert len(report["audit_log"]) >= 1


def test_verify_causality():
    """Test full causality verification."""
    config = {
        "dimension": 2,
        "environment_coupling": 0.1,
        "t1_t2_separation": 5,
        "n_timesteps": 30,
        "dt": 0.1,
    }

    model = CPTPProcessTensorModel(config, seed=42)
    results_I0, results_I1 = model.run_twin_simulations()

    auditor = CausalityAuditor(config)
    passed = auditor.verify_causality(model, results_I0, results_I1)

    assert passed
    assert len(auditor.audit_log) > 0
