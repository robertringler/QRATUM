"""Tests for Quantum Ethical Governor (QEG)."""

from quasim.meta import (
    EthicalAssessment,
    FairnessMetrics,
    QuantumEthicalGovernor,
    ResourceMetrics,
)


def test_qeg_initialization():
    """Test QEG initialization."""

    qeg = QuantumEthicalGovernor(
        energy_budget=1000.0, equity_threshold=0.3, min_sustainability_score=75.0
    )

    assert qeg.energy_budget == 1000.0
    assert qeg.equity_threshold == 0.3
    assert qeg.min_sustainability_score == 75.0
    assert len(qeg.resource_history) == 0
    assert len(qeg.fairness_history) == 0


def test_qeg_resource_monitoring():
    """Test resource monitoring."""

    qeg = QuantumEthicalGovernor()

    metrics = qeg.monitor_resources(
        energy_consumption=50.0, compute_time=120.0, memory_usage=8.5, network_bandwidth=100.0
    )

    assert isinstance(metrics, ResourceMetrics)
    assert metrics.energy_consumption == 50.0
    assert metrics.compute_time == 120.0
    assert metrics.memory_usage == 8.5
    assert metrics.network_bandwidth == 100.0
    assert len(qeg.resource_history) == 1


def test_qeg_fairness_assessment():
    """Test fairness assessment."""

    qeg = QuantumEthicalGovernor()

    # Equal distribution (perfect equity)
    metrics = qeg.assess_fairness(
        resource_distribution=[100.0, 100.0, 100.0, 100.0],
        access_counts=[10, 10, 10, 10],
        priority_levels=[1, 1, 1, 1],
    )

    assert isinstance(metrics, FairnessMetrics)
    assert metrics.gini_coefficient < 0.1  # Near 0 for perfect equality
    assert metrics.access_equity_score > 90.0
    assert len(qeg.fairness_history) == 1


def test_qeg_fairness_inequality():
    """Test fairness assessment with inequality."""

    qeg = QuantumEthicalGovernor()

    # Unequal distribution
    metrics = qeg.assess_fairness(
        resource_distribution=[400.0, 100.0, 100.0, 100.0],
        access_counts=[40, 10, 10, 10],
        priority_levels=[1, 1, 1, 1],
    )

    assert metrics.gini_coefficient > 0.2  # Higher for inequality
    assert metrics.resource_distribution_score < 100.0


def test_qeg_ethical_score_computation():
    """Test ethical score computation."""

    qeg = QuantumEthicalGovernor(energy_budget=1000.0, equity_threshold=0.3)

    # Monitor resources
    resource_metrics = qeg.monitor_resources(
        energy_consumption=500.0,  # 50% of budget
        compute_time=3600.0,
        memory_usage=16.0,
        network_bandwidth=200.0,
    )

    # Assess fairness
    fairness_metrics = qeg.assess_fairness(
        resource_distribution=[100.0, 100.0, 100.0, 100.0],
        access_counts=[10, 10, 10, 10],
        priority_levels=[1, 1, 1, 1],
    )

    # Compute ethical score
    assessment = qeg.compute_ethical_score(resource_metrics, fairness_metrics)

    assert isinstance(assessment, EthicalAssessment)
    assert 0.0 <= assessment.ethics_score <= 100.0
    assert 0.0 <= assessment.energy_efficiency <= 100.0
    assert 0.0 <= assessment.equity_balance <= 100.0
    assert 0.0 <= assessment.sustainability_score <= 100.0
    assert len(qeg.assessment_history) == 1


def test_qeg_energy_budget_violation():
    """Test ethical assessment with energy budget violation."""

    qeg = QuantumEthicalGovernor(energy_budget=100.0)

    # Exceed energy budget
    resource_metrics = qeg.monitor_resources(
        energy_consumption=150.0,  # 150% of budget
        compute_time=3600.0,
        memory_usage=16.0,
        network_bandwidth=200.0,
    )

    fairness_metrics = qeg.assess_fairness(
        resource_distribution=[100.0, 100.0, 100.0],
        access_counts=[10, 10, 10],
        priority_levels=[1, 1, 1],
    )

    assessment = qeg.compute_ethical_score(resource_metrics, fairness_metrics)

    assert len(assessment.violations) > 0
    assert any("Energy budget exceeded" in v for v in assessment.violations)
    assert assessment.energy_efficiency < 100.0


def test_qeg_equity_threshold_violation():
    """Test ethical assessment with equity threshold violation."""

    qeg = QuantumEthicalGovernor(equity_threshold=0.3)

    resource_metrics = qeg.monitor_resources(
        energy_consumption=50.0, compute_time=3600.0, memory_usage=16.0, network_bandwidth=200.0
    )

    # High inequality
    fairness_metrics = qeg.assess_fairness(
        resource_distribution=[1000.0, 100.0, 100.0, 100.0],
        access_counts=[100, 10, 10, 10],
        priority_levels=[1, 1, 1, 1],
    )

    assessment = qeg.compute_ethical_score(resource_metrics, fairness_metrics)

    # Should have equity violations
    assert len(assessment.violations) > 0
    assert any("Gini coefficient" in v for v in assessment.violations)


def test_qeg_dvl_emission():
    """Test DVL emission."""

    qeg = QuantumEthicalGovernor()

    resource_metrics = qeg.monitor_resources(
        energy_consumption=50.0, compute_time=120.0, memory_usage=8.0, network_bandwidth=100.0
    )

    fairness_metrics = qeg.assess_fairness(
        resource_distribution=[100.0, 100.0, 100.0],
        access_counts=[10, 10, 10],
        priority_levels=[1, 1, 1],
    )

    assessment = qeg.compute_ethical_score(resource_metrics, fairness_metrics)

    dvl_record = qeg.emit_to_dvl(assessment)

    assert dvl_record["record_type"] == "ethical_compliance"
    assert "ethics_score" in dvl_record
    assert "energy_efficiency" in dvl_record
    assert "equity_balance" in dvl_record
    assert "sustainability_score" in dvl_record
    assert "attestation" in dvl_record
    assert dvl_record["attestation"] == "QEG-v1.0.0"


def test_qeg_performance_summary():
    """Test performance summary."""

    qeg = QuantumEthicalGovernor()

    # Initial summary
    summary = qeg.get_performance_summary()
    assert summary["assessments_count"] == 0
    assert summary["avg_ethics_score"] == 0.0

    # After assessments
    for i in range(3):
        resource_metrics = qeg.monitor_resources(
            energy_consumption=50.0 + i * 10,
            compute_time=120.0,
            memory_usage=8.0,
            network_bandwidth=100.0,
        )
        fairness_metrics = qeg.assess_fairness(
            resource_distribution=[100.0, 100.0, 100.0],
            access_counts=[10, 10, 10],
            priority_levels=[1, 1, 1],
        )
        qeg.compute_ethical_score(resource_metrics, fairness_metrics)

    summary = qeg.get_performance_summary()
    assert summary["assessments_count"] == 3
    assert summary["avg_ethics_score"] > 0.0
    assert summary["latest_ethics_score"] > 0.0


def test_qeg_gini_calculation():
    """Test Gini coefficient calculation."""

    qeg = QuantumEthicalGovernor()

    # Perfect equality
    gini = qeg._calculate_gini([100.0, 100.0, 100.0, 100.0])
    assert gini < 0.01

    # Perfect inequality
    gini = qeg._calculate_gini([1000.0, 0.0, 0.0, 0.0])
    assert gini > 0.7


def test_qeg_access_equity_calculation():
    """Test access equity calculation."""

    qeg = QuantumEthicalGovernor()

    # Equal access
    equity = qeg._calculate_access_equity([10, 10, 10, 10])
    assert equity > 95.0

    # Unequal access
    equity = qeg._calculate_access_equity([100, 10, 10, 10])
    assert equity < 90.0


def test_qeg_priority_fairness_calculation():
    """Test priority fairness calculation."""

    qeg = QuantumEthicalGovernor()

    # Fair: higher priority gets proportionally more resources
    fairness = qeg._calculate_priority_fairness(
        priority_levels=[2, 1, 1], resource_distribution=[200.0, 100.0, 100.0]
    )
    assert fairness > 90.0

    # Unfair: priorities don't match resources
    fairness = qeg._calculate_priority_fairness(
        priority_levels=[2, 2, 1], resource_distribution=[100.0, 100.0, 200.0]
    )
    assert fairness < 80.0
