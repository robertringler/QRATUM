"""Tests for QRATUM Metrics Module.

Tests the QRATUM directive-mandated metrics:
- Outcome Superiority Ratio (OSR)
- Compute Efficiency Index (CEI)
- Sovereignty Factor (SF)
- Hallucination Risk Density (HRD)
"""

import pytest
from qratum.metrics import (
    QRATUMMetrics,
    MetricMeasurement,
    MetricStatus,
    MetricsAggregator,
    validate_module_metrics,
)


class TestQRATUMMetrics:
    """Tests for QRATUMMetrics dataclass."""

    def test_default_metrics_are_valid(self):
        """Default metrics should be valid."""
        metrics = QRATUMMetrics()
        assert metrics.is_valid()

    def test_negative_osr_is_invalid(self):
        """Negative OSR should be invalid."""
        metrics = QRATUMMetrics(outcome_superiority_ratio=-1.0)
        assert not metrics.is_valid()

    def test_sovereignty_factor_bounds(self):
        """SF must be between 0 and 1."""
        # Valid
        assert QRATUMMetrics(sovereignty_factor=0.0).is_valid()
        assert QRATUMMetrics(sovereignty_factor=1.0).is_valid()
        assert QRATUMMetrics(sovereignty_factor=0.5).is_valid()

        # Invalid
        assert not QRATUMMetrics(sovereignty_factor=-0.1).is_valid()
        assert not QRATUMMetrics(sovereignty_factor=1.1).is_valid()

    def test_hrd_bounds(self):
        """HRD must be between 0 and 1."""
        # Valid
        assert QRATUMMetrics(hallucination_risk_density=0.0).is_valid()
        assert QRATUMMetrics(hallucination_risk_density=0.5).is_valid()

        # Invalid
        assert not QRATUMMetrics(hallucination_risk_density=-0.1).is_valid()
        assert not QRATUMMetrics(hallucination_risk_density=1.1).is_valid()

    def test_record_operation_updates_metrics(self):
        """Recording operations should update metrics."""
        metrics = QRATUMMetrics()

        # Record some operations
        metrics.record_operation(external=False, deterministic=True)
        metrics.record_operation(external=False, deterministic=True)
        metrics.record_operation(external=True, deterministic=True)

        # SF should decrease with external calls
        assert metrics.sovereignty_factor < 1.0
        assert metrics.sovereignty_factor > 0.5  # 2/3 are local

        # HRD should be 0 (all deterministic)
        assert metrics.hallucination_risk_density == 0.0

    def test_record_non_deterministic_increases_hrd(self):
        """Non-deterministic operations increase HRD."""
        metrics = QRATUMMetrics()

        metrics.record_operation(deterministic=True)
        metrics.record_operation(deterministic=False)
        metrics.record_operation(deterministic=True)

        # 1/3 non-deterministic
        assert metrics.hallucination_risk_density == pytest.approx(1 / 3, rel=0.01)

    def test_record_outcome_updates_osr(self):
        """Recording outcomes should update OSR."""
        metrics = QRATUMMetrics()

        metrics.record_outcome(actual=2.0, baseline=1.0)
        assert metrics.outcome_superiority_ratio == 2.0

        metrics.record_outcome(actual=0.5, baseline=1.0)
        assert metrics.outcome_superiority_ratio == 0.5

    def test_to_dict(self):
        """Metrics should convert to dictionary."""
        metrics = QRATUMMetrics(
            outcome_superiority_ratio=1.5,
            compute_efficiency_index=2.0,
            sovereignty_factor=0.9,
            hallucination_risk_density=0.01,
        )

        d = metrics.to_dict()

        assert d["OSR"] == 1.5
        assert d["CEI"] == 2.0
        assert d["SF"] == 0.9
        assert d["HRD"] == 0.01


class TestMetricMeasurement:
    """Tests for MetricMeasurement."""

    def test_valid_measurement(self):
        """Valid measurements should report as valid."""
        m = MetricMeasurement(name="test", value=1.0)
        assert m.is_valid()

    def test_negative_value_is_invalid(self):
        """Negative values should be invalid."""
        m = MetricMeasurement(name="test", value=-1.0)
        assert not m.is_valid()

    def test_pending_status_is_invalid(self):
        """Pending status should be invalid."""
        m = MetricMeasurement(name="test", value=1.0, status=MetricStatus.PENDING)
        assert not m.is_valid()


class TestMetricsAggregator:
    """Tests for MetricsAggregator."""

    def test_aggregate_empty(self):
        """Aggregating empty should return default metrics."""
        agg = MetricsAggregator()
        result = agg.aggregate()
        assert isinstance(result, QRATUMMetrics)

    def test_register_and_collect(self):
        """Should collect from registered modules."""

        class MockModule:
            def get_metrics(self) -> QRATUMMetrics:
                return QRATUMMetrics(
                    outcome_superiority_ratio=1.5,
                    sovereignty_factor=0.8,
                )

        agg = MetricsAggregator()
        agg.register_module("test", MockModule())

        collected = agg.collect()
        assert "test" in collected
        assert collected["test"].outcome_superiority_ratio == 1.5

    def test_aggregate_multiple_modules(self):
        """Should aggregate metrics from multiple modules."""

        class Module1:
            def get_metrics(self) -> QRATUMMetrics:
                return QRATUMMetrics(
                    outcome_superiority_ratio=2.0,
                    sovereignty_factor=0.9,
                    hallucination_risk_density=0.01,
                )

        class Module2:
            def get_metrics(self) -> QRATUMMetrics:
                return QRATUMMetrics(
                    outcome_superiority_ratio=1.0,
                    sovereignty_factor=0.7,
                    hallucination_risk_density=0.05,
                )

        agg = MetricsAggregator()
        agg.register_module("m1", Module1())
        agg.register_module("m2", Module2())

        result = agg.aggregate()

        # OSR is averaged
        assert result.outcome_superiority_ratio == pytest.approx(1.5, rel=0.01)
        # SF is minimum (weakest link)
        assert result.sovereignty_factor == 0.7
        # HRD is maximum (worst case)
        assert result.hallucination_risk_density == 0.05

    def test_validate_all(self):
        """Should validate all registered modules."""

        class ValidModule:
            def get_metrics(self) -> QRATUMMetrics:
                return QRATUMMetrics()

        class InvalidModule:
            def get_metrics(self) -> QRATUMMetrics:
                return QRATUMMetrics(sovereignty_factor=-1.0)

        agg = MetricsAggregator()
        agg.register_module("valid", ValidModule())
        agg.register_module("invalid", InvalidModule())

        validity = agg.validate_all()

        assert validity["valid"] is True
        assert validity["invalid"] is False

    def test_get_invalid_modules(self):
        """Should return list of invalid modules."""

        class InvalidModule:
            def get_metrics(self) -> QRATUMMetrics:
                return QRATUMMetrics(hallucination_risk_density=2.0)  # Invalid

        agg = MetricsAggregator()
        agg.register_module("bad", InvalidModule())

        invalid = agg.get_invalid_modules()
        assert "bad" in invalid


class TestValidateModuleMetrics:
    """Tests for validate_module_metrics function."""

    def test_module_without_get_metrics(self):
        """Module without get_metrics should fail."""

        class NoMetrics:
            pass

        valid, msg = validate_module_metrics(NoMetrics())
        assert not valid
        assert "missing get_metrics" in msg

    def test_module_with_exception(self):
        """Module that raises exception should fail."""

        class RaisingModule:
            def get_metrics(self):
                raise RuntimeError("oops")

        valid, msg = validate_module_metrics(RaisingModule())
        assert not valid
        assert "exception" in msg.lower()

    def test_module_with_wrong_return_type(self):
        """Module returning wrong type should fail."""

        class WrongType:
            def get_metrics(self):
                return {"osr": 1.0}  # Wrong type

        valid, msg = validate_module_metrics(WrongType())
        assert not valid
        assert "QRATUMMetrics" in msg

    def test_valid_module(self):
        """Valid module should pass."""

        class ValidModule:
            def get_metrics(self) -> QRATUMMetrics:
                return QRATUMMetrics()

        valid, msg = validate_module_metrics(ValidModule())
        assert valid
        assert "valid" in msg.lower()
