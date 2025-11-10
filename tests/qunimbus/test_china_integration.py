"""Tests for China Photonic Factory integration."""

from quasim.qunimbus.china_integration import (
    ChinaFactoryConfig,
    ChinaFactoryMetrics,
    ChinaPhotonicFactory,
)


def test_china_factory_config_defaults():
    """Test ChinaFactoryConfig default values."""
    config = ChinaFactoryConfig()

    assert config.partner == "Shenzhen Quantum Valley"
    assert config.capacity_qubits_per_year == 1_000_000
    assert config.pilots_per_day == 500
    assert config.qkd_enabled is True
    assert config.qkd_latency_ms == 0.18
    assert config.qkd_bandwidth_gbps == 1.0
    assert config.compliance_level == "MLPS Level 3"
    assert config.room_temperature is True


def test_china_factory_config_custom():
    """Test ChinaFactoryConfig with custom values."""
    config = ChinaFactoryConfig(
        pilots_per_day=1000,
        qkd_latency_ms=0.2,
        qkd_bandwidth_gbps=2.0,
    )

    assert config.pilots_per_day == 1000
    assert config.qkd_latency_ms == 0.2
    assert config.qkd_bandwidth_gbps == 2.0


def test_china_factory_initialization():
    """Test ChinaPhotonicFactory initialization."""
    factory = ChinaPhotonicFactory()

    assert factory.config.partner == "Shenzhen Quantum Valley"
    assert factory.metrics.pilots_generated_today == 0
    assert factory.metrics.qubits_capacity == 1_000_000
    assert factory.metrics.efficiency_multiplier == 22.1
    assert factory.metrics.mera_compression == 100.0
    assert not factory._connected


def test_china_factory_custom_config():
    """Test ChinaPhotonicFactory with custom config."""
    config = ChinaFactoryConfig(pilots_per_day=800)
    factory = ChinaPhotonicFactory(config)

    assert factory.config.pilots_per_day == 800


def test_china_factory_connect():
    """Test connection to China Photonic Factory."""
    factory = ChinaPhotonicFactory()

    result = factory.connect()

    assert result is True
    assert factory._connected is True


def test_china_factory_generate_pilots():
    """Test pilot generation at China factory."""
    factory = ChinaPhotonicFactory()

    result = factory.generate_pilots(count=100)

    assert result["status"] == "SUCCESS"
    assert result["pilots_generated"] == 100
    assert factory.metrics.pilots_generated_today == 100
    assert factory._connected is True  # Should auto-connect


def test_china_factory_multiple_generations():
    """Test multiple pilot generations."""
    factory = ChinaPhotonicFactory()

    factory.generate_pilots(100)
    factory.generate_pilots(200)

    assert factory.metrics.pilots_generated_today == 300


def test_china_factory_get_metrics():
    """Test metrics retrieval."""
    factory = ChinaPhotonicFactory()

    metrics = factory.get_metrics()

    assert isinstance(metrics, ChinaFactoryMetrics)
    assert metrics.qubits_capacity == 1_000_000
    assert metrics.efficiency_multiplier == 22.1
    assert metrics.timestamp is not None


def test_china_factory_compliance_status():
    """Test compliance status retrieval."""
    factory = ChinaPhotonicFactory()

    compliance = factory.get_compliance_status()

    assert "china_standards" in compliance
    assert "us_bridge" in compliance
    assert "qkd_security" in compliance
    assert "data_sovereignty" in compliance

    assert compliance["china_standards"]["mlps_level"] == "3"
    assert compliance["china_standards"]["status"] == "PASS"
    assert compliance["us_bridge"]["cmmc_l2"] == "compatible"
    assert compliance["qkd_security"]["protocol"] == "BB84"


def test_china_factory_disconnect():
    """Test disconnection from China factory."""
    factory = ChinaPhotonicFactory()

    factory.connect()
    assert factory._connected is True

    factory.disconnect()
    assert factory._connected is False


def test_china_factory_metrics_fields():
    """Test that metrics have all required fields."""
    factory = ChinaPhotonicFactory()
    metrics = factory.get_metrics()

    assert hasattr(metrics, "pilots_generated_today")
    assert hasattr(metrics, "qubits_capacity")
    assert hasattr(metrics, "efficiency_multiplier")
    assert hasattr(metrics, "mera_compression")
    assert hasattr(metrics, "uptime_percent")
    assert hasattr(metrics, "qkd_latency_ms")
    assert hasattr(metrics, "first_pilot_runtime_s")
    assert hasattr(metrics, "first_pilot_fidelity")
    assert hasattr(metrics, "timestamp")
