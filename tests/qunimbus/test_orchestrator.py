"""Tests for QuNimbus Wave 3 orchestration."""

import pytest

from quasim.qunimbus.orchestrator import (
    ExecutionMode,
    OrchestrationConfig,
    QuNimbusOrchestrator,
)


def test_orchestration_config_defaults():
    """Test OrchestrationConfig default values."""

    config = OrchestrationConfig()

    assert config.parallel is True
    assert config.tasks == ["wave3_launch", "china_photonic_scale"]
    assert config.mode == ExecutionMode.LIVE_ACCELERATED
    assert config.wave == 3
    assert config.pilot_target == 1000
    assert config.china_enabled is True
    assert len(config.compliance) == 4


def test_orchestration_config_custom():
    """Test OrchestrationConfig with custom values."""

    config = OrchestrationConfig(
        parallel=False,
        tasks=["wave3_launch"],
        mode=ExecutionMode.SIMULATION,
        pilot_target=500,
        china_enabled=False,
    )

    assert config.parallel is False
    assert config.tasks == ["wave3_launch"]
    assert config.mode == ExecutionMode.SIMULATION
    assert config.pilot_target == 500
    assert config.china_enabled is False


def test_orchestrator_initialization():
    """Test QuNimbusOrchestrator initialization."""

    config = OrchestrationConfig()
    orchestrator = QuNimbusOrchestrator(config)

    assert orchestrator.config == config
    assert orchestrator.metrics.pilots_generated == 0
    assert orchestrator.metrics.efficiency_multiplier == 0.0
    assert orchestrator.metrics.mera_compression == 0.0


@pytest.mark.asyncio
async def test_execute_wave3_launch():
    """Test Wave 3 launch execution."""

    config = OrchestrationConfig()
    orchestrator = QuNimbusOrchestrator(config)

    result = await orchestrator.execute_wave3_launch()

    assert result["status"] == "SUCCESS"
    assert result["task"] == "wave3_launch"
    assert result["pilots_generated"] == 1000
    assert result["efficiency"] == "22.0×"
    assert result["mera_compression"] == "100.0×"
    assert result["rl_convergence"] == "99.1%"
    assert orchestrator.metrics.akron_pilots == 1000
    assert orchestrator.metrics.efficiency_multiplier == 22.0


@pytest.mark.asyncio
async def test_execute_china_photonic_scale():
    """Test China Photonic Factory execution."""

    config = OrchestrationConfig()
    orchestrator = QuNimbusOrchestrator(config)

    result = await orchestrator.execute_china_photonic_scale()

    assert result["status"] == "SUCCESS"
    assert result["task"] == "china_photonic_scale"
    assert result["pilots_generated"] == 500
    assert result["capacity"] == "1M+ qubits/yr"
    assert result["qkd_latency_ms"] == 0.18
    assert result["uptime"] == "100%"
    assert orchestrator.metrics.china_pilots == 500


@pytest.mark.asyncio
async def test_execute_parallel():
    """Test parallel execution of both tasks."""

    config = OrchestrationConfig(parallel=True)
    orchestrator = QuNimbusOrchestrator(config)

    result = await orchestrator.execute_parallel()

    assert result["status"] == "SUCCESS"
    assert result["mode"] == "parallel"
    assert "wave3_result" in result
    assert "china_result" in result
    assert result["combined_pilots_per_day"] == 1500
    assert result["total_value_unlocked"] == "$20B/yr"
    assert orchestrator.metrics.pilots_generated == 1500


@pytest.mark.asyncio
async def test_orchestrate():
    """Test main orchestration entry point."""

    config = OrchestrationConfig()
    orchestrator = QuNimbusOrchestrator(config)

    result = await orchestrator.orchestrate()

    assert result["status"] == "SUCCESS"
    assert "wave3_result" in result or "china_result" in result


def test_get_metrics():
    """Test metrics retrieval."""

    config = OrchestrationConfig()
    orchestrator = QuNimbusOrchestrator(config)

    metrics = orchestrator.get_metrics()

    assert metrics.pilots_generated == 0
    assert metrics.efficiency_multiplier == 0.0
    assert metrics.timestamp is not None
