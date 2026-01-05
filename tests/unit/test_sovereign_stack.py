"""Tests for Sovereign Stack Module.

Tests the formalized sovereign stack layers:
- QRADLE: Deterministic runtime substrate
- AetherNET: Decentralized agent transport
- QuASIM: Quantum-ready simulation lattice
- VITRA-E0: Epistemic validation firewall
- DGEC: Distributed governance enforcement
- MICRO-OS: Edge-embedded cognition kernel
- DESKTOP-OS: Human-interface shell
"""

import pytest

from qratum.sovereign_stack import (
    AetherNETLayer,
    BaseSovereignLayer,
    ConnectivityMode,
    DESKTOPOSLayer,
    DGECLayer,
    LayerCapabilities,
    LayerMetrics,
    LayerStatus,
    MICROOSLayer,
    QRADLELayer,
    QuASIMLayer,
    SovereignStack,
    VITRAE0Layer,
    get_sovereign_stack,
)


class TestLayerCapabilities:
    """Tests for LayerCapabilities."""

    def test_default_capabilities(self):
        """Default capabilities should support offline operation."""
        caps = LayerCapabilities()
        assert caps.offline_operation is True
        assert caps.deterministic_execution is True
        assert caps.auditability is True

    def test_custom_capabilities(self):
        """Should create custom capabilities."""
        caps = LayerCapabilities(
            offline_operation=True,
            quantum_ready=True,
            human_interface=True,
        )

        assert caps.quantum_ready is True
        assert caps.human_interface is True


class TestLayerMetrics:
    """Tests for LayerMetrics."""

    def test_extends_qratum_metrics(self):
        """Should include QRATUM metrics."""
        metrics = LayerMetrics()

        # Base QRATUM metrics
        assert hasattr(metrics, "outcome_superiority_ratio")
        assert hasattr(metrics, "sovereignty_factor")

        # Extended metrics
        assert hasattr(metrics, "availability")
        assert hasattr(metrics, "latency_ms")


class TestQRADLELayer:
    """Tests for QRADLE layer."""

    def test_creation(self):
        """Should create QRADLE layer."""
        layer = QRADLELayer()
        assert layer.layer_id == "QRADLE"

    def test_capabilities(self):
        """Should have correct capabilities."""
        layer = QRADLELayer()
        caps = layer.capabilities

        assert caps.offline_operation is True
        assert caps.deterministic_execution is True
        assert caps.reversibility is True
        assert caps.edge_deployable is True

    def test_initialize_and_shutdown(self):
        """Should initialize and shutdown successfully."""
        layer = QRADLELayer()

        assert layer.initialize()
        assert layer.status == LayerStatus.ONLINE

        assert layer.shutdown()
        assert layer.status == LayerStatus.OFFLINE

    def test_health_check(self):
        """Health check should pass when online."""
        layer = QRADLELayer()

        # Not online yet
        assert not layer.health_check()

        layer.initialize()
        assert layer.health_check()

        layer.shutdown()
        assert not layer.health_check()


class TestAetherNETLayer:
    """Tests for AetherNET layer."""

    def test_creation(self):
        """Should create AetherNET layer."""
        layer = AetherNETLayer()
        assert layer.layer_id == "AetherNET"

    def test_connectivity_mode(self):
        """Should have connectivity mode."""
        layer = AetherNETLayer()
        assert layer.connectivity_mode == ConnectivityMode.AIR_GAPPED

    def test_offline_operation(self):
        """Should support offline operation."""
        layer = AetherNETLayer()
        assert layer.capabilities.offline_operation is True


class TestQuASIMLayer:
    """Tests for QuASIM layer."""

    def test_creation(self):
        """Should create QuASIM layer."""
        layer = QuASIMLayer()
        assert layer.layer_id == "QuASIM"

    def test_quantum_ready(self):
        """Should be quantum ready."""
        layer = QuASIMLayer()
        assert layer.capabilities.quantum_ready is True

    def test_not_edge_deployable(self):
        """Should not be edge deployable (requires compute)."""
        layer = QuASIMLayer()
        assert layer.capabilities.edge_deployable is False


class TestVITRAE0Layer:
    """Tests for VITRA-E0 layer."""

    def test_creation(self):
        """Should create VITRA-E0 layer."""
        layer = VITRAE0Layer()
        assert layer.layer_id == "VITRA-E0"


class TestDGECLayer:
    """Tests for DGEC layer."""

    def test_creation(self):
        """Should create DGEC layer."""
        layer = DGECLayer()
        assert layer.layer_id == "DGEC"


class TestMICROOSLayer:
    """Tests for MICRO-OS layer."""

    def test_creation(self):
        """Should create MICRO-OS layer."""
        layer = MICROOSLayer()
        assert layer.layer_id == "MICRO-OS"

    def test_edge_deployable(self):
        """Should be edge deployable."""
        layer = MICROOSLayer()
        assert layer.capabilities.edge_deployable is True


class TestDESKTOPOSLayer:
    """Tests for DESKTOP-OS layer."""

    def test_creation(self):
        """Should create DESKTOP-OS layer."""
        layer = DESKTOPOSLayer()
        assert layer.layer_id == "DESKTOP-OS"

    def test_human_interface(self):
        """Should have human interface capability."""
        layer = DESKTOPOSLayer()
        assert layer.capabilities.human_interface is True


class TestSovereignStack:
    """Tests for SovereignStack orchestrator."""

    def test_creation(self):
        """Should create stack with default layers."""
        stack = SovereignStack()

        assert stack.get_layer("QRADLE") is not None
        assert stack.get_layer("AetherNET") is not None
        assert stack.get_layer("QuASIM") is not None
        assert stack.get_layer("VITRA-E0") is not None
        assert stack.get_layer("DGEC") is not None
        assert stack.get_layer("MICRO-OS") is not None
        assert stack.get_layer("DESKTOP-OS") is not None

    def test_initialize_all(self):
        """Should initialize all layers."""
        stack = SovereignStack()
        assert stack.initialize_all()

        status = stack.get_status()
        assert status.all_online

    def test_shutdown_all(self):
        """Should shutdown all layers."""
        stack = SovereignStack()
        stack.initialize_all()

        assert stack.shutdown_all()

        status = stack.get_status()
        assert not status.all_online

    def test_health_check_all(self):
        """Should check health of all layers."""
        stack = SovereignStack()
        stack.initialize_all()

        health = stack.health_check_all()

        assert all(health.values())

    def test_offline_capability(self):
        """All layers should support offline operation."""
        stack = SovereignStack()
        status = stack.get_status()

        assert status.offline_capable

    def test_register_non_offline_layer_fails(self):
        """Should reject layers that don't support offline."""

        class NonOfflineLayer(BaseSovereignLayer):
            def __init__(self):
                super().__init__("non-offline")
                self._capabilities = LayerCapabilities(offline_operation=False)

            def _do_initialize(self):
                return True

            def _do_shutdown(self):
                return True

            def _do_health_check(self):
                return True

        stack = SovereignStack()

        with pytest.raises(ValueError, match="offline operation"):
            stack.register_layer(NonOfflineLayer())

    def test_get_metrics(self):
        """Should return aggregated metrics."""
        stack = SovereignStack()
        stack.initialize_all()

        metrics = stack.get_metrics()
        assert metrics.is_valid()

    def test_get_status_tracks_degraded(self):
        """Should track degraded layers."""
        stack = SovereignStack()
        stack.initialize_all()

        # Manually set a layer to degraded
        layer = stack.get_layer("QRADLE")
        layer._status = LayerStatus.DEGRADED

        status = stack.get_status()
        assert "QRADLE" in status.degraded_layers
        assert not status.all_online


class TestGetSovereignStack:
    """Tests for get_sovereign_stack singleton."""

    def test_returns_stack(self):
        """Should return a SovereignStack instance."""
        stack = get_sovereign_stack()
        assert isinstance(stack, SovereignStack)

    def test_returns_same_instance(self):
        """Should return the same instance."""
        stack1 = get_sovereign_stack()
        stack2 = get_sovereign_stack()

        assert stack1 is stack2


class TestAllLayersOfflineCapable:
    """Verify all layers support offline operation (sovereignty requirement)."""

    @pytest.mark.parametrize(
        "layer_class",
        [
            QRADLELayer,
            AetherNETLayer,
            QuASIMLayer,
            VITRAE0Layer,
            DGECLayer,
            MICROOSLayer,
            DESKTOPOSLayer,
        ],
    )
    def test_layer_offline_capable(self, layer_class):
        """Each layer must support offline operation."""
        layer = layer_class()
        assert (
            layer.capabilities.offline_operation is True
        ), f"{layer.layer_id} must support offline operation per sovereignty requirement"

    @pytest.mark.parametrize(
        "layer_class",
        [
            QRADLELayer,
            AetherNETLayer,
            QuASIMLayer,
            VITRAE0Layer,
            DGECLayer,
            MICROOSLayer,
            DESKTOPOSLayer,
        ],
    )
    def test_layer_deterministic(self, layer_class):
        """Each layer must support deterministic execution."""
        layer = layer_class()
        assert layer.capabilities.deterministic_execution is True
