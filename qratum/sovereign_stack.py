"""QRATUM Sovereign Stack Specification.

This module formalizes the Sovereign Stack layers as defined in the
QRATUM ASCENSION DIRECTIVE:

| Layer       | Function                                    |
|-------------|---------------------------------------------|
| QRADLE      | Deterministic runtime substrate             |
| AetherNET   | Decentralized agent transport               |
| QuASIM      | Quantum-ready simulation lattice            |
| VITRA-E0    | Epistemic validation firewall               |
| DGEC        | Distributed governance enforcement          |
| MICRO-OS    | Edge-embedded cognition kernel              |
| DESKTOP-OS  | Human-interface shell                       |

All layers must remain operational offline (sovereignty requirement).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from qratum.metrics import QRATUMMetrics


class LayerStatus(Enum):
    """Operational status of a stack layer."""

    OFFLINE = "offline"
    INITIALIZING = "initializing"
    ONLINE = "online"
    DEGRADED = "degraded"
    ERROR = "error"


class ConnectivityMode(Enum):
    """Network connectivity mode."""

    CONNECTED = "connected"
    AIR_GAPPED = "air_gapped"
    MIXED = "mixed"


@dataclass
class LayerCapabilities:
    """Capabilities of a sovereign stack layer."""

    offline_operation: bool = True  # Must be True for sovereignty
    deterministic_execution: bool = True
    auditability: bool = True
    reversibility: bool = False
    quantum_ready: bool = False
    human_interface: bool = False
    edge_deployable: bool = False


@dataclass
class LayerMetrics(QRATUMMetrics):
    """Extended metrics for stack layers."""

    availability: float = 1.0  # Uptime ratio
    latency_ms: float = 0.0  # Average response time
    throughput: float = 0.0  # Operations per second
    error_rate: float = 0.0  # Error ratio


@runtime_checkable
class SovereignLayer(Protocol):
    """Protocol for sovereign stack layers."""

    @property
    def layer_id(self) -> str:
        """Unique layer identifier."""
        ...

    @property
    def status(self) -> LayerStatus:
        """Current operational status."""
        ...

    @property
    def capabilities(self) -> LayerCapabilities:
        """Layer capabilities."""
        ...

    def initialize(self) -> bool:
        """Initialize the layer. Returns success status."""
        ...

    def shutdown(self) -> bool:
        """Gracefully shutdown the layer. Returns success status."""
        ...

    def health_check(self) -> bool:
        """Check layer health. Returns True if healthy."""
        ...

    def get_metrics(self) -> LayerMetrics:
        """Get layer metrics."""
        ...


class BaseSovereignLayer(ABC):
    """Base implementation for sovereign stack layers."""

    def __init__(self, layer_id: str) -> None:
        """Initialize base layer.

        Args:
            layer_id: Unique layer identifier.
        """
        self._layer_id = layer_id
        self._status = LayerStatus.OFFLINE
        self._capabilities = LayerCapabilities()
        self._metrics = LayerMetrics()

    @property
    def layer_id(self) -> str:
        """Unique layer identifier."""
        return self._layer_id

    @property
    def status(self) -> LayerStatus:
        """Current operational status."""
        return self._status

    @property
    def capabilities(self) -> LayerCapabilities:
        """Layer capabilities."""
        return self._capabilities

    def initialize(self) -> bool:
        """Initialize the layer."""
        self._status = LayerStatus.INITIALIZING
        try:
            success = self._do_initialize()
            self._status = LayerStatus.ONLINE if success else LayerStatus.ERROR
            return success
        except Exception:
            self._status = LayerStatus.ERROR
            return False

    @abstractmethod
    def _do_initialize(self) -> bool:
        """Layer-specific initialization."""
        ...

    def shutdown(self) -> bool:
        """Gracefully shutdown the layer."""
        try:
            success = self._do_shutdown()
            self._status = LayerStatus.OFFLINE
            return success
        except Exception:
            self._status = LayerStatus.ERROR
            return False

    @abstractmethod
    def _do_shutdown(self) -> bool:
        """Layer-specific shutdown."""
        ...

    def health_check(self) -> bool:
        """Check layer health."""
        if self._status != LayerStatus.ONLINE:
            return False
        return self._do_health_check()

    @abstractmethod
    def _do_health_check(self) -> bool:
        """Layer-specific health check."""
        ...

    def get_metrics(self) -> LayerMetrics:
        """Get layer metrics."""
        return self._metrics


# ============================================================================
# LAYER IMPLEMENTATIONS
# ============================================================================


class QRADLELayer(BaseSovereignLayer):
    """QRADLE - Deterministic Runtime Substrate.

    The foundational layer providing:
    - Deterministic execution guarantees
    - Contract-based computation
    - Merkle-chained audit trails
    - Rollback capabilities
    """

    def __init__(self) -> None:
        """Initialize QRADLE layer."""
        super().__init__("QRADLE")
        self._capabilities = LayerCapabilities(
            offline_operation=True,
            deterministic_execution=True,
            auditability=True,
            reversibility=True,
            quantum_ready=False,
            human_interface=False,
            edge_deployable=True,
        )

    def _do_initialize(self) -> bool:
        """Initialize QRADLE runtime."""
        # Initialize Merkle chain, contract engine, etc.
        return True

    def _do_shutdown(self) -> bool:
        """Shutdown QRADLE runtime."""
        # Persist state, close connections
        return True

    def _do_health_check(self) -> bool:
        """Check QRADLE health."""
        # Verify Merkle chain integrity, contract engine status
        return True


class AetherNETLayer(BaseSovereignLayer):
    """AetherNET - Decentralized Agent Transport.

    Provides:
    - P2P agent communication
    - Decentralized message routing
    - Anti-censorship transport
    - Offline-capable mesh networking
    """

    def __init__(self) -> None:
        """Initialize AetherNET layer."""
        super().__init__("AetherNET")
        self._capabilities = LayerCapabilities(
            offline_operation=True,  # Local mesh still works
            deterministic_execution=True,
            auditability=True,
            reversibility=False,
            quantum_ready=False,
            human_interface=False,
            edge_deployable=True,
        )
        self._connectivity_mode = ConnectivityMode.AIR_GAPPED

    @property
    def connectivity_mode(self) -> ConnectivityMode:
        """Current connectivity mode."""
        return self._connectivity_mode

    def _do_initialize(self) -> bool:
        """Initialize AetherNET transport."""
        # Initialize P2P stack, discovery protocol
        return True

    def _do_shutdown(self) -> bool:
        """Shutdown AetherNET transport."""
        # Close connections, persist peer state
        return True

    def _do_health_check(self) -> bool:
        """Check AetherNET health."""
        # Verify P2P connectivity, message queue status
        return True


class QuASIMLayer(BaseSovereignLayer):
    """QuASIM - Quantum-Ready Simulation Lattice.

    Provides:
    - Classical simulation engine
    - Quantum algorithm simulation
    - VQE/QAOA support
    - Hybrid classical-quantum workflows
    """

    def __init__(self) -> None:
        """Initialize QuASIM layer."""
        super().__init__("QuASIM")
        self._capabilities = LayerCapabilities(
            offline_operation=True,
            deterministic_execution=True,  # Simulation is deterministic
            auditability=True,
            reversibility=False,
            quantum_ready=True,
            human_interface=False,
            edge_deployable=False,  # Requires significant compute
        )

    def _do_initialize(self) -> bool:
        """Initialize QuASIM simulation engine."""
        # Initialize simulation backends
        return True

    def _do_shutdown(self) -> bool:
        """Shutdown QuASIM simulation engine."""
        return True

    def _do_health_check(self) -> bool:
        """Check QuASIM health."""
        # Verify simulation backends available
        return True


class VITRAE0Layer(BaseSovereignLayer):
    """VITRA-E0 - Epistemic Validation Firewall.

    Provides:
    - Input validation and sanitization
    - Epistemic confidence scoring
    - Hallucination detection
    - Output verification
    """

    def __init__(self) -> None:
        """Initialize VITRA-E0 layer."""
        super().__init__("VITRA-E0")
        self._capabilities = LayerCapabilities(
            offline_operation=True,
            deterministic_execution=True,
            auditability=True,
            reversibility=False,
            quantum_ready=False,
            human_interface=False,
            edge_deployable=True,
        )

    def _do_initialize(self) -> bool:
        """Initialize VITRA-E0 validation engine."""
        return True

    def _do_shutdown(self) -> bool:
        """Shutdown VITRA-E0 validation engine."""
        return True

    def _do_health_check(self) -> bool:
        """Check VITRA-E0 health."""
        return True


class DGECLayer(BaseSovereignLayer):
    """DGEC - Distributed Governance Enforcement Controller.

    Provides:
    - Stake-weighted governance
    - Policy enforcement
    - Consensus mechanisms
    - Slashing conditions
    """

    def __init__(self) -> None:
        """Initialize DGEC layer."""
        super().__init__("DGEC")
        self._capabilities = LayerCapabilities(
            offline_operation=True,  # Local governance cache
            deterministic_execution=True,
            auditability=True,
            reversibility=False,
            quantum_ready=False,
            human_interface=False,
            edge_deployable=True,
        )

    def _do_initialize(self) -> bool:
        """Initialize DGEC governance engine."""
        return True

    def _do_shutdown(self) -> bool:
        """Shutdown DGEC governance engine."""
        return True

    def _do_health_check(self) -> bool:
        """Check DGEC health."""
        return True


class MICROOSLayer(BaseSovereignLayer):
    """MICRO-OS - Edge-Embedded Cognition Kernel.

    Provides:
    - Minimal footprint AI runtime
    - Edge device deployment
    - Low-latency inference
    - Offline-first design
    """

    def __init__(self) -> None:
        """Initialize MICRO-OS layer."""
        super().__init__("MICRO-OS")
        self._capabilities = LayerCapabilities(
            offline_operation=True,
            deterministic_execution=True,
            auditability=True,
            reversibility=False,
            quantum_ready=False,
            human_interface=False,
            edge_deployable=True,
        )

    def _do_initialize(self) -> bool:
        """Initialize MICRO-OS kernel."""
        return True

    def _do_shutdown(self) -> bool:
        """Shutdown MICRO-OS kernel."""
        return True

    def _do_health_check(self) -> bool:
        """Check MICRO-OS health."""
        return True


class DESKTOPOSLayer(BaseSovereignLayer):
    """DESKTOP-OS - Human Interface Shell.

    Provides:
    - User interface components
    - Visualization tools
    - Human-in-the-loop workflows
    - Accessibility features
    """

    def __init__(self) -> None:
        """Initialize DESKTOP-OS layer."""
        super().__init__("DESKTOP-OS")
        self._capabilities = LayerCapabilities(
            offline_operation=True,
            deterministic_execution=True,
            auditability=True,
            reversibility=False,
            quantum_ready=False,
            human_interface=True,
            edge_deployable=True,
        )

    def _do_initialize(self) -> bool:
        """Initialize DESKTOP-OS shell."""
        return True

    def _do_shutdown(self) -> bool:
        """Shutdown DESKTOP-OS shell."""
        return True

    def _do_health_check(self) -> bool:
        """Check DESKTOP-OS health."""
        return True


# ============================================================================
# SOVEREIGN STACK ORCHESTRATOR
# ============================================================================


@dataclass
class StackStatus:
    """Overall status of the sovereign stack."""

    all_online: bool = False
    layer_statuses: dict[str, LayerStatus] = field(default_factory=dict)
    degraded_layers: list[str] = field(default_factory=list)
    offline_capable: bool = True
    aggregate_metrics: LayerMetrics = field(default_factory=LayerMetrics)


class SovereignStack:
    """Orchestrator for the complete sovereign stack.

    Manages initialization, monitoring, and coordination of all layers.
    Ensures offline operation capability is maintained.
    """

    def __init__(self) -> None:
        """Initialize sovereign stack."""
        self._layers: dict[str, BaseSovereignLayer] = {}
        self._initialization_order = [
            "QRADLE",
            "AetherNET",
            "QuASIM",
            "VITRA-E0",
            "DGEC",
            "MICRO-OS",
            "DESKTOP-OS",
        ]

        # Register default layers
        self._register_default_layers()

    def _register_default_layers(self) -> None:
        """Register the default sovereign stack layers."""
        self._layers["QRADLE"] = QRADLELayer()
        self._layers["AetherNET"] = AetherNETLayer()
        self._layers["QuASIM"] = QuASIMLayer()
        self._layers["VITRA-E0"] = VITRAE0Layer()
        self._layers["DGEC"] = DGECLayer()
        self._layers["MICRO-OS"] = MICROOSLayer()
        self._layers["DESKTOP-OS"] = DESKTOPOSLayer()

    def register_layer(self, layer: BaseSovereignLayer) -> None:
        """Register a layer with the stack.

        Args:
            layer: Layer to register.

        Raises:
            ValueError: If layer doesn't support offline operation.
        """
        if not layer.capabilities.offline_operation:
            raise ValueError(
                f"Layer {layer.layer_id} does not support offline operation. "
                "All sovereign stack layers must remain operational offline."
            )
        self._layers[layer.layer_id] = layer

    def initialize_all(self) -> bool:
        """Initialize all layers in order.

        Returns:
            True if all layers initialized successfully.
        """
        for layer_id in self._initialization_order:
            if layer_id in self._layers:
                if not self._layers[layer_id].initialize():
                    return False
        return True

    def shutdown_all(self) -> bool:
        """Shutdown all layers in reverse order.

        Returns:
            True if all layers shutdown successfully.
        """
        for layer_id in reversed(self._initialization_order):
            if layer_id in self._layers:
                if not self._layers[layer_id].shutdown():
                    return False
        return True

    def get_status(self) -> StackStatus:
        """Get current stack status.

        Returns:
            Complete stack status.
        """
        status = StackStatus()

        for layer_id, layer in self._layers.items():
            layer_status = layer.status
            status.layer_statuses[layer_id] = layer_status

            if layer_status == LayerStatus.DEGRADED:
                status.degraded_layers.append(layer_id)
            elif layer_status != LayerStatus.ONLINE:
                status.all_online = False

            if not layer.capabilities.offline_operation:
                status.offline_capable = False

        status.all_online = all(
            s == LayerStatus.ONLINE for s in status.layer_statuses.values()
        )

        return status

    def health_check_all(self) -> dict[str, bool]:
        """Run health check on all layers.

        Returns:
            Dictionary mapping layer IDs to health status.
        """
        return {lid: layer.health_check() for lid, layer in self._layers.items()}

    def get_layer(self, layer_id: str) -> BaseSovereignLayer | None:
        """Get a specific layer.

        Args:
            layer_id: Layer identifier.

        Returns:
            Layer instance or None if not found.
        """
        return self._layers.get(layer_id)

    def get_metrics(self) -> QRATUMMetrics:
        """Get aggregated metrics for the stack.

        Returns:
            Aggregated QRATUM metrics.
        """
        metrics = QRATUMMetrics()

        if not self._layers:
            return metrics

        # Aggregate metrics from all layers
        all_metrics = [layer.get_metrics() for layer in self._layers.values()]
        
        # Filter out None metrics (defensive)
        valid_metrics = [m for m in all_metrics if m is not None]
        
        if not valid_metrics:
            return metrics

        metrics.outcome_superiority_ratio = (
            sum(m.outcome_superiority_ratio for m in valid_metrics) / len(valid_metrics)
        )
        metrics.compute_efficiency_index = (
            sum(m.compute_efficiency_index for m in valid_metrics) / len(valid_metrics)
        )
        # Sovereignty is the weakest link
        metrics.sovereignty_factor = min(m.sovereignty_factor for m in valid_metrics)
        # HRD is worst case
        metrics.hallucination_risk_density = max(
            m.hallucination_risk_density for m in valid_metrics
        )

        return metrics


# Singleton stack instance
_sovereign_stack: SovereignStack | None = None


def get_sovereign_stack() -> SovereignStack:
    """Get the global sovereign stack instance.

    Returns:
        Sovereign stack singleton.
    """
    global _sovereign_stack
    if _sovereign_stack is None:
        _sovereign_stack = SovereignStack()
    return _sovereign_stack
