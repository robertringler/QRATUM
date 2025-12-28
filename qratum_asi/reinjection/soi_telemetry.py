"""SOI Telemetry Integration for Autonomous Reinjection Orchestrator.

Provides real-time telemetry streaming for the autonomous reinjection system
to the Sovereign Operations Interface (SOI) dashboard.

Features:
- Real-time reinjection status updates
- Cross-vertical propagation visualization
- Optionality density metrics
- Historical evolution graphs
- Merkle-protected audit status

Version: 1.0.0
Status: Production Ready
QuASIM: v2025.12.26
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from qradle.merkle import MerkleChain
from qratum_asi.reinjection.autonomous_orchestrator import (
    AutonomousReinjectionOrchestrator,
    DiscoveryArtifact,
    PropagationResult,
)
from qratum_asi.reinjection.engine import ReinjectionCycleResult


@dataclass
class TelemetryEvent:
    """Telemetry event for SOI streaming.

    Attributes:
        event_type: Type of telemetry event
        payload: Event data payload
        epoch: Current epoch number
        timestamp: ISO timestamp
        proof: Merkle proof
    """

    event_type: str
    payload: dict[str, Any]
    epoch: int
    timestamp: str
    proof: str | None = None

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class OptionalityMetrics:
    """Optionality density metrics for SOI visualization.

    Attributes:
        total_options: Total number of reinjection options
        active_options: Currently active options
        successful_rate: Success rate of reinjections
        cross_vertical_density: Density of cross-vertical connections
        entropy_reduction: Overall entropy reduction
        mutual_information_gain: Total MI gain from reinjections
    """

    total_options: int = 0
    active_options: int = 0
    successful_rate: float = 0.0
    cross_vertical_density: float = 0.0
    entropy_reduction: float = 0.0
    mutual_information_gain: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class EvolutionDataPoint:
    """Data point for historical evolution graphs.

    Attributes:
        timestamp: ISO timestamp
        artifacts_processed: Cumulative artifacts processed
        reinjections_completed: Cumulative reinjections
        propagations_executed: Cumulative propagations
        success_rate: Rolling success rate
        system_health: System health score
    """

    timestamp: str
    artifacts_processed: int
    reinjections_completed: int
    propagations_executed: int
    success_rate: float
    system_health: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class SOIReinjectionTelemetry:
    """SOI telemetry integration for autonomous reinjection.

    Provides:
    - Real-time status broadcasting
    - Cross-vertical propagation updates
    - Optionality density metrics
    - Historical evolution tracking
    - Merkle audit status
    """

    def __init__(
        self,
        orchestrator: AutonomousReinjectionOrchestrator,
        merkle_chain: MerkleChain | None = None,
    ):
        """Initialize telemetry integration.

        Args:
            orchestrator: The autonomous orchestrator to monitor
            merkle_chain: Optional Merkle chain for provenance
        """
        self.orchestrator = orchestrator
        self.merkle_chain = merkle_chain or MerkleChain()

        # State tracking
        self.current_epoch = 0
        self.evolution_history: list[EvolutionDataPoint] = []
        self.telemetry_events: list[TelemetryEvent] = []

        # Callbacks for WebSocket broadcasting
        self._broadcast_callbacks: list[Callable[[TelemetryEvent], None]] = []

        # Register with orchestrator
        self._register_orchestrator_callbacks()

        # Log initialization
        self.merkle_chain.add_event(
            "soi_telemetry_initialized",
            {
                "version": "1.0.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _register_orchestrator_callbacks(self) -> None:
        """Register callbacks with the orchestrator."""
        self.orchestrator.register_artifact_callback(self._on_artifact_submitted)
        self.orchestrator.register_reinjection_callback(self._on_reinjection_completed)
        self.orchestrator.register_propagation_callback(self._on_propagation_completed)

    def register_broadcast_callback(self, callback: Callable[[TelemetryEvent], None]) -> None:
        """Register a callback for telemetry broadcasts.

        Args:
            callback: Function to call with each telemetry event
        """
        self._broadcast_callbacks.append(callback)

    def _broadcast(self, event: TelemetryEvent) -> None:
        """Broadcast a telemetry event to all registered callbacks."""
        self.telemetry_events.append(event)

        for callback in self._broadcast_callbacks:
            try:
                callback(event)
            except Exception:
                pass  # Don't let callback errors affect operation

    def _create_event(
        self,
        event_type: str,
        payload: dict[str, Any],
    ) -> TelemetryEvent:
        """Create a new telemetry event.

        Args:
            event_type: Type of event
            payload: Event payload

        Returns:
            Created TelemetryEvent
        """
        self.current_epoch += 1

        event = TelemetryEvent(
            event_type=event_type,
            payload=payload,
            epoch=self.current_epoch,
            timestamp=datetime.now(timezone.utc).isoformat(),
            proof=self.merkle_chain.get_chain_proof(),
        )

        # Log to merkle chain
        self.merkle_chain.add_event(
            f"telemetry_{event_type}",
            {
                "epoch": self.current_epoch,
                "event_type": event_type,
            },
        )

        return event

    def _on_artifact_submitted(self, artifact: DiscoveryArtifact) -> None:
        """Handle artifact submission event."""
        event = self._create_event(
            "reinjection:artifact_submitted",
            {
                "artifact_id": artifact.artifact_id,
                "domain": artifact.domain.value,
                "confidence": artifact.confidence,
                "fidelity": artifact.fidelity,
                "sensitivity": artifact.sensitivity.value,
                "source_pipeline": artifact.source_pipeline,
            },
        )
        self._broadcast(event)

    def _on_reinjection_completed(self, result: ReinjectionCycleResult) -> None:
        """Handle reinjection completion event."""
        event = self._create_event(
            "reinjection:cycle_completed",
            {
                "cycle_id": result.cycle_id,
                "candidate_id": result.candidate.candidate_id,
                "success": result.success,
                "error_message": result.error_message if not result.success else None,
                "execution_time_ms": result.execution_time_ms,
                "validation_passed": (
                    result.validation_result.valid if result.validation_result else False
                ),
            },
        )
        self._broadcast(event)

        # Update evolution history
        self._record_evolution_point()

    def _on_propagation_completed(self, propagation: PropagationResult) -> None:
        """Handle propagation completion event."""
        event = self._create_event(
            "reinjection:propagation_completed",
            {
                "propagation_id": propagation.propagation_id,
                "source_artifact_id": propagation.source_artifact_id,
                "target_verticals": [t.value for t in propagation.target_verticals],
                "status": propagation.propagation_status,
                "impact_metrics": propagation.impact_metrics,
            },
        )
        self._broadcast(event)

    def _record_evolution_point(self) -> None:
        """Record a point in the evolution history."""
        status = self.orchestrator.get_status_summary()

        # Calculate success rate
        total_completed = status.reinjections_completed + status.reinjections_failed
        success_rate = (
            status.reinjections_completed / total_completed if total_completed > 0 else 1.0
        )

        # Calculate system health
        system_health = 1.0 - min(1.0, status.current_system_load)

        point = EvolutionDataPoint(
            timestamp=datetime.now(timezone.utc).isoformat(),
            artifacts_processed=status.total_artifacts_monitored,
            reinjections_completed=status.reinjections_completed,
            propagations_executed=status.cross_vertical_propagations,
            success_rate=success_rate,
            system_health=system_health,
        )

        self.evolution_history.append(point)

        # Limit history size
        if len(self.evolution_history) > 1000:
            self.evolution_history = self.evolution_history[-1000:]

    def get_reinjection_status(self) -> dict[str, Any]:
        """Get current reinjection status for SOI dashboard.

        Returns:
            Status dictionary with:
            - system_state: Current orchestrator state
            - summary: Status summary
            - recent_activity: Recent telemetry events
        """
        status = self.orchestrator.get_status_summary()

        return {
            "system_state": self.orchestrator.system_state.value,
            "summary": status.to_dict(),
            "epoch": self.current_epoch,
            "recent_events": [e.to_dict() for e in self.telemetry_events[-10:]],
            "merkle_proof": self.merkle_chain.get_chain_proof(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_cross_vertical_propagation_status(self) -> dict[str, Any]:
        """Get cross-vertical propagation status for visualization.

        Returns:
            Propagation status with:
            - active_propagations: Currently active propagations
            - vertical_impacts: Impact metrics per vertical
            - dependency_graph: Simplified dependency graph
        """
        propagations = list(self.orchestrator.propagation_results.values())

        # Aggregate impacts by vertical
        vertical_impacts: dict[str, float] = {}
        for prop in propagations:
            for vertical, impact in prop.impact_metrics.items():
                if vertical in vertical_impacts:
                    vertical_impacts[vertical] += impact
                else:
                    vertical_impacts[vertical] = impact

        # Build simplified dependency graph from module-level constant
        from qratum_asi.reinjection.autonomous_orchestrator import CROSS_VERTICAL_DEPENDENCIES

        dependency_graph = {
            domain.value: [{"target": target.value, "weight": weight} for target, weight in deps]
            for domain, deps in CROSS_VERTICAL_DEPENDENCIES.items()
        }

        return {
            "total_propagations": len(propagations),
            "vertical_impacts": vertical_impacts,
            "dependency_graph": dependency_graph,
            "recent_propagations": [p.to_dict() for p in propagations[-5:]],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_optionality_metrics(self) -> OptionalityMetrics:
        """Get optionality density metrics for SOI.

        Returns:
            OptionalityMetrics with current values
        """
        status = self.orchestrator.get_status_summary()
        stats = self.orchestrator.get_orchestrator_stats()

        # Calculate metrics
        total_options = status.artifacts_filtered + status.reinjections_pending
        active_options = status.reinjections_pending

        # Success rate
        total_completed = status.reinjections_completed + status.reinjections_failed
        success_rate = (
            status.reinjections_completed / total_completed if total_completed > 0 else 0.0
        )

        # Cross-vertical density
        total_propagations = status.cross_vertical_propagations
        density = (
            total_propagations / max(status.reinjections_completed, 1)
            if status.reinjections_completed > 0
            else 0.0
        )

        # Aggregate MI and entropy from completed reinjections
        mi_gain = 0.0
        entropy_reduction = 0.0
        for result in self.orchestrator.completed_reinjections.values():
            if result.success and result.reinjection_result:
                metrics = result.reinjection_result.metrics
                mi_gain += metrics.get("mutual_information_gain", 0.0)
                entropy_reduction += metrics.get("entropy_reduction", 0.0)

        return OptionalityMetrics(
            total_options=total_options,
            active_options=active_options,
            successful_rate=success_rate,
            cross_vertical_density=density,
            entropy_reduction=entropy_reduction,
            mutual_information_gain=mi_gain,
        )

    def get_evolution_history(
        self,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get historical evolution data for graphs.

        Args:
            limit: Maximum number of points to return

        Returns:
            List of evolution data points
        """
        points = self.evolution_history[-limit:]
        return [p.to_dict() for p in points]

    def get_audit_status(self) -> dict[str, Any]:
        """Get Merkle-protected audit status.

        Returns:
            Audit status with:
            - chain_length: Length of Merkle chain
            - chain_valid: Chain integrity status
            - latest_proof: Latest Merkle proof
            - recent_audits: Recent audit records
        """
        # Get audit reports from completed reinjections
        recent_audits = []
        for artifact_id, result in list(self.orchestrator.completed_reinjections.items())[-5:]:
            if result.audit_report:
                recent_audits.append(
                    {
                        "artifact_id": artifact_id,
                        "report_id": result.audit_report.report_id,
                        "compliance_summary": result.audit_report.get_compliance_summary(),
                    }
                )

        return {
            "chain_length": len(self.merkle_chain.chain),
            "chain_valid": self.merkle_chain.verify_integrity(),
            "latest_proof": self.merkle_chain.get_chain_proof(),
            "orchestrator_chain_valid": self.orchestrator.verify_provenance(),
            "recent_audits": recent_audits,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_full_dashboard_state(self) -> dict[str, Any]:
        """Get complete dashboard state for SOI.

        Returns:
            Complete state dictionary for SOI visualization
        """
        return {
            "reinjection_status": self.get_reinjection_status(),
            "propagation_status": self.get_cross_vertical_propagation_status(),
            "optionality_metrics": self.get_optionality_metrics().to_dict(),
            "evolution_history": self.get_evolution_history(),
            "audit_status": self.get_audit_status(),
            "orchestrator_stats": self.orchestrator.get_orchestrator_stats(),
        }

    def get_telemetry_stats(self) -> dict[str, Any]:
        """Get telemetry statistics.

        Returns:
            Statistics dictionary
        """
        event_types = {}
        for event in self.telemetry_events:
            event_type = event.event_type
            event_types[event_type] = event_types.get(event_type, 0) + 1

        return {
            "total_events": len(self.telemetry_events),
            "current_epoch": self.current_epoch,
            "events_by_type": event_types,
            "evolution_points": len(self.evolution_history),
            "merkle_chain_length": len(self.merkle_chain.chain),
        }
