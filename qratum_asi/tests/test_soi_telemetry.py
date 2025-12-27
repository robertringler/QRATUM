"""Tests for SOI Telemetry Integration.

Validates the SOI dashboard telemetry integration including:
- Telemetry event generation
- Broadcast callback registration
- Optionality metrics calculation
- Evolution history tracking
- Dashboard state aggregation
"""

import pytest

from qratum_asi.reinjection import (
    AutonomousReinjectionOrchestrator,
    DiscoveryDomain,
    EvolutionDataPoint,
    OptionalityMetrics,
    SOIReinjectionTelemetry,
    TelemetryEvent,
)


class TestTelemetryEvent:
    """Tests for TelemetryEvent."""

    def test_event_creation(self):
        """Test creating a telemetry event."""
        event = TelemetryEvent(
            event_type="test:event",
            payload={"key": "value"},
            epoch=1,
            timestamp="2025-12-27T00:00:00+00:00",
            proof="a" * 64,
        )

        assert event.event_type == "test:event"
        assert event.epoch == 1
        assert event.proof == "a" * 64

    def test_event_serialization(self):
        """Test event serializes to JSON."""
        event = TelemetryEvent(
            event_type="test:event",
            payload={"key": "value"},
            epoch=1,
            timestamp="2025-12-27T00:00:00+00:00",
        )

        json_str = event.to_json()
        assert "test:event" in json_str
        assert "epoch" in json_str

    def test_event_to_dict(self):
        """Test event converts to dictionary."""
        event = TelemetryEvent(
            event_type="test:event",
            payload={"key": "value"},
            epoch=1,
            timestamp="2025-12-27T00:00:00+00:00",
        )

        data = event.to_dict()
        assert data["event_type"] == "test:event"
        assert data["epoch"] == 1


class TestOptionalityMetrics:
    """Tests for OptionalityMetrics."""

    def test_metrics_creation(self):
        """Test creating optionality metrics."""
        metrics = OptionalityMetrics(
            total_options=10,
            active_options=5,
            successful_rate=0.85,
            cross_vertical_density=0.5,
        )

        assert metrics.total_options == 10
        assert metrics.active_options == 5
        assert metrics.successful_rate == 0.85

    def test_metrics_serialization(self):
        """Test metrics serializes correctly."""
        metrics = OptionalityMetrics(
            total_options=10,
            active_options=5,
        )

        data = metrics.to_dict()
        assert "total_options" in data
        assert "timestamp" in data


class TestEvolutionDataPoint:
    """Tests for EvolutionDataPoint."""

    def test_datapoint_creation(self):
        """Test creating evolution data point."""
        point = EvolutionDataPoint(
            timestamp="2025-12-27T00:00:00+00:00",
            artifacts_processed=100,
            reinjections_completed=50,
            propagations_executed=25,
            success_rate=0.9,
            system_health=0.95,
        )

        assert point.artifacts_processed == 100
        assert point.success_rate == 0.9

    def test_datapoint_serialization(self):
        """Test data point serializes correctly."""
        point = EvolutionDataPoint(
            timestamp="2025-12-27T00:00:00+00:00",
            artifacts_processed=100,
            reinjections_completed=50,
            propagations_executed=25,
            success_rate=0.9,
            system_health=0.95,
        )

        data = point.to_dict()
        assert "artifacts_processed" in data
        assert data["system_health"] == 0.95


class TestSOIReinjectionTelemetry:
    """Tests for SOIReinjectionTelemetry."""

    def test_telemetry_creation(self):
        """Test creating telemetry integration."""
        orchestrator = AutonomousReinjectionOrchestrator()
        telemetry = SOIReinjectionTelemetry(orchestrator)

        assert telemetry is not None
        assert telemetry.current_epoch == 0

    def test_artifact_triggers_telemetry(self):
        """Test artifact submission triggers telemetry event."""
        orchestrator = AutonomousReinjectionOrchestrator()
        telemetry = SOIReinjectionTelemetry(orchestrator)

        received_events = []

        def on_event(event):
            received_events.append(event)

        telemetry.register_broadcast_callback(on_event)

        # Submit an artifact
        orchestrator.submit_artifact(
            source_pipeline="test_pipeline",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="Test discovery",
            data_payload={"test": "data"},
            confidence=0.96,
            fidelity=0.9999,
            provenance_hash="a" * 64,
        )

        assert len(received_events) > 0
        assert received_events[0].event_type == "reinjection:artifact_submitted"

    def test_reinjection_triggers_telemetry(self):
        """Test reinjection triggers telemetry event."""
        orchestrator = AutonomousReinjectionOrchestrator()
        telemetry = SOIReinjectionTelemetry(orchestrator)

        received_events = []

        def on_event(event):
            received_events.append(event)

        telemetry.register_broadcast_callback(on_event)

        # Submit and reinject
        artifact = orchestrator.submit_artifact(
            source_pipeline="test_pipeline",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="Test discovery",
            data_payload={"compound_data": {"name": "Test"}},
            confidence=0.96,
            fidelity=0.9999,
            provenance_hash="b" * 64,
        )

        orchestrator.auto_reinject_if_valid(artifact, auto_approve=True)

        # Should have artifact and reinjection events
        event_types = [e.event_type for e in received_events]
        assert "reinjection:artifact_submitted" in event_types
        assert "reinjection:cycle_completed" in event_types

    def test_get_reinjection_status(self):
        """Test getting reinjection status."""
        orchestrator = AutonomousReinjectionOrchestrator()
        telemetry = SOIReinjectionTelemetry(orchestrator)

        status = telemetry.get_reinjection_status()

        assert "system_state" in status
        assert "summary" in status
        assert "epoch" in status
        assert "merkle_proof" in status

    def test_get_propagation_status(self):
        """Test getting propagation status."""
        orchestrator = AutonomousReinjectionOrchestrator()
        telemetry = SOIReinjectionTelemetry(orchestrator)

        status = telemetry.get_cross_vertical_propagation_status()

        assert "total_propagations" in status
        assert "vertical_impacts" in status
        assert "dependency_graph" in status

    def test_get_optionality_metrics(self):
        """Test getting optionality metrics."""
        orchestrator = AutonomousReinjectionOrchestrator()
        telemetry = SOIReinjectionTelemetry(orchestrator)

        # Submit some artifacts
        orchestrator.submit_artifact(
            source_pipeline="pipeline_1",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="Discovery 1",
            data_payload={"test": "data"},
            confidence=0.96,
            fidelity=0.9999,
            provenance_hash="c" * 64,
        )

        orchestrator.filter_artifacts()

        metrics = telemetry.get_optionality_metrics()

        assert isinstance(metrics, OptionalityMetrics)
        assert metrics.total_options >= 0

    def test_get_evolution_history(self):
        """Test getting evolution history."""
        orchestrator = AutonomousReinjectionOrchestrator()
        telemetry = SOIReinjectionTelemetry(orchestrator)

        # Submit and reinject to generate history
        artifact = orchestrator.submit_artifact(
            source_pipeline="test_pipeline",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="Test discovery",
            data_payload={"compound_data": {"name": "Test"}},
            confidence=0.96,
            fidelity=0.9999,
            provenance_hash="d" * 64,
        )

        orchestrator.auto_reinject_if_valid(artifact, auto_approve=True)

        history = telemetry.get_evolution_history()

        assert isinstance(history, list)
        if len(history) > 0:
            assert "timestamp" in history[0]
            assert "success_rate" in history[0]

    def test_get_audit_status(self):
        """Test getting audit status."""
        orchestrator = AutonomousReinjectionOrchestrator()
        telemetry = SOIReinjectionTelemetry(orchestrator)

        status = telemetry.get_audit_status()

        assert "chain_length" in status
        assert "chain_valid" in status
        assert "latest_proof" in status

    def test_get_full_dashboard_state(self):
        """Test getting full dashboard state."""
        orchestrator = AutonomousReinjectionOrchestrator()
        telemetry = SOIReinjectionTelemetry(orchestrator)

        state = telemetry.get_full_dashboard_state()

        assert "reinjection_status" in state
        assert "propagation_status" in state
        assert "optionality_metrics" in state
        assert "evolution_history" in state
        assert "audit_status" in state

    def test_telemetry_stats(self):
        """Test getting telemetry statistics."""
        orchestrator = AutonomousReinjectionOrchestrator()
        telemetry = SOIReinjectionTelemetry(orchestrator)

        # Generate some events
        orchestrator.submit_artifact(
            source_pipeline="test_pipeline",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="Test",
            data_payload={"test": "data"},
            confidence=0.96,
            fidelity=0.9999,
            provenance_hash="e" * 64,
        )

        stats = telemetry.get_telemetry_stats()

        assert "total_events" in stats
        assert "current_epoch" in stats
        assert "events_by_type" in stats


class TestTelemetryIntegration:
    """Integration tests for telemetry."""

    def test_full_cycle_telemetry(self):
        """Test telemetry through full reinjection cycle."""
        orchestrator = AutonomousReinjectionOrchestrator()
        telemetry = SOIReinjectionTelemetry(orchestrator)

        all_events = []

        def collect_events(event):
            all_events.append(event)

        telemetry.register_broadcast_callback(collect_events)

        # Submit multiple artifacts
        for i in range(3):
            artifact = orchestrator.submit_artifact(
                source_pipeline=f"pipeline_{i}",
                domain=DiscoveryDomain.BIODISCOVERY,
                description=f"Discovery {i}",
                data_payload={"compound_data": {"name": f"Compound {i}"}},
                confidence=0.96 + i * 0.01,
                fidelity=0.9999,
                provenance_hash=f"{chr(ord('f') + i)}" * 64,
            )

            orchestrator.auto_reinject_if_valid(artifact, auto_approve=True)

        # Verify events
        assert len(all_events) >= 6  # At least 3 submissions + 3 completions

        # Verify evolution history
        history = telemetry.get_evolution_history()
        assert len(history) >= 1

        # Verify full state
        state = telemetry.get_full_dashboard_state()
        assert state["reinjection_status"]["summary"]["total_artifacts_monitored"] == 3

    def test_multiple_callback_registration(self):
        """Test multiple callbacks receive events."""
        orchestrator = AutonomousReinjectionOrchestrator()
        telemetry = SOIReinjectionTelemetry(orchestrator)

        events_1 = []
        events_2 = []

        telemetry.register_broadcast_callback(lambda e: events_1.append(e))
        telemetry.register_broadcast_callback(lambda e: events_2.append(e))

        orchestrator.submit_artifact(
            source_pipeline="test_pipeline",
            domain=DiscoveryDomain.BIODISCOVERY,
            description="Test",
            data_payload={"test": "data"},
            confidence=0.96,
            fidelity=0.9999,
            provenance_hash="g" * 64,
        )

        assert len(events_1) == len(events_2)
        assert len(events_1) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
