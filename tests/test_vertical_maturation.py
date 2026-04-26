"""Tests for Vertical Maturation Module."""

from verticals.maturation import (
    TXO,
    CrossVerticalIntent,
    DeterministicPipeline,
    PipelineStage,
    PipelineStatus,
    TXORouter,
    TXOType,
    VerticalCoordinator,
)


class TestTXO:
    """Tests for TXO."""

    def test_txo_creation(self):
        """Test creating a TXO."""
        txo = TXO(
            txo_id="txo_001",
            txo_type=TXOType.DATA,
            source_vertical="VITRA",
            target_vertical="JURIS",
            payload_hash="hash123",
            provenance_hash="prov123",
            zone="Z1",
            timestamp="2025-01-01T00:00:00Z",
        )
        assert txo.txo_id == "txo_001"
        assert txo.txo_type == TXOType.DATA

    def test_txo_hash(self):
        """Test TXO hash is deterministic."""
        txo1 = TXO("t1", TXOType.DATA, "V1", "V2", "h1", "p1", "Z1", "2025-01-01T00:00:00Z")
        txo2 = TXO("t1", TXOType.DATA, "V1", "V2", "h1", "p1", "Z1", "2025-01-01T00:00:00Z")
        assert txo1.compute_hash() == txo2.compute_hash()

    def test_txo_serialization(self):
        """Test TXO serialization."""
        txo = TXO("t1", TXOType.EVENT, "V1", None, "h1", "p1", "Z1", "2025-01-01T00:00:00Z")
        serialized = txo.serialize()
        assert "txo_hash" in serialized
        assert serialized["txo_type"] == "event"


class TestTXORouter:
    """Tests for TXORouter."""

    def test_router_creation(self):
        """Test creating router."""
        router = TXORouter()
        assert len(router.txo_store) == 0

    def test_create_txo(self):
        """Test creating TXO via router."""
        router = TXORouter()
        txo = router.create_txo(
            txo_type=TXOType.DATA,
            source_vertical="VITRA",
            payload={"result": "data"},
            provenance_hash="prov123",
        )
        assert txo is not None
        assert txo.source_vertical == "VITRA"
        assert txo.txo_id in router.txo_store

    def test_route_txo(self):
        """Test routing TXO."""
        router = TXORouter()
        txo = router.create_txo(TXOType.DATA, "VITRA", {"data": "test"}, "prov")
        result = router.route_txo(txo.txo_id, ["JURIS", "SENTRA"])
        assert result is True
        assert txo.txo_id in router.routing_table

    def test_get_txos_for_vertical(self):
        """Test getting TXOs for a vertical."""
        router = TXORouter()
        txo1 = router.create_txo(TXOType.DATA, "VITRA", {"d": 1}, "p1")
        txo2 = router.create_txo(TXOType.DATA, "VITRA", {"d": 2}, "p2")
        router.route_txo(txo1.txo_id, ["JURIS"])
        router.route_txo(txo2.txo_id, ["SENTRA"])

        juris_txos = router.get_txos_for_vertical("JURIS")
        assert len(juris_txos) == 1
        assert juris_txos[0].txo_id == txo1.txo_id

    def test_verify_provenance_chain(self):
        """Test provenance chain verification."""
        router = TXORouter()
        # Use proper SHA256-length provenance hashes
        prov_hash_1 = "a" * 64  # SHA256 length
        prov_hash_2 = "b" * 64
        txo1 = router.create_txo(TXOType.DATA, "V1", {"d": 1}, prov_hash_1)
        txo2 = router.create_txo(TXOType.DATA, "V1", {"d": 2}, prov_hash_2)

        result = router.verify_provenance_chain([txo1.txo_id, txo2.txo_id])
        assert result is True


class TestPipelineStage:
    """Tests for PipelineStage."""

    def test_stage_creation(self):
        """Test creating a stage."""
        stage = PipelineStage(
            stage_id="stage_001",
            stage_name="Analysis",
            vertical="VITRA",
            operation="sequence_analysis",
        )
        assert stage.status == PipelineStatus.PENDING

    def test_compute_input_checksum(self):
        """Test input checksum computation."""
        stage = PipelineStage(
            stage_id="s1",
            stage_name="Test",
            vertical="V1",
            operation="op1",
        )
        checksum = stage.compute_input_checksum(["hash1", "hash2"])
        assert len(checksum) == 64  # SHA256 hex


class TestDeterministicPipeline:
    """Tests for DeterministicPipeline."""

    def test_pipeline_creation(self):
        """Test creating a pipeline."""
        pipeline = DeterministicPipeline(
            pipeline_id="pipe_001",
            pipeline_name="Test Pipeline",
            zone="Z1",
        )
        assert pipeline.status == PipelineStatus.PENDING
        assert pipeline.provenance_chain is not None

    def test_add_stage(self):
        """Test adding stages."""
        pipeline = DeterministicPipeline(
            pipeline_id="pipe_001",
            pipeline_name="Test",
        )
        stage = pipeline.add_stage("Stage 1", "VITRA", "analysis")
        assert len(pipeline.stages) == 1
        assert stage.vertical == "VITRA"

    def test_link_stages(self):
        """Test linking stages."""
        pipeline = DeterministicPipeline(
            pipeline_id="pipe_001",
            pipeline_name="Test",
        )
        s1 = pipeline.add_stage("S1", "VITRA", "op1")
        s2 = pipeline.add_stage("S2", "JURIS", "op2")
        pipeline.link_stages(s1.stage_id, s2.stage_id)
        assert s1.stage_id in s2.input_txos

    def test_get_next_stage(self):
        """Test getting next stage."""
        pipeline = DeterministicPipeline(
            pipeline_id="pipe_001",
            pipeline_name="Test",
        )
        pipeline.add_stage("S1", "VITRA", "op1")
        pipeline.add_stage("S2", "JURIS", "op2")

        stage = pipeline.get_next_stage()
        assert stage.stage_name == "S1"

    def test_advance_stage(self):
        """Test advancing to next stage."""
        pipeline = DeterministicPipeline(
            pipeline_id="pipe_001",
            pipeline_name="Test",
        )
        pipeline.add_stage("S1", "VITRA", "op1")
        pipeline.add_stage("S2", "JURIS", "op2")

        pipeline.advance_stage()
        assert pipeline._current_stage == 1

        pipeline.advance_stage()
        assert pipeline.status == PipelineStatus.COMPLETED


class TestCrossVerticalIntent:
    """Tests for CrossVerticalIntent."""

    def test_intent_creation(self):
        """Test creating an intent."""
        intent = CrossVerticalIntent(
            intent_id="intent_001",
            source_vertical="VITRA",
            target_verticals=["JURIS", "SENTRA"],
            operation="propagate_result",
            parameters={"key": "value"},
        )
        assert len(intent.target_verticals) == 2
        assert intent.propagation_status["JURIS"] == "pending"

    def test_mark_propagated(self):
        """Test marking target as propagated."""
        intent = CrossVerticalIntent(
            intent_id="i1",
            source_vertical="V1",
            target_verticals=["V2", "V3"],
            operation="op",
            parameters={},
        )
        intent.mark_propagated("V2")
        assert intent.propagation_status["V2"] == "propagated"

    def test_is_complete(self):
        """Test completion check."""
        intent = CrossVerticalIntent(
            intent_id="i1",
            source_vertical="V1",
            target_verticals=["V2", "V3"],
            operation="op",
            parameters={},
        )
        assert intent.is_complete() is False

        intent.mark_propagated("V2")
        intent.mark_propagated("V3")
        assert intent.is_complete() is True

    def test_z2_requires_dual_control(self):
        """Test Z2+ requires dual control."""
        intent = CrossVerticalIntent(
            intent_id="i1",
            source_vertical="V1",
            target_verticals=["V2"],
            operation="op",
            parameters={},
            zone="Z2",
            requires_dual_control=True,
        )
        assert intent.requires_dual_control is True


class TestVerticalCoordinator:
    """Tests for VerticalCoordinator."""

    def test_coordinator_creation(self):
        """Test creating coordinator."""
        coordinator = VerticalCoordinator()
        assert coordinator.txo_router is not None

    def test_create_pipeline(self):
        """Test creating a pipeline."""
        coordinator = VerticalCoordinator()
        pipeline = coordinator.create_pipeline("Test Pipeline", "Z1")
        assert pipeline is not None
        assert pipeline.pipeline_id in coordinator.pipelines

    def test_execute_pipeline_stage(self):
        """Test executing a pipeline stage."""
        coordinator = VerticalCoordinator()
        pipeline = coordinator.create_pipeline("Test")
        pipeline.add_stage("Stage1", "VITRA", "analysis")

        def mock_executor(vertical, operation, params):
            return {"result": f"{vertical}_{operation}"}

        txo = coordinator.execute_pipeline_stage(
            pipeline.pipeline_id,
            mock_executor,
            {"input": "data"},
        )

        assert txo is not None
        assert pipeline.stages[0].status == PipelineStatus.COMPLETED

    def test_create_cross_vertical_intent(self):
        """Test creating a cross-vertical intent."""
        coordinator = VerticalCoordinator()
        intent = coordinator.create_cross_vertical_intent(
            source_vertical="VITRA",
            target_verticals=["JURIS", "SENTRA"],
            operation="propagate",
            parameters={"data": "test"},
        )
        assert intent is not None
        assert intent.intent_id in coordinator.intents

    def test_propagate_intent(self):
        """Test propagating an intent."""
        coordinator = VerticalCoordinator()
        intent = coordinator.create_cross_vertical_intent("VITRA", ["JURIS"], "propagate", {})

        def mock_executor(target, op, params):
            return True

        results = coordinator.propagate_intent(intent.intent_id, mock_executor)
        assert results["JURIS"] is True
        assert intent.is_complete() is True

    def test_verify_vitra_e0_parity(self):
        """Test VITRA-E0 parity verification."""
        coordinator = VerticalCoordinator()
        coordinator.txo_router.create_txo(TXOType.DATA, "VITRA", {"d": 1}, "prov1")

        parity = coordinator.verify_vitra_e0_parity("VITRA")
        assert parity["determinism"] is True
        assert parity["provenance"] is True

    def test_get_stats(self):
        """Test getting statistics."""
        coordinator = VerticalCoordinator()
        coordinator.create_pipeline("P1")
        coordinator.create_cross_vertical_intent("V1", ["V2"], "op", {})

        stats = coordinator.get_stats()
        assert stats["total_pipelines"] == 1
        assert stats["total_intents"] == 1
