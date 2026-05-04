"""Tests for temporal engine core functionality."""

import pytest

from qratum.temporal import (
    StateChain,
    TemporalCoordinate,
    TemporalEngine,
    TemporalState,
)


class TestTemporalEngine:
    """Test the core temporal engine"""

    def test_engine_initialization(self):
        """Test that engine initializes correctly"""
        engine = TemporalEngine(
            peak_flops=2.5e18,
            deterministic=True,
            merkle_verified=True,
        )

        assert engine.config.peak_flops == 2.5e18
        assert engine.config.deterministic is True
        assert engine.config.merkle_verified is True
        assert engine.operation_count == 0

    def test_forward_computation(self):
        """Test forward time travel"""
        engine = TemporalEngine(deterministic=True)

        # Simple evolution function: increment by delta_t
        def evolve(state, dt):
            return state + dt

        initial_state = 0.0
        delta_t = 100.0

        final_state, proof = engine.forward(
            initial_state=initial_state,
            delta_t=delta_t,
            evolution_fn=evolve,
        )

        assert final_state.data == pytest.approx(delta_t, rel=1e-6)
        assert proof.operation == "forward"
        assert proof.computational_delta_t == delta_t
        assert "effective_velocity_c" in proof.metadata

    def test_backward_computation(self):
        """Test backward time travel"""
        engine = TemporalEngine(deterministic=True)

        # Inverse evolution function: decrement by delta_t
        def inverse_evolve(state, dt):
            return state - dt

        final_state = 100.0
        delta_t = -100.0

        initial_state, proof = engine.backward(
            final_state=final_state,
            delta_t=delta_t,
            inverse_fn=inverse_evolve,
        )

        assert initial_state.data == pytest.approx(0.0, rel=1e-6)
        assert proof.operation == "backward"

    def test_timeline_branching(self):
        """Test branching into parallel timelines"""
        engine = TemporalEngine(deterministic=True)

        def evolve(state, dt):
            return state + dt

        branch_point = 10.0
        branches = [
            ("path_a", lambda s: s + 1.0),
            ("path_b", lambda s: s + 2.0),
        ]

        results = engine.branch(
            branch_point=branch_point,
            branches=branches,
            evolution_fn=evolve,
            delta_t=50.0,
        )

        assert len(results) == 2
        assert "path_a" in results
        assert "path_b" in results

        state_a, proof_a = results["path_a"]
        state_b, proof_b = results["path_b"]

        # path_a starts at 11.0, evolves for 50.0
        assert state_a.data == pytest.approx(61.0, rel=1e-6)
        # path_b starts at 12.0, evolves for 50.0
        assert state_b.data == pytest.approx(62.0, rel=1e-6)

    def test_timeline_convergence(self):
        """Test merging parallel timelines"""
        engine = TemporalEngine(deterministic=True)

        # Create some temporal states
        coord1 = TemporalCoordinate(
            state_hash="hash1",
            timeline_id="t1",
            computational_t=100.0,
            physical_t=1.0,
            depth=10,
        )
        state1 = TemporalState(
            coordinate=coord1,
            data=10.0,
        )

        coord2 = TemporalCoordinate(
            state_hash="hash2",
            timeline_id="t2",
            computational_t=100.0,
            physical_t=1.0,
            depth=10,
        )
        state2 = TemporalState(
            coordinate=coord2,
            data=20.0,
        )

        branches = {"branch1": state1, "branch2": state2}

        # Merge function: average
        def merge(states):
            return sum(s.data for s in states) / len(states)

        converged_state, proof = engine.converge(
            branches=branches,
            convergence_fn=merge,
        )

        assert converged_state.data == pytest.approx(15.0)
        assert proof.operation == "converge"

    def test_temporal_consistency_verification(self):
        """Test verification of temporal consistency"""
        engine = TemporalEngine(deterministic=True, merkle_verified=True)

        def evolve(state, dt):
            return state + dt

        final_state, proof = engine.forward(
            initial_state=0.0,
            delta_t=100.0,
            evolution_fn=evolve,
        )

        timeline_id = final_state.coordinate.timeline_id
        is_consistent = engine.verify_temporal_consistency(timeline_id)

        assert is_consistent is True

    def test_statistics_collection(self):
        """Test that statistics are collected correctly"""
        engine = TemporalEngine(deterministic=True)

        def evolve(state, dt):
            return state + dt

        # Perform some operations
        engine.forward(initial_state=0.0, delta_t=10.0, evolution_fn=evolve)
        engine.forward(initial_state=0.0, delta_t=20.0, evolution_fn=evolve)

        stats = engine.get_statistics()

        assert stats["total_operations"] == 2
        assert stats["active_timelines"] == 2
        assert stats["config"]["deterministic"] is True


class TestTemporalState:
    """Test temporal state management"""

    def test_temporal_coordinate_creation(self):
        """Test creating temporal coordinates"""
        coord = TemporalCoordinate(
            state_hash="abc123",
            timeline_id="timeline_1",
            computational_t=100.0,
            physical_t=1.0,
            depth=5,
        )

        assert coord.state_hash == "abc123"
        assert coord.timeline_id == "timeline_1"
        assert coord.computational_t == 100.0
        assert coord.depth == 5

    def test_temporal_state_hash_computation(self):
        """Test state hash computation"""
        coord = TemporalCoordinate(
            state_hash="",
            timeline_id="t1",
            computational_t=0.0,
            physical_t=0.0,
            depth=0,
        )

        state = TemporalState(
            coordinate=coord,
            data={"value": 42},
            parent_hash=None,
        )

        hash1 = state.compute_hash()
        assert len(hash1) == 64  # SHA-256 produces 64 hex chars

        # Same state should produce same hash
        hash2 = state.compute_hash()
        assert hash1 == hash2

        # Different data should produce different hash
        state.data = {"value": 43}
        hash3 = state.compute_hash()
        assert hash1 != hash3

    def test_state_chain_creation(self):
        """Test creating and managing state chains"""
        chain = StateChain(timeline_id="test_chain")

        # Create initial state
        coord = TemporalCoordinate(
            state_hash="",
            timeline_id="test_chain",
            computational_t=0.0,
            physical_t=0.0,
            depth=0,
        )
        state = TemporalState(
            coordinate=coord,
            data=0,
            parent_hash=None,
        )
        state.coordinate.state_hash = state.compute_hash()

        chain.add_state(state)
        assert len(chain) == 1

        # Add second state
        coord2 = TemporalCoordinate(
            state_hash="",
            timeline_id="test_chain",
            computational_t=1.0,
            physical_t=1.0,
            depth=1,
        )
        state2 = TemporalState(
            coordinate=coord2,
            data=1,
            parent_hash=state.coordinate.state_hash,
        )
        state2.coordinate.state_hash = state2.compute_hash()

        chain.add_state(state2)
        assert len(chain) == 2

    def test_merkle_root_computation(self):
        """Test Merkle root computation for state chains"""
        chain = StateChain(timeline_id="test_chain")

        # Add multiple states
        prev_hash = None
        for i in range(5):
            coord = TemporalCoordinate(
                state_hash="",
                timeline_id="test_chain",
                computational_t=float(i),
                physical_t=float(i),
                depth=i,
            )
            state = TemporalState(
                coordinate=coord,
                data=i,
                parent_hash=prev_hash,
            )
            state.coordinate.state_hash = state.compute_hash()
            chain.add_state(state)
            prev_hash = state.coordinate.state_hash

        merkle_root = chain.compute_merkle_root()
        assert len(merkle_root) == 64
        assert merkle_root == chain.merkle_root

    def test_effective_velocity_calculation(self):
        """Test effective velocity between temporal coordinates"""
        coord1 = TemporalCoordinate(
            state_hash="hash1",
            timeline_id="t1",
            computational_t=0.0,
            physical_t=0.0,
            depth=0,
        )

        coord2 = TemporalCoordinate(
            state_hash="hash2",
            timeline_id="t1",
            computational_t=1000.0,  # 1000 seconds simulated
            physical_t=0.001,  # 1 millisecond elapsed
            depth=100,
        )

        velocity = coord1.effective_velocity(coord2)

        # Should be 1000 / 0.001 = 1,000,000 times speed of light
        assert velocity == pytest.approx(1_000_000.0)

    def test_state_serialization(self):
        """Test state serialization and deserialization"""
        coord = TemporalCoordinate(
            state_hash="hash1",
            timeline_id="t1",
            computational_t=10.0,
            physical_t=1.0,
            depth=5,
        )

        state = TemporalState(
            coordinate=coord,
            data={"test": "data"},
            metadata={"key": "value"},
        )

        # Serialize
        serialized = state.serialize()
        assert isinstance(serialized, bytes)

        # Deserialize
        deserialized = TemporalState.deserialize(serialized)
        assert deserialized.coordinate.timeline_id == state.coordinate.timeline_id
        assert deserialized.data == state.data
        assert deserialized.metadata == state.metadata
