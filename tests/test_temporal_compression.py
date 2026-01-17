"""Tests for Temporal Compression Experiment Framework.

Validates the deterministic temporal compression experimental
framework including:
- Cellular automata determinism
- Temporal compression metrics
- Cryptographic verification chains
- Reproducibility across runs
"""

from __future__ import annotations

import pytest
import numpy as np

from temporal_compression import (
    CellularAutomaton,
    Rule110,
    Rule30,
    TemporalMetrics,
    CompressionStatistics,
    VerificationChain,
    StateVerifier,
    AuditEntry,
    TemporalCompressionExperiment,
    ExperimentConfig,
)
from temporal_compression.automata import BranchingSimulator, AutomatonState
from temporal_compression.verification import EventType


class TestCellularAutomaton:
    """Tests for cellular automaton implementation."""

    def test_initialization(self):
        """Test basic initialization."""
        automaton = CellularAutomaton(rule=110, size=100, seed=42)
        assert automaton.rule == 110
        assert automaton.size == 100
        assert automaton.seed == 42
        assert automaton.step == 0
        assert len(automaton.cells) == 100

    def test_deterministic_initialization(self):
        """Test that same seed produces identical initial state."""
        automaton1 = CellularAutomaton(rule=110, size=100, seed=42)
        automaton2 = CellularAutomaton(rule=110, size=100, seed=42)

        assert np.array_equal(automaton1.cells, automaton2.cells)
        assert automaton1.get_state_hash() == automaton2.get_state_hash()

    def test_different_seeds_different_states(self):
        """Test that different seeds produce different states."""
        automaton1 = CellularAutomaton(rule=110, size=100, seed=42)
        automaton2 = CellularAutomaton(rule=110, size=100, seed=123)

        assert not np.array_equal(automaton1.cells, automaton2.cells)

    def test_evolve_determinism(self):
        """Test that evolution is deterministic."""
        automaton1 = CellularAutomaton(rule=110, size=100, seed=42)
        automaton2 = CellularAutomaton(rule=110, size=100, seed=42)

        automaton1.evolve(100)
        automaton2.evolve(100)

        assert np.array_equal(automaton1.cells, automaton2.cells)
        assert automaton1.get_state_hash() == automaton2.get_state_hash()

    def test_evolve_step_count(self):
        """Test that step count is tracked correctly."""
        automaton = CellularAutomaton(rule=110, size=50, seed=42)
        assert automaton.step == 0

        automaton.evolve(10)
        assert automaton.step == 10

        automaton.evolve(5)
        assert automaton.step == 15

    def test_reset(self):
        """Test reset returns to initial state."""
        automaton = CellularAutomaton(rule=110, size=100, seed=42)
        initial_hash = automaton.get_state_hash()

        automaton.evolve(100)
        assert automaton.get_state_hash() != initial_hash

        automaton.reset()
        assert automaton.get_state_hash() == initial_hash
        assert automaton.step == 0

    def test_clone(self):
        """Test cloning preserves state."""
        automaton = CellularAutomaton(rule=110, size=100, seed=42)
        automaton.evolve(50)

        clone = automaton.clone()
        assert clone.get_state_hash() == automaton.get_state_hash()
        assert clone.step == automaton.step

        # Clones should evolve independently
        automaton.evolve(10)
        assert clone.step != automaton.step

    def test_rule_110_behavior(self):
        """Test Rule 110 specific behavior."""
        automaton = Rule110(size=100, seed=42)
        assert automaton.rule == 110

        # Evolution should change state
        initial_hash = automaton.get_state_hash()
        automaton.evolve(10)
        assert automaton.get_state_hash() != initial_hash

    def test_rule_30_behavior(self):
        """Test Rule 30 specific behavior."""
        automaton = Rule30(size=100, seed=42)
        assert automaton.rule == 30

        # Evolution should change state
        initial_hash = automaton.get_state_hash()
        automaton.evolve(10)
        assert automaton.get_state_hash() != initial_hash

    def test_custom_initial_state(self):
        """Test initialization with custom state."""
        initial = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
        automaton = CellularAutomaton(
            rule=110, size=5, seed=42, initial_state=initial
        )

        assert np.array_equal(automaton.cells, initial)

    def test_invalid_rule(self):
        """Test that invalid rule raises error."""
        with pytest.raises(ValueError):
            CellularAutomaton(rule=256, size=100, seed=42)

        with pytest.raises(ValueError):
            CellularAutomaton(rule=-1, size=100, seed=42)

    def test_serialization(self):
        """Test serialization round-trip."""
        automaton = CellularAutomaton(rule=110, size=50, seed=42)
        automaton.evolve(25)

        data = automaton.to_dict()
        restored = CellularAutomaton.from_dict(data)

        assert restored.rule == automaton.rule
        assert restored.size == automaton.size
        assert np.array_equal(restored.cells, automaton.cells)


class TestAutomatonState:
    """Tests for automaton state snapshots."""

    def test_state_creation(self):
        """Test state snapshot creation."""
        cells = np.array([1, 0, 1, 0], dtype=np.uint8)
        state = AutomatonState(step=10, cells=cells, timestamp=1234567890.0)

        assert state.step == 10
        assert np.array_equal(state.cells, cells)
        assert state.timestamp == 1234567890.0
        assert len(state.state_hash) == 64  # SHA3-256 hex

    def test_hash_determinism(self):
        """Test that state hash is deterministic."""
        cells = np.array([1, 0, 1, 0], dtype=np.uint8)

        state1 = AutomatonState(step=10, cells=cells, timestamp=1234567890.0)
        state2 = AutomatonState(step=10, cells=cells.copy(), timestamp=1234567890.0)

        assert state1.state_hash == state2.state_hash

    def test_different_steps_different_hash(self):
        """Test that different steps produce different hashes."""
        cells = np.array([1, 0, 1, 0], dtype=np.uint8)

        state1 = AutomatonState(step=10, cells=cells, timestamp=1234567890.0)
        state2 = AutomatonState(step=11, cells=cells, timestamp=1234567890.0)

        assert state1.state_hash != state2.state_hash


class TestBranchingSimulator:
    """Tests for branching simulation."""

    def test_branch_creation(self):
        """Test creating branches."""
        base = CellularAutomaton(rule=110, size=50, seed=42)
        base.evolve(10)

        branching = BranchingSimulator(base)
        branch = branching.create_branch("test_branch")

        assert branch.get_state_hash() == base.get_state_hash()
        assert "test_branch" in branching.branches

    def test_branch_with_perturbation(self):
        """Test branch with cell perturbation."""
        base = CellularAutomaton(rule=110, size=50, seed=42)

        branching = BranchingSimulator(base)
        branch = branching.create_branch(
            "perturbed_branch",
            perturbation_indices=[25],
        )

        # States should differ at perturbed position
        assert base.cells[25] != branch.cells[25]

    def test_branch_evolution_independence(self):
        """Test that branches evolve independently."""
        base = CellularAutomaton(rule=110, size=50, seed=42)
        base.evolve(10)

        branching = BranchingSimulator(base)
        branch = branching.create_branch("test_branch")

        base.evolve(10)
        # Branch hasn't evolved yet
        assert branch.step == base.step - 10


class TestTemporalMetrics:
    """Tests for temporal metrics calculation."""

    def test_measurement_cycle(self):
        """Test start/end measurement cycle."""
        metrics = TemporalMetrics()

        metrics.start_measurement()
        # Simulate some work
        import time
        time.sleep(0.01)
        measurement = metrics.end_measurement(steps=1000, simulated_time=1000.0)

        assert measurement.steps_simulated == 1000
        assert measurement.simulated_duration == 1000.0
        assert measurement.wall_clock_duration > 0
        assert measurement.compression_ratio > 0

    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation."""
        metrics = TemporalMetrics()

        metrics.start_measurement()
        import time
        time.sleep(0.1)
        measurement = metrics.end_measurement(steps=1000, simulated_time=1000.0)

        # Should compress significantly (1000 time units in ~0.1s)
        assert measurement.compression_ratio > 100

    def test_multiple_measurements(self):
        """Test aggregation of multiple measurements."""
        metrics = TemporalMetrics()

        for _ in range(3):
            metrics.start_measurement()
            import time
            time.sleep(0.01)
            metrics.end_measurement(steps=100, simulated_time=100.0)

        assert len(metrics.statistics.measurements) == 3
        assert metrics.statistics.total_steps == 300
        assert metrics.get_compression_ratio() > 0

    def test_reproducibility_verification(self):
        """Test reproducibility verification."""
        metrics = TemporalMetrics()

        # Same hashes = reproducible
        assert metrics.verify_reproducibility(["abc", "abc", "abc"])

        # Different hashes = not reproducible
        assert not metrics.verify_reproducibility(["abc", "abc", "def"])

    def test_overhead_tracking(self):
        """Test verification overhead tracking."""
        metrics = TemporalMetrics()

        metrics.record_hash_time(0.001)
        metrics.record_merkle_time(0.002)

        assert metrics.verification_overhead.hash_computation_time == 0.001
        assert metrics.verification_overhead.merkle_update_time == 0.002
        assert metrics.verification_overhead.total_overhead_time == 0.003


class TestCompressionStatistics:
    """Tests for compression statistics aggregation."""

    def test_statistics_update(self):
        """Test statistics update on measurement addition."""
        from temporal_compression.metrics import TimingMeasurement

        stats = CompressionStatistics()

        m1 = TimingMeasurement(
            start_time=0, end_time=0.1, steps_simulated=1000, simulated_duration=1000.0
        )
        m2 = TimingMeasurement(
            start_time=0.1, end_time=0.2, steps_simulated=1000, simulated_duration=1000.0
        )

        stats.add_measurement(m1)
        stats.add_measurement(m2)

        assert len(stats.measurements) == 2
        assert stats.total_steps == 2000
        assert stats.mean_compression_ratio > 0


class TestVerificationChain:
    """Tests for cryptographic verification chain."""

    def test_genesis_creation(self):
        """Test genesis block creation."""
        chain = VerificationChain()

        assert len(chain.entries) == 1
        assert chain.entries[0].event_type == EventType.GENESIS
        assert chain.entries[0].previous_hash == "0" * 64

    def test_add_event(self):
        """Test adding events to chain."""
        chain = VerificationChain()

        entry = chain.add_event(
            EventType.STATE_TRANSITION,
            step=1,
            state_hash="a" * 64,
            data={"test": "data"},
        )

        assert len(chain.entries) == 2
        assert entry.event_type == EventType.STATE_TRANSITION
        assert entry.previous_hash == chain.entries[0].entry_hash

    def test_chain_integrity(self):
        """Test chain integrity verification."""
        chain = VerificationChain()

        for i in range(5):
            chain.add_state_transition(
                step=i,
                state_hash=f"{i:064x}",
            )

        assert chain.verify_integrity()

    def test_chain_proof(self):
        """Test chain proof generation."""
        chain = VerificationChain()
        chain.add_state_transition(step=1, state_hash="a" * 64)

        proof = chain.get_chain_proof()
        assert len(proof) == 64  # SHA3-256 hex

    def test_tamper_detection(self):
        """Test that tampering is detected."""
        chain = VerificationChain()
        chain.add_state_transition(step=1, state_hash="a" * 64)

        # Tamper with an entry
        chain.entries[1].data["tampered"] = True
        # Hash no longer matches
        assert not chain.entries[1].verify_hash()

    def test_export_for_verification(self):
        """Test export for independent verification."""
        chain = VerificationChain()
        chain.add_state_transition(step=1, state_hash="a" * 64)

        export = chain.export_for_verification()

        assert "version" in export
        assert "algorithm" in export
        assert "chain_proof" in export
        assert "entries" in export


class TestStateVerifier:
    """Tests for state verification."""

    def test_compute_state_hash(self):
        """Test state hash computation."""
        verifier = StateVerifier()

        state = np.array([1, 0, 1, 0], dtype=np.uint8)
        hash1 = verifier.compute_state_hash(state)
        hash2 = verifier.compute_state_hash(state.copy())

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_verify_identical_states(self):
        """Test verification of identical states."""
        verifier = StateVerifier()

        state = np.array([1, 0, 1, 0], dtype=np.uint8)
        is_identical, details = verifier.verify_identical_states(state, state.copy())

        assert is_identical
        assert details["reason"] == "identical"

    def test_verify_different_states(self):
        """Test verification of different states."""
        verifier = StateVerifier()

        state1 = np.array([1, 0, 1, 0], dtype=np.uint8)
        state2 = np.array([1, 1, 1, 0], dtype=np.uint8)

        is_identical, details = verifier.verify_identical_states(state1, state2)

        assert not is_identical
        assert details["reason"] == "content_mismatch"

    def test_verify_chain(self):
        """Test verification chain validation."""
        verifier = StateVerifier()
        chain = VerificationChain()
        chain.add_state_transition(step=1, state_hash="a" * 64)

        is_valid, details = verifier.verify_chain(chain)

        assert is_valid
        assert details["is_valid"]


class TestAuditEntry:
    """Tests for audit entry."""

    def test_entry_hash_determinism(self):
        """Test that entry hash is deterministic."""
        entry1 = AuditEntry(
            event_type=EventType.STATE_TRANSITION,
            step=10,
            timestamp=1234567890.0,
            state_hash="a" * 64,
            data={"key": "value"},
            previous_hash="0" * 64,
        )

        entry2 = AuditEntry(
            event_type=EventType.STATE_TRANSITION,
            step=10,
            timestamp=1234567890.0,
            state_hash="a" * 64,
            data={"key": "value"},
            previous_hash="0" * 64,
        )

        assert entry1.entry_hash == entry2.entry_hash

    def test_verify_hash(self):
        """Test hash verification."""
        entry = AuditEntry(
            event_type=EventType.STATE_TRANSITION,
            step=10,
            timestamp=1234567890.0,
            state_hash="a" * 64,
            data={},
            previous_hash="0" * 64,
        )

        assert entry.verify_hash()

    def test_serialization(self):
        """Test entry serialization."""
        entry = AuditEntry(
            event_type=EventType.CHECKPOINT,
            step=100,
            timestamp=1234567890.0,
            state_hash="b" * 64,
            data={"checkpoint_id": "test"},
            previous_hash="a" * 64,
        )

        data = entry.to_dict()
        restored = AuditEntry.from_dict(data)

        assert restored.event_type == entry.event_type
        assert restored.step == entry.step
        assert restored.entry_hash == entry.entry_hash


class TestExperimentConfig:
    """Tests for experiment configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExperimentConfig()

        assert config.system_rule == 110
        assert config.system_size == 200
        assert config.seed == 42
        assert config.simulated_steps == 10000
        assert config.num_repetitions == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExperimentConfig(
            system_rule=30,
            system_size=500,
            simulated_steps=50000,
        )

        assert config.system_rule == 30
        assert config.system_size == 500
        assert config.simulated_steps == 50000

    def test_serialization(self):
        """Test config serialization."""
        config = ExperimentConfig(system_size=100)
        data = config.to_dict()
        restored = ExperimentConfig.from_dict(data)

        assert restored.system_size == config.system_size


class TestTemporalCompressionExperiment:
    """Tests for main experiment class."""

    def test_quick_experiment(self):
        """Test running a quick experiment."""
        config = ExperimentConfig(
            system_size=20,
            simulated_steps=100,
            num_repetitions=2,
            checkpoint_interval=50,
        )

        experiment = TemporalCompressionExperiment(config, verbose=False)
        result = experiment.run()

        assert result.is_reproducible
        assert result.compression_ratio > 0
        assert result.verification_chain.verify_integrity()
        assert len(result.final_state_hashes) == 2

    def test_reproducibility_verification(self):
        """Test that experiment verifies reproducibility."""
        config = ExperimentConfig(
            system_size=30,
            simulated_steps=200,
            num_repetitions=3,
            checkpoint_interval=100,
        )

        experiment = TemporalCompressionExperiment(config, verbose=False)
        result = experiment.run()

        # All runs should produce same final hash
        assert len(set(result.final_state_hashes)) == 1
        assert result.is_reproducible

    def test_hypothesis_evaluation(self):
        """Test hypothesis evaluation."""
        config = ExperimentConfig(
            system_size=50,
            simulated_steps=500,
            num_repetitions=2,
            checkpoint_interval=100,
        )

        experiment = TemporalCompressionExperiment(config, verbose=False)
        result = experiment.run()

        assert "conclusion" in result.hypothesis_evaluation
        assert "interpretation" in result.hypothesis_evaluation
        assert "compression_ratio" in result.hypothesis_evaluation

    def test_report_generation(self):
        """Test report generation."""
        config = ExperimentConfig(
            system_size=20,
            simulated_steps=100,
            num_repetitions=2,
            checkpoint_interval=50,
        )

        experiment = TemporalCompressionExperiment(config, verbose=False)
        result = experiment.run()
        report = experiment.generate_report(result)

        assert "# Temporal Compression Experiment Report" in report
        assert "Compression Ratio" in report
        assert "Hypothesis" in report


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_full_workflow(self):
        """Test complete experiment workflow."""
        from temporal_compression.experiment import run_quick_validation

        # Quick validation should pass
        assert run_quick_validation()

    def test_determinism_across_experiment_runs(self):
        """Test that multiple experiment runs produce same results."""
        config = ExperimentConfig(
            system_size=30,
            simulated_steps=100,
            num_repetitions=2,
            checkpoint_interval=50,
        )

        exp1 = TemporalCompressionExperiment(config, verbose=False)
        result1 = exp1.run()

        exp2 = TemporalCompressionExperiment(config, verbose=False)
        result2 = exp2.run()

        # Final hashes should match across experiments
        assert result1.final_state_hashes == result2.final_state_hashes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
