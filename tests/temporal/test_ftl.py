"""Tests for FTL computation framework."""

import pytest

from qratum.temporal import (EffectiveVelocity, FTLComputation, FTLMechanism,
                             PossibilityOracle)


class TestFTLComputation:
    """Test FTL computation mechanisms"""

    def test_temporal_compression(self):
        """Test temporal compression for FTL speeds"""
        ftl = FTLComputation(peak_flops=2.5e18)

        def evolve(state, dt):
            return state + dt

        initial_state = 0.0
        delta_t = 100.0

        final_state, velocity = ftl.compress_time(
            initial_state=initial_state,
            evolution_fn=evolve,
            delta_t=delta_t,
            compression_target=1e6,
        )

        assert final_state.data == pytest.approx(delta_t, rel=1e-5)
        assert velocity.mechanism == FTLMechanism.TEMPORAL_COMPRESSION
        assert velocity.computational_delta_t == delta_t
        assert velocity.is_ftl()

    def test_inverse_causality(self):
        """Test computing backward from desired outcome"""
        ftl = FTLComputation(peak_flops=2.5e18)

        def inverse_evolve(state, dt):
            return state - dt

        desired_outcome = 100.0
        delta_t = 100.0

        initial_state, velocity = ftl.inverse_causality(
            desired_outcome=desired_outcome,
            inverse_fn=inverse_evolve,
            delta_t=delta_t,
        )

        assert initial_state.data == pytest.approx(0.0, rel=1e-5)
        assert velocity.mechanism == FTLMechanism.INVERSE_CAUSALITY

    def test_parallel_reality_search(self):
        """Test searching across parallel realities"""
        ftl = FTLComputation(peak_flops=2.5e18)

        def evolve(state, dt):
            return state + dt

        def mutate(state):
            # Add some variation
            import random
            return state + random.random()

        def score(state):
            # Simple scoring: closer to 100 is better
            return -abs(state - 100.0)

        branch_point = 0.0
        branch_count = 10

        best_state, best_branch, velocity = ftl.parallel_reality_search(
            branch_point=branch_point,
            branch_count=branch_count,
            evolution_fn=evolve,
            mutation_fn=mutate,
            target_fn=score,
            duration=50.0,
        )

        assert best_state is not None
        assert best_branch is not None
        assert velocity.mechanism == FTLMechanism.PARALLEL_BRANCHING
        assert velocity.is_ftl()

        # Should have explored branch_count timelines
        assert velocity.operations_executed == branch_count

    def test_statistics_collection(self):
        """Test FTL computation statistics"""
        ftl = FTLComputation(peak_flops=2.5e18)

        stats = ftl.get_statistics()

        assert stats['peak_flops'] == 2.5e18
        assert 'operations_per_light_meter' in stats
        assert 'engine_stats' in stats


class TestPossibilityOracle:
    """Test possibility oracle for instant lookups"""

    def test_precompute_all(self):
        """Test precomputing all possibilities"""
        oracle = PossibilityOracle()

        def evolve(state, dt):
            return state * 2

        initial_states = [1.0, 2.0, 3.0, 4.0, 5.0]
        delta_t = 10.0

        space, velocity = oracle.precompute_all(
            initial_states=initial_states,
            evolution_fn=evolve,
            delta_t=delta_t,
            space_id="test_space",
        )

        assert space.size() == len(initial_states)
        assert velocity.mechanism == FTLMechanism.PRECOMPUTATION
        assert velocity.is_ftl()

    def test_instant_query(self):
        """Test instant lookup from precomputed space"""
        oracle = PossibilityOracle()

        def evolve(state, dt):
            return state + 10

        initial_states = [1.0, 2.0, 3.0]

        space, _ = oracle.precompute_all(
            initial_states=initial_states,
            evolution_fn=evolve,
            delta_t=10.0,
            space_id="test_space",
        )

        # Query one of the precomputed results
        state_key = oracle._hash_state(1.0)
        result, velocity = oracle.query(
            state_key=state_key,
            space_id="test_space",
        )

        assert result == 11.0
        # Instant lookup should have very high velocity
        assert velocity.effective_c_multiple > 1e6


class TestEffectiveVelocity:
    """Test effective velocity calculations"""

    def test_velocity_calculation(self):
        """Test velocity calculation from time deltas"""
        velocity = EffectiveVelocity(
            mechanism=FTLMechanism.TEMPORAL_COMPRESSION,
            computational_delta_t=1000.0,
            physical_delta_t=0.001,
        )

        assert velocity.effective_c_multiple == pytest.approx(1_000_000.0)
        assert velocity.is_ftl()

    def test_infinite_velocity(self):
        """Test infinite velocity for instant operations"""
        velocity = EffectiveVelocity(
            mechanism=FTLMechanism.PRECOMPUTATION,
            computational_delta_t=1.0,
            physical_delta_t=0.0,
        )

        assert velocity.effective_c_multiple == float('inf')
        assert velocity.is_ftl()

    def test_sub_light_velocity(self):
        """Test sub-light velocity detection"""
        velocity = EffectiveVelocity(
            mechanism=FTLMechanism.TEMPORAL_COMPRESSION,
            computational_delta_t=0.001,
            physical_delta_t=0.01,
        )

        assert velocity.effective_c_multiple < 1.0
        assert not velocity.is_ftl()

    def test_distance_equivalent(self):
        """Test calculation of light-distance equivalent"""
        from qratum.temporal import SPEED_OF_LIGHT

        computational_delta_t = 10.0  # 10 seconds

        velocity = EffectiveVelocity(
            mechanism=FTLMechanism.TEMPORAL_COMPRESSION,
            computational_delta_t=computational_delta_t,
            physical_delta_t=0.001,
        )

        # Light travels SPEED_OF_LIGHT meters per second
        expected_distance = SPEED_OF_LIGHT * computational_delta_t
        assert velocity.distance_equivalent == pytest.approx(expected_distance)

    def test_velocity_string_representation(self):
        """Test string representation of velocity"""
        velocity = EffectiveVelocity(
            mechanism=FTLMechanism.PARALLEL_BRANCHING,
            computational_delta_t=1000.0,
            physical_delta_t=1.0,
        )

        velocity_str = str(velocity)
        assert "EffectiveVelocity" in velocity_str
        assert "parallel_branching" in velocity_str

        # Test infinite velocity string
        infinite_velocity = EffectiveVelocity(
            mechanism=FTLMechanism.PRECOMPUTATION,
            computational_delta_t=1.0,
            physical_delta_t=0.0,
        )

        infinite_str = str(infinite_velocity)
        assert "∞c" in infinite_str or "inf" in infinite_str.lower()
