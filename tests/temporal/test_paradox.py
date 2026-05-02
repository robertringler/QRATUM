"""Tests for paradox resolution system."""

from qratum.temporal import (CausalityValidator, ParadoxResolver,
                             ParadoxStrategy, ParadoxType, StateChain,
                             TemporalCoordinate, TemporalState)


class TestParadoxResolver:
    """Test paradox resolution strategies"""

    def test_novikov_resolution(self):
        """Test Novikov self-consistency resolution"""
        resolver = ParadoxResolver(strategy=ParadoxStrategy.NOVIKOV)

        # Create a paradoxical chain
        chain = StateChain(timeline_id="test")

        # Add some states
        for i in range(5):
            coord = TemporalCoordinate(
                state_hash="",
                timeline_id="test",
                computational_t=float(i),
                physical_t=float(i),
                depth=i,
            )
            parent_hash = chain.states[-1].coordinate.state_hash if chain.states else None
            state = TemporalState(
                coordinate=coord,
                data=i,
                parent_hash=parent_hash,
            )
            state.coordinate.state_hash = state.compute_hash()
            chain.add_state(state)

        # Create a paradox detection
        from qratum.temporal.paradox import ParadoxDetection
        paradox = ParadoxDetection(
            paradox_type=ParadoxType.GRANDFATHER,
            timeline_id="test",
            state_depth=3,
            description="Test paradox",
            severity=1.0,
        )

        success, resolved_chain = resolver.resolve(paradox, chain)

        assert success is True
        assert resolved_chain is not None
        assert len(resolved_chain) == 3  # Truncated at paradox point

    def test_many_worlds_resolution(self):
        """Test Many-Worlds branching resolution"""
        resolver = ParadoxResolver(strategy=ParadoxStrategy.MANY_WORLDS)

        chain = StateChain(timeline_id="original")

        # Add some states
        for i in range(5):
            coord = TemporalCoordinate(
                state_hash="",
                timeline_id="original",
                computational_t=float(i),
                physical_t=float(i),
                depth=i,
            )
            parent_hash = chain.states[-1].coordinate.state_hash if chain.states else None
            state = TemporalState(
                coordinate=coord,
                data=i,
                parent_hash=parent_hash,
            )
            state.coordinate.state_hash = state.compute_hash()
            chain.add_state(state)

        from qratum.temporal.paradox import ParadoxDetection
        paradox = ParadoxDetection(
            paradox_type=ParadoxType.BOOTSTRAP,
            timeline_id="original",
            state_depth=2,
            description="Bootstrap paradox",
            severity=0.6,
        )

        success, resolved_chain = resolver.resolve(paradox, chain)

        assert success is True
        assert resolved_chain is not None
        # Should create new timeline
        assert resolved_chain.timeline_id != "original"

    def test_chronology_protection_resolution(self):
        """Test Chronology Protection rejection"""
        resolver = ParadoxResolver(strategy=ParadoxStrategy.CHRONOLOGY_PROTECTION)

        chain = StateChain(timeline_id="test")

        from qratum.temporal.paradox import ParadoxDetection
        paradox = ParadoxDetection(
            paradox_type=ParadoxType.CONSISTENCY,
            timeline_id="test",
            state_depth=1,
            description="Consistency violation",
            severity=0.9,
        )

        success, resolved_chain = resolver.resolve(paradox, chain)

        # Chronology protection rejects paradoxical timelines
        assert success is False
        assert resolved_chain is None


class TestCausalityValidator:
    """Test causality validation"""

    def test_valid_chain_validation(self):
        """Test validation of a valid causal chain"""
        validator = CausalityValidator()

        chain = StateChain(timeline_id="test")

        # Create valid chain
        prev_hash = None
        for i in range(5):
            coord = TemporalCoordinate(
                state_hash="",
                timeline_id="test",
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

        is_valid, paradoxes = validator.validate_causal_chain(chain)

        assert is_valid is True
        assert len(paradoxes) == 0

    def test_invalid_chain_detection(self):
        """Test detection of invalid causal chains"""
        validator = CausalityValidator()

        chain = StateChain(timeline_id="test")

        # Create first state
        coord1 = TemporalCoordinate(
            state_hash="",
            timeline_id="test",
            computational_t=0.0,
            physical_t=0.0,
            depth=0,
        )
        state1 = TemporalState(
            coordinate=coord1,
            data=0,
            parent_hash=None,
        )
        state1.coordinate.state_hash = state1.compute_hash()
        chain.add_state(state1)

        # Create second state with invalid parent hash
        coord2 = TemporalCoordinate(
            state_hash="",
            timeline_id="test",
            computational_t=1.0,
            physical_t=1.0,
            depth=1,
        )
        state2 = TemporalState(
            coordinate=coord2,
            data=1,
            parent_hash="wrong_hash",  # Invalid!
        )
        state2.coordinate.state_hash = state2.compute_hash()

        # Manually add to bypass validation
        chain.states.append(state2)

        is_valid, paradoxes = validator.validate_causal_chain(chain)

        assert is_valid is False
        assert len(paradoxes) > 0
        assert paradoxes[0].paradox_type == ParadoxType.CONSISTENCY

    def test_grandfather_paradox_detection(self):
        """Test detection of grandfather paradox"""
        validator = CausalityValidator()

        coord_initial = TemporalCoordinate(
            state_hash="hash1",
            timeline_id="t1",
            computational_t=0.0,
            physical_t=0.0,
            depth=0,
        )
        initial_state = TemporalState(
            coordinate=coord_initial,
            data={"exists": True},
        )

        coord_final = TemporalCoordinate(
            state_hash="hash2",
            timeline_id="t1",
            computational_t=100.0,
            physical_t=1.0,
            depth=10,
        )
        final_state = TemporalState(
            coordinate=coord_final,
            data={"prevents_initial": True},
        )

        def prevents_existence(final_data, initial_data):
            # Final state prevents initial state
            return final_data.get("prevents_initial", False)

        paradox = validator.detect_grandfather_paradox(
            initial_state=initial_state,
            final_state=final_state,
            prevents_existence_fn=prevents_existence,
        )

        assert paradox is not None
        assert paradox.paradox_type == ParadoxType.GRANDFATHER

    def test_causal_loop_detection(self):
        """Test detection of causal loops"""
        validator = CausalityValidator()

        chain = StateChain(timeline_id="test")

        # Create chain with repeated state (loop)
        for i in range(5):
            coord = TemporalCoordinate(
                state_hash="",
                timeline_id="test",
                computational_t=float(i),
                physical_t=float(i),
                depth=i,
            )
            # Create loop by repeating state 2 at position 4
            data = 2 if i == 4 else i
            parent_hash = chain.states[-1].coordinate.state_hash if chain.states else None
            state = TemporalState(
                coordinate=coord,
                data=data,
                parent_hash=parent_hash,
            )
            state.coordinate.state_hash = state.compute_hash()
            chain.add_state(state)

        def state_equal(s1, s2):
            return s1 == s2

        paradoxes = validator.detect_causal_loop(
            chain=chain,
            state_equality_fn=state_equal,
        )

        # Should detect the loop
        assert len(paradoxes) > 0
        assert paradoxes[0].paradox_type == ParadoxType.PREDESTINATION


class TestParadoxIntegration:
    """Test integration of paradox detection and resolution"""

    def test_check_and_resolve(self):
        """Test combined checking and resolution"""
        resolver = ParadoxResolver(strategy=ParadoxStrategy.NOVIKOV)

        # Create valid chain
        chain = StateChain(timeline_id="test")
        prev_hash = None
        for i in range(5):
            coord = TemporalCoordinate(
                state_hash="",
                timeline_id="test",
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

        is_valid, resolved_chain, paradoxes = resolver.check_and_resolve(chain)

        assert is_valid is True
        assert resolved_chain.timeline_id == chain.timeline_id
        assert len(paradoxes) == 0

    def test_statistics(self):
        """Test statistics collection"""
        resolver = ParadoxResolver(strategy=ParadoxStrategy.MANY_WORLDS)

        stats = resolver.get_statistics()

        assert 'strategy' in stats
        assert stats['strategy'] == ParadoxStrategy.MANY_WORLDS
        assert 'total_resolutions' in stats
        assert 'total_detections' in stats
