"""Tests for timeline management."""

import pytest

from qratum.temporal import (
    StateChain,
    TemporalCoordinate,
    TemporalState,
    Timeline,
    TimelineManager,
)


class TestTimeline:
    """Test timeline management"""

    def test_timeline_creation(self):
        """Test creating a timeline"""
        chain = StateChain(timeline_id="test_timeline")
        timeline = Timeline(
            timeline_id="test_timeline",
            chain=chain,
        )

        assert timeline.timeline_id == "test_timeline"
        assert timeline.is_active is True
        assert timeline.length() == 0

    def test_timeline_add_state(self):
        """Test adding states to timeline"""
        chain = StateChain(timeline_id="test")
        timeline = Timeline(timeline_id="test", chain=chain)

        coord = TemporalCoordinate(
            state_hash="",
            timeline_id="test",
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

        timeline.add_state(state)

        assert timeline.length() == 1
        assert timeline.get_latest_state() == state

    def test_timeline_branch_recording(self):
        """Test recording branch points"""
        chain = StateChain(timeline_id="parent")
        timeline = Timeline(timeline_id="parent", chain=chain)

        branch = timeline.branch(
            branch_point_depth=5,
            child_timeline_ids=["child1", "child2"],
            reason="test_branch",
        )

        assert len(timeline.child_branches) == 1
        assert branch.parent_timeline_id == "parent"
        assert len(branch.child_timeline_ids) == 2


class TestTimelineManager:
    """Test timeline manager"""

    def test_manager_creation(self):
        """Test creating timeline manager"""
        manager = TimelineManager()

        assert len(manager.timelines) == 0
        assert manager.creation_count == 0

    def test_create_timeline(self):
        """Test creating timelines"""
        manager = TimelineManager()

        timeline = manager.create_timeline(timeline_id="test1")

        assert timeline.timeline_id == "test1"
        assert manager.creation_count == 1
        assert len(manager.timelines) == 1

    def test_get_timeline(self):
        """Test retrieving timelines"""
        manager = TimelineManager()

        manager.create_timeline(timeline_id="test1")
        timeline = manager.get_timeline("test1")

        assert timeline is not None
        assert timeline.timeline_id == "test1"

        # Non-existent timeline
        assert manager.get_timeline("nonexistent") is None

    def test_branch_timeline(self):
        """Test branching timelines"""
        manager = TimelineManager()

        # Create parent timeline with some states
        parent = manager.create_timeline(timeline_id="parent")

        # Add states to parent
        for i in range(10):
            coord = TemporalCoordinate(
                state_hash="",
                timeline_id="parent",
                computational_t=float(i),
                physical_t=float(i),
                depth=i,
            )
            parent_hash = (
                parent.chain.states[-1].coordinate.state_hash if parent.chain.states else None
            )
            state = TemporalState(
                coordinate=coord,
                data=i,
                parent_hash=parent_hash,
            )
            state.coordinate.state_hash = state.compute_hash()
            parent.add_state(state)

        # Branch at depth 5
        children = manager.branch_timeline(
            parent_id="parent",
            branch_names=["branch_a", "branch_b"],
            branch_point_depth=5,
            reason="test_branching",
        )

        assert len(children) == 2
        assert "branch_a" in children
        assert "branch_b" in children

        # Each child should have states up to branch point
        for child in children.values():
            assert child.length() == 6  # 0 through 5 inclusive

    def test_merge_timelines(self):
        """Test merging timelines"""
        manager = TimelineManager()

        # Create two timelines
        t1 = manager.create_timeline(timeline_id="t1")
        t2 = manager.create_timeline(timeline_id="t2")

        # Add a final state to each fresh timeline. As the first state in an
        # empty timeline it must be an initial state (depth 0, no parent_hash).
        coord1 = TemporalCoordinate(
            state_hash="",
            timeline_id="t1",
            computational_t=100.0,
            physical_t=1.0,
            depth=0,
        )
        state1 = TemporalState(coordinate=coord1, data=10.0)
        state1.coordinate.state_hash = state1.compute_hash()
        t1.add_state(state1)

        coord2 = TemporalCoordinate(
            state_hash="",
            timeline_id="t2",
            computational_t=100.0,
            physical_t=1.0,
            depth=0,
        )
        state2 = TemporalState(coordinate=coord2, data=20.0)
        state2.coordinate.state_hash = state2.compute_hash()
        t2.add_state(state2)

        # Merge function: average
        def merge_fn(states):
            return sum(s.data for s in states) / len(states)

        merged = manager.merge_timelines(
            timeline_ids=["t1", "t2"],
            merge_fn=merge_fn,
        )

        assert merged is not None
        assert merged.length() == 1
        assert merged.get_latest_state().data == pytest.approx(15.0)

    def test_find_optimal_timeline(self):
        """Test finding optimal timeline"""
        manager = TimelineManager()

        # Create multiple timelines with different final values
        for i in range(5):
            timeline = manager.create_timeline(timeline_id=f"t{i}")

            coord = TemporalCoordinate(
                state_hash="",
                timeline_id=f"t{i}",
                computational_t=100.0,
                physical_t=1.0,
                depth=0,
            )
            state = TemporalState(coordinate=coord, data=float(i * 10))
            state.coordinate.state_hash = state.compute_hash()
            timeline.add_state(state)

        # Find timeline with maximum final value
        def objective(timeline):
            latest = timeline.get_latest_state()
            return latest.data if latest else float("-inf")

        best_timeline, best_score = manager.find_optimal_timeline(objective)

        assert best_timeline is not None
        assert best_score == 40.0  # Timeline t4 has final value 40
        assert best_timeline.timeline_id == "t4"

    def test_get_timeline_graph(self):
        """Test getting timeline relationship graph"""
        manager = TimelineManager()

        # Create parent with children
        parent = manager.create_timeline(timeline_id="parent")

        # Add some states
        for i in range(5):
            coord = TemporalCoordinate(
                state_hash="",
                timeline_id="parent",
                computational_t=float(i),
                physical_t=float(i),
                depth=i,
            )
            parent_hash = (
                parent.chain.states[-1].coordinate.state_hash if parent.chain.states else None
            )
            state = TemporalState(
                coordinate=coord,
                data=i,
                parent_hash=parent_hash,
            )
            state.coordinate.state_hash = state.compute_hash()
            parent.add_state(state)

        # Branch
        manager.branch_timeline(
            parent_id="parent",
            branch_names=["child1", "child2"],
            branch_point_depth=2,
        )

        graph = manager.get_timeline_graph()

        assert "parent" in graph
        assert len(graph["parent"]) == 2

    def test_get_all_leaves(self):
        """Test getting leaf timelines"""
        manager = TimelineManager()

        # Create parent
        parent = manager.create_timeline(timeline_id="parent")
        for i in range(5):
            coord = TemporalCoordinate(
                state_hash="",
                timeline_id="parent",
                computational_t=float(i),
                physical_t=float(i),
                depth=i,
            )
            parent_hash = (
                parent.chain.states[-1].coordinate.state_hash if parent.chain.states else None
            )
            state = TemporalState(
                coordinate=coord,
                data=i,
                parent_hash=parent_hash,
            )
            state.coordinate.state_hash = state.compute_hash()
            parent.add_state(state)

        # Branch to create leaves
        manager.branch_timeline(
            parent_id="parent",
            branch_names=["leaf1", "leaf2"],
            branch_point_depth=2,
        )

        leaves = manager.get_all_leaves()

        # Should have 2 leaves (the children)
        assert len(leaves) >= 2

    def test_prune_timelines(self):
        """Test pruning timelines"""
        manager = TimelineManager()

        # Create several timelines
        for i in range(10):
            timeline = manager.create_timeline(timeline_id=f"t{i}")
            timeline.metadata["score"] = i

        # Keep only timelines with score >= 5
        def keep_fn(timeline):
            return timeline.metadata.get("score", 0) >= 5

        pruned_count = manager.prune_timelines(keep_fn)

        assert pruned_count == 5
        assert len(manager.timelines) == 5

    def test_statistics(self):
        """Test statistics collection"""
        manager = TimelineManager()

        # Create some timelines
        for i in range(3):
            manager.create_timeline(timeline_id=f"t{i}")

        stats = manager.get_statistics()

        assert stats["total_timelines"] == 3
        assert stats["total_created"] == 3
        assert "active_timelines" in stats
