"""
Tests for QRADLE Rollback Manager

Tests checkpoint creation and rollback functionality.
"""

import pytest

from qradle.core.rollback import RollbackManager


class TestRollbackManager:
    """Test suite for rollback manager."""

    def test_create_checkpoint(self):
        """Test checkpoint creation."""
        manager = RollbackManager()

        state = {"counter": 1, "data": "test"}
        checkpoint = manager.create_checkpoint(state)

        assert checkpoint.checkpoint_id is not None
        assert checkpoint.state_data == state
        assert checkpoint.verify()

    def test_get_checkpoint(self):
        """Test retrieving checkpoint."""
        manager = RollbackManager()

        state = {"value": 42}
        checkpoint = manager.create_checkpoint(state, checkpoint_id="test_checkpoint")

        retrieved = manager.get_checkpoint("test_checkpoint")
        assert retrieved is not None
        assert retrieved.checkpoint_id == "test_checkpoint"
        assert retrieved.state_data == state

    def test_has_checkpoint(self):
        """Test checking checkpoint existence."""
        manager = RollbackManager()

        manager.create_checkpoint({"data": "test"}, checkpoint_id="cp1")

        assert manager.has_checkpoint("cp1")
        assert not manager.has_checkpoint("nonexistent")

    def test_rollback_to_checkpoint(self):
        """Test rolling back to a checkpoint."""
        manager = RollbackManager()

        # Create initial state
        state1 = {"counter": 1}
        cp1 = manager.create_checkpoint(state1, checkpoint_id="cp1")

        # Create second state
        state2 = {"counter": 2}
        cp2 = manager.create_checkpoint(state2, checkpoint_id="cp2")

        # Rollback to first checkpoint
        restored_state = manager.rollback_to("cp1")
        assert restored_state == state1

    def test_rollback_to_invalid_checkpoint(self):
        """Test rollback to nonexistent checkpoint."""
        manager = RollbackManager()

        with pytest.raises(ValueError):
            manager.rollback_to("nonexistent")

    def test_current_checkpoint(self):
        """Test getting current checkpoint."""
        manager = RollbackManager()

        state1 = {"value": 1}
        cp1 = manager.create_checkpoint(state1, checkpoint_id="cp1")

        current = manager.get_current_checkpoint()
        assert current is not None
        assert current.checkpoint_id == "cp1"

        state2 = {"value": 2}
        cp2 = manager.create_checkpoint(state2, checkpoint_id="cp2")

        current = manager.get_current_checkpoint()
        assert current.checkpoint_id == "cp2"

    def test_list_checkpoints(self):
        """Test listing all checkpoints."""
        manager = RollbackManager()

        manager.create_checkpoint({"v": 1}, checkpoint_id="cp1")
        manager.create_checkpoint({"v": 2}, checkpoint_id="cp2")
        manager.create_checkpoint({"v": 3}, checkpoint_id="cp3")

        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 3
        assert checkpoints[0]["checkpoint_id"] == "cp1"
        assert checkpoints[2]["checkpoint_id"] == "cp3"

    def test_verify_all_checkpoints(self):
        """Test verifying all checkpoints."""
        manager = RollbackManager()

        manager.create_checkpoint({"data": "test1"})
        manager.create_checkpoint({"data": "test2"})

        failed = manager.verify_all_checkpoints()
        assert len(failed) == 0

    def test_prune_checkpoints(self):
        """Test pruning old checkpoints."""
        manager = RollbackManager()

        # Create 10 checkpoints
        for i in range(10):
            manager.create_checkpoint({"counter": i}, checkpoint_id=f"cp{i}")

        # Keep only 5 most recent
        removed = manager.prune_checkpoints(keep_count=5)
        assert removed == 5
        assert len(manager.checkpoints) == 5

    def test_checkpoint_metadata(self):
        """Test checkpoint with metadata."""
        manager = RollbackManager()

        state = {"value": 123}
        metadata = {"author": "test", "reason": "testing"}
        checkpoint = manager.create_checkpoint(state, metadata=metadata)

        assert checkpoint.metadata == metadata

    def test_rollback_manager_stats(self):
        """Test getting rollback manager statistics."""
        manager = RollbackManager()

        manager.create_checkpoint({"v": 1}, checkpoint_id="cp1")
        manager.create_checkpoint({"v": 2}, checkpoint_id="cp2")

        stats = manager.get_stats()
        assert stats["total_checkpoints"] == 2
        assert stats["current_checkpoint_id"] == "cp2"
        assert stats["oldest_checkpoint"] == "cp1"
        assert stats["newest_checkpoint"] == "cp2"
