"""
Rollback system for QRADLE.

Provides checkpoint creation and rollback capability to return
to any previous verified state.
"""

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Checkpoint:
    """Snapshot of system state at a specific point in time."""

    checkpoint_id: str
    timestamp: float = field(default_factory=lambda: time.time())
    state: Dict[str, Any] = field(default_factory=dict)
    merkle_proof: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "timestamp": self.timestamp,
            "state": self.state,
            "merkle_proof": self.merkle_proof,
            "description": self.description,
        }


class RollbackManager:
    """Manages checkpoints and rollback operations.

    Enables returning to any previous verified state while
    maintaining audit trail of all operations.
    """

    def __init__(self, max_checkpoints: int = 100):
        """Initialize rollback manager.

        Args:
            max_checkpoints: Maximum number of checkpoints to retain
        """
        self.max_checkpoints = max_checkpoints
        self.checkpoints: List[Checkpoint] = []

    def create_checkpoint(
        self, checkpoint_id: str, state: Dict[str, Any], merkle_proof: str, description: str = ""
    ) -> Checkpoint:
        """Create a new checkpoint.

        Args:
            checkpoint_id: Unique identifier for checkpoint
            state: System state to save
            merkle_proof: Cryptographic proof from Merkle chain
            description: Human-readable description

        Returns:
            Created checkpoint
        """
        # Deep copy state to ensure immutability
        state_copy = copy.deepcopy(state)

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            state=state_copy,
            merkle_proof=merkle_proof,
            description=description,
        )

        self.checkpoints.append(checkpoint)

        # Maintain max checkpoints limit
        if len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints.pop(0)

        return checkpoint

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Retrieve a checkpoint by ID.

        Args:
            checkpoint_id: ID of checkpoint to retrieve

        Returns:
            Checkpoint if found, None otherwise
        """
        for checkpoint in self.checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint
        return None

    def list_checkpoints(self) -> List[Checkpoint]:
        """Get list of all available checkpoints.

        Returns:
            List of checkpoints sorted by timestamp
        """
        return sorted(self.checkpoints, key=lambda c: c.timestamp)

    def rollback_to(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Rollback to a specific checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to rollback to

        Returns:
            State from checkpoint, or None if not found
        """
        checkpoint = self.get_checkpoint(checkpoint_id)
        if checkpoint:
            return copy.deepcopy(checkpoint.state)
        return None

    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        """Get the most recent checkpoint.

        Returns:
            Latest checkpoint, or None if no checkpoints exist
        """
        if self.checkpoints:
            return self.checkpoints[-1]
        return None
