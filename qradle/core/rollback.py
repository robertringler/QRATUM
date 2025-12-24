"""
Rollback Manager - Contract-based state rollback capability

Provides the ability to return to any previous verified state.
This is a fatal invariant (Invariant 6) - the system MUST maintain rollback capability.

Version: 1.0.0
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass(frozen=True)
class Checkpoint:
    """Immutable checkpoint representing a verified system state.
    
    Attributes:
        checkpoint_id: Unique identifier
        timestamp: When checkpoint was created
        state_hash: Hash of the system state
        state_data: The actual state data
        metadata: Additional checkpoint metadata
    """
    checkpoint_id: str
    timestamp: str
    state_hash: str
    state_data: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def verify(self) -> bool:
        """Verify checkpoint hash matches state data."""
        serialized = json.dumps(self.state_data, sort_keys=True)
        computed_hash = hashlib.sha256(serialized.encode()).hexdigest()
        return computed_hash == self.state_hash


class RollbackManager:
    """Manages system checkpoints and rollback operations.
    
    The rollback capability is a fatal invariant - it must always be available.
    Checkpoints are immutable and cryptographically verified.
    """
    
    def __init__(self):
        """Initialize rollback manager."""
        self.checkpoints: dict[str, Checkpoint] = {}
        self.checkpoint_order: list[str] = []
        self._current_checkpoint_id: Optional[str] = None
    
    def create_checkpoint(
        self, 
        state_data: dict[str, Any], 
        checkpoint_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> Checkpoint:
        """Create a new checkpoint.
        
        Args:
            state_data: System state to checkpoint
            checkpoint_id: Optional custom checkpoint ID
            metadata: Optional metadata
            
        Returns:
            The created Checkpoint
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Compute state hash
        serialized = json.dumps(state_data, sort_keys=True)
        state_hash = hashlib.sha256(serialized.encode()).hexdigest()
        
        # Generate checkpoint ID if not provided
        if checkpoint_id is None:
            checkpoint_id = f"checkpoint_{state_hash[:16]}_{int(datetime.now(timezone.utc).timestamp())}"
        
        # Create checkpoint
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            timestamp=timestamp,
            state_hash=state_hash,
            state_data=state_data,
            metadata=metadata or {}
        )
        
        # Store checkpoint
        self.checkpoints[checkpoint_id] = checkpoint
        self.checkpoint_order.append(checkpoint_id)
        self._current_checkpoint_id = checkpoint_id
        
        return checkpoint
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get checkpoint by ID.
        
        Args:
            checkpoint_id: ID of checkpoint to retrieve
            
        Returns:
            Checkpoint or None if not found
        """
        return self.checkpoints.get(checkpoint_id)
    
    def has_checkpoint(self, checkpoint_id: str) -> bool:
        """Check if checkpoint exists.
        
        Args:
            checkpoint_id: ID to check
            
        Returns:
            True if checkpoint exists
        """
        return checkpoint_id in self.checkpoints
    
    def rollback_to(self, checkpoint_id: str) -> dict[str, Any]:
        """Rollback to a specific checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to rollback to
            
        Returns:
            The restored state data
            
        Raises:
            ValueError: If checkpoint doesn't exist or is invalid
        """
        checkpoint = self.get_checkpoint(checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        
        # Verify checkpoint integrity
        if not checkpoint.verify():
            raise ValueError(f"Checkpoint integrity check failed: {checkpoint_id}")
        
        # Update current checkpoint
        self._current_checkpoint_id = checkpoint_id
        
        # Return the state data for restoration
        return checkpoint.state_data.copy()
    
    def get_current_checkpoint(self) -> Optional[Checkpoint]:
        """Get the current (most recent) checkpoint.
        
        Returns:
            Current Checkpoint or None
        """
        if self._current_checkpoint_id:
            return self.checkpoints.get(self._current_checkpoint_id)
        return None
    
    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all checkpoints.
        
        Returns:
            List of checkpoint summaries
        """
        return [
            {
                "checkpoint_id": cp_id,
                "timestamp": self.checkpoints[cp_id].timestamp,
                "state_hash": self.checkpoints[cp_id].state_hash,
                "metadata": self.checkpoints[cp_id].metadata,
            }
            for cp_id in self.checkpoint_order
        ]
    
    def verify_all_checkpoints(self) -> list[str]:
        """Verify integrity of all checkpoints.
        
        Returns:
            List of checkpoint IDs that failed verification (empty if all valid)
        """
        failed = []
        for cp_id in self.checkpoint_order:
            checkpoint = self.checkpoints[cp_id]
            if not checkpoint.verify():
                failed.append(cp_id)
        return failed
    
    def prune_checkpoints(self, keep_count: int = 100) -> int:
        """Prune old checkpoints, keeping the most recent ones.
        
        Args:
            keep_count: Number of recent checkpoints to keep
            
        Returns:
            Number of checkpoints removed
        """
        if len(self.checkpoint_order) <= keep_count:
            return 0
        
        # Keep the most recent checkpoints
        to_remove = self.checkpoint_order[:-keep_count]
        
        for cp_id in to_remove:
            del self.checkpoints[cp_id]
        
        self.checkpoint_order = self.checkpoint_order[-keep_count:]
        
        return len(to_remove)
    
    def get_stats(self) -> dict[str, Any]:
        """Get rollback manager statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_checkpoints": len(self.checkpoints),
            "current_checkpoint_id": self._current_checkpoint_id,
            "oldest_checkpoint": self.checkpoint_order[0] if self.checkpoint_order else None,
            "newest_checkpoint": self.checkpoint_order[-1] if self.checkpoint_order else None,
        }
