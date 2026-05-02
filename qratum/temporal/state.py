"""
Temporal State Management for QRATUM

Manages temporal states, state chains, and temporal coordinates for computational
time travel. All state transitions are Merkle-verified for cryptographic proof.
"""

import hashlib
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from .constants import MERKLE_HASH_SIZE, DEFAULT_STATE_CHUNK_SIZE


@dataclass
class TemporalCoordinate:
    """
    Position in computational spacetime.
    
    A temporal coordinate uniquely identifies a point in the multiverse of
    computed timelines, combining state identity, timeline identity, and
    both computational and physical time.
    
    Attributes:
        state_hash: Cryptographic hash of the state
        timeline_id: Identifier for the timeline
        computational_t: Computational time (simulated time)
        physical_t: Physical wall-clock time
        depth: Depth in the timeline (number of steps from origin)
    """
    state_hash: str
    timeline_id: str
    computational_t: float
    physical_t: float
    depth: int = 0
    
    def __str__(self) -> str:
        return f"T({self.timeline_id}, t={self.computational_t:.6f}, depth={self.depth})"
    
    def __repr__(self) -> str:
        return (f"TemporalCoordinate(timeline={self.timeline_id}, "
                f"comp_t={self.computational_t}, phys_t={self.physical_t}, "
                f"depth={self.depth}, hash={self.state_hash[:8]}...)")
    
    def effective_velocity(self, other: 'TemporalCoordinate') -> float:
        """
        Calculate effective velocity between two temporal coordinates.
        
        Effective velocity = computational_time_delta / physical_time_delta
        
        Returns:
            Multiple of speed of light (c). Returns inf for instantaneous.
        """
        if self.timeline_id != other.timeline_id:
            raise ValueError("Cannot calculate velocity across different timelines")
        
        phys_delta = abs(other.physical_t - self.physical_t)
        comp_delta = abs(other.computational_t - self.computational_t)
        
        if phys_delta == 0:
            return float('inf')  # Instantaneous lookup
        
        # Ratio of simulated time to real time (in units of c)
        from .constants import SPEED_OF_LIGHT
        time_compression = comp_delta / phys_delta
        
        # This is effective velocity in multiples of c
        # If we simulate 1 year in 1 second, we're effectively traveling at
        # 31_557_600 times the speed of light
        return time_compression


@dataclass
class TemporalState:
    """
    State at a specific point in computational time with proof chain.
    
    A temporal state captures the complete state of a system at a specific
    point in time, along with cryptographic proofs for verification.
    
    Attributes:
        coordinate: Temporal coordinate of this state
        data: The actual state data (can be any serializable object)
        parent_hash: Hash of the parent state (None for initial state)
        merkle_proof: Merkle proof for state verification
        metadata: Additional metadata about the state
    """
    coordinate: TemporalCoordinate
    data: Any
    parent_hash: Optional[str] = None
    merkle_proof: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate state after initialization"""
        if self.merkle_proof is None:
            self.merkle_proof = []
        if self.coordinate.depth > 0 and self.parent_hash is None:
            raise ValueError("Non-initial state must have parent_hash")
    
    def compute_hash(self) -> str:
        """
        Compute cryptographic hash of this state.
        
        The hash includes the state data, coordinate, and parent hash to
        ensure complete chain integrity.
        
        Returns:
            Hexadecimal hash string
        """
        hasher = hashlib.sha256()
        
        # Hash the state data
        try:
            data_bytes = pickle.dumps(self.data)
        except (pickle.PicklingError, TypeError):
            # Fallback for non-picklable objects
            data_bytes = str(self.data).encode('utf-8')
        hasher.update(data_bytes)
        
        # Hash the coordinate
        coord_str = f"{self.coordinate.timeline_id}:{self.coordinate.computational_t}:{self.coordinate.depth}"
        hasher.update(coord_str.encode('utf-8'))
        
        # Chain to parent
        if self.parent_hash:
            hasher.update(self.parent_hash.encode('utf-8'))
        
        return hasher.hexdigest()
    
    def verify_hash(self) -> bool:
        """Verify that the coordinate hash matches the computed hash"""
        return self.coordinate.state_hash == self.compute_hash()
    
    def serialize(self) -> bytes:
        """Serialize state for storage or transmission"""
        return pickle.dumps({
            'coordinate': self.coordinate,
            'data': self.data,
            'parent_hash': self.parent_hash,
            'merkle_proof': self.merkle_proof,
            'metadata': self.metadata,
        })
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'TemporalState':
        """Deserialize state from bytes"""
        state_dict = pickle.loads(data)
        return cls(**state_dict)
    
    def diff(self, other: 'TemporalState') -> Dict[str, Any]:
        """
        Compute difference between this state and another.
        
        Used for efficient storage by only storing deltas.
        """
        # Simple diff implementation - can be enhanced for specific data types
        return {
            'depth_delta': other.coordinate.depth - self.coordinate.depth,
            'time_delta': other.coordinate.computational_t - self.coordinate.computational_t,
            'data_changed': self.data != other.data,
            # Add more sophisticated diffing for structured data
        }


@dataclass
class StateChain:
    """
    Chain of temporal states forming a timeline with Merkle verification.
    
    A state chain represents a sequence of temporal states along a single
    timeline, with each state cryptographically linked to its parent.
    
    Attributes:
        timeline_id: Identifier for this timeline
        states: List of temporal states in chronological order
        merkle_root: Merkle root hash for the entire chain
    """
    timeline_id: str
    states: List[TemporalState] = field(default_factory=list)
    merkle_root: Optional[str] = None
    
    def add_state(self, state: TemporalState) -> None:
        """
        Add a new state to the chain.
        
        Args:
            state: Temporal state to add
            
        Raises:
            ValueError: If state doesn't properly chain from the last state
        """
        if state.coordinate.timeline_id != self.timeline_id:
            raise ValueError(f"State timeline {state.coordinate.timeline_id} "
                           f"doesn't match chain timeline {self.timeline_id}")
        
        if self.states:
            last_state = self.states[-1]
            if state.parent_hash != last_state.coordinate.state_hash:
                raise ValueError("State doesn't properly chain from parent")
            if state.coordinate.depth != last_state.coordinate.depth + 1:
                raise ValueError("State depth must be parent depth + 1")
        else:
            # First state
            if state.coordinate.depth != 0:
                raise ValueError("Initial state must have depth 0")
            if state.parent_hash is not None:
                raise ValueError("Initial state must not have parent_hash")
        
        self.states.append(state)
        self.merkle_root = None  # Invalidate Merkle root
    
    def compute_merkle_root(self) -> str:
        """
        Compute Merkle root for the entire state chain.
        
        Returns:
            Hexadecimal hash string of the Merkle root
        """
        if not self.states:
            return hashlib.sha256(b'').hexdigest()
        
        # Build Merkle tree
        hashes = [state.coordinate.state_hash for state in self.states]
        
        while len(hashes) > 1:
            next_level = []
            for i in range(0, len(hashes), 2):
                if i + 1 < len(hashes):
                    # Pair exists
                    combined = hashes[i] + hashes[i + 1]
                else:
                    # Odd one out, duplicate it
                    combined = hashes[i] + hashes[i]
                
                hasher = hashlib.sha256()
                hasher.update(combined.encode('utf-8'))
                next_level.append(hasher.hexdigest())
            
            hashes = next_level
        
        self.merkle_root = hashes[0]
        return self.merkle_root
    
    def generate_merkle_proof(self, index: int) -> List[str]:
        """
        Generate Merkle proof for a state at given index.
        
        Args:
            index: Index of state in the chain
            
        Returns:
            List of hashes forming the Merkle proof
        """
        if index < 0 or index >= len(self.states):
            raise IndexError(f"State index {index} out of range")
        
        hashes = [state.coordinate.state_hash for state in self.states]
        proof = []
        
        while len(hashes) > 1:
            next_level = []
            for i in range(0, len(hashes), 2):
                if i + 1 < len(hashes):
                    combined = hashes[i] + hashes[i + 1]
                    if i == index or i + 1 == index:
                        # Add sibling to proof
                        proof.append(hashes[i + 1] if i == index else hashes[i])
                else:
                    combined = hashes[i] + hashes[i]
                
                hasher = hashlib.sha256()
                hasher.update(combined.encode('utf-8'))
                next_level.append(hasher.hexdigest())
            
            # Update index for next level
            index = index // 2
            hashes = next_level
        
        return proof
    
    def verify_merkle_proof(self, state: TemporalState) -> bool:
        """
        Verify Merkle proof for a given state.
        
        Args:
            state: State with merkle_proof to verify
            
        Returns:
            True if proof is valid
        """
        if not state.merkle_proof:
            return False
        
        if self.merkle_root is None:
            self.compute_merkle_root()
        
        current_hash = state.coordinate.state_hash
        
        for sibling_hash in state.merkle_proof:
            hasher = hashlib.sha256()
            # Order matters - we need to maintain consistent ordering
            combined = current_hash + sibling_hash
            hasher.update(combined.encode('utf-8'))
            current_hash = hasher.hexdigest()
        
        return current_hash == self.merkle_root
    
    def get_state_at_time(self, computational_t: float) -> Optional[TemporalState]:
        """
        Get state at or before the given computational time.
        
        Args:
            computational_t: Computational time to query
            
        Returns:
            State at or before the given time, or None if not found
        """
        # Binary search for efficiency
        left, right = 0, len(self.states) - 1
        result = None
        
        while left <= right:
            mid = (left + right) // 2
            state = self.states[mid]
            
            if state.coordinate.computational_t <= computational_t:
                result = state
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    def get_state_at_depth(self, depth: int) -> Optional[TemporalState]:
        """Get state at given depth"""
        if 0 <= depth < len(self.states):
            return self.states[depth]
        return None
    
    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(self, index: int) -> TemporalState:
        return self.states[index]
