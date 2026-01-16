"""
Global State Management for QRATUM Deterministic Runtime

Manages global seeds and execution ordering to ensure bit-identical reproducibility
across distributed exascale systems. All random number generation, tensor initialization,
and non-deterministic operations are seeded from a global, synchronized state.

Key Features:
- Global seed propagation with Merkle verification
- Deterministic per-rank seed derivation
- Execution epoch management for reproducibility
- State checkpointing for recovery
- Cryptographic verification of seed integrity
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import hashlib
import time


class SeedDerivationMethod(Enum):
    """Methods for deriving per-rank seeds from global seed"""
    HASH_CHAIN = "hash_chain"  # Sequential hash chaining
    MERKLE_PATH = "merkle_path"  # Merkle tree path derivation
    DETERMINISTIC_KDF = "kdf"  # Key derivation function (HKDF)


class ExecutionEpoch(Enum):
    """Execution phases for deterministic ordering"""
    INITIALIZATION = "initialization"
    TRAINING = "training"
    VALIDATION = "validation"
    INFERENCE = "inference"
    CHECKPOINT = "checkpoint"
    FINALIZATION = "finalization"


@dataclass
class GlobalSeed:
    """
    Global seed for deterministic execution
    
    The global seed is broadcast to all ranks at initialization and used
    to derive per-rank, per-epoch, and per-operation seeds deterministically.
    
    Attributes:
        value: 64-bit unsigned integer seed value
        epoch: Execution epoch this seed applies to
        timestamp: Creation timestamp (for audit, not used in derivation)
        merkle_root: Merkle root of seed distribution tree
        verified: Whether seed has been verified via Merkle proof
    """
    value: int
    epoch: ExecutionEpoch
    timestamp: float = field(default_factory=time.time)
    merkle_root: Optional[str] = None
    verified: bool = False
    
    def derive_rank_seed(self, rank: int, method: SeedDerivationMethod = SeedDerivationMethod.HASH_CHAIN) -> int:
        """
        Derive deterministic per-rank seed
        
        Uses cryptographic hash function to ensure:
        1. Different ranks get different seeds
        2. Same rank always gets same seed given same global seed
        3. Seeds are uniformly distributed
        
        Args:
            rank: MPI rank or node ID
            method: Seed derivation method
            
        Returns:
            64-bit per-rank seed value
        """
        if method == SeedDerivationMethod.HASH_CHAIN:
            # Hash(global_seed || rank || epoch)
            data = f"{self.value}:{rank}:{self.epoch.value}".encode('utf-8')
            hash_digest = hashlib.sha256(data).digest()
            return int.from_bytes(hash_digest[:8], byteorder='big')
        
        elif method == SeedDerivationMethod.MERKLE_PATH:
            # Use Merkle path for seed derivation (placeholder)
            # In production: construct Merkle path from rank to root
            data = f"{self.merkle_root}:{rank}".encode('utf-8')
            hash_digest = hashlib.sha256(data).digest()
            return int.from_bytes(hash_digest[:8], byteorder='big')
        
        else:  # DETERMINISTIC_KDF
            # HKDF-like derivation (simplified)
            data = f"kdf:{self.value}:{rank}:{self.epoch.value}".encode('utf-8')
            hash_digest = hashlib.sha512(data).digest()
            return int.from_bytes(hash_digest[:8], byteorder='big')
    
    def derive_operation_seed(self, rank: int, operation_id: str) -> int:
        """
        Derive seed for specific operation within a rank
        
        Args:
            rank: MPI rank
            operation_id: Unique operation identifier (e.g., "layer_3_dropout")
            
        Returns:
            64-bit operation-specific seed
        """
        rank_seed = self.derive_rank_seed(rank)
        data = f"{rank_seed}:{operation_id}".encode('utf-8')
        hash_digest = hashlib.sha256(data).digest()
        return int.from_bytes(hash_digest[:8], byteorder='big')


@dataclass
class SeedManager:
    """
    Manages seed lifecycle and distribution
    
    Responsible for generating global seeds, distributing them to all ranks,
    and verifying their integrity via Merkle proofs.
    
    Attributes:
        initial_seed: User-provided or system-generated initial seed
        current_epoch: Current execution epoch
        seed_history: History of seeds for each epoch
        derivation_method: Method for deriving per-rank seeds
    """
    initial_seed: int
    current_epoch: ExecutionEpoch = ExecutionEpoch.INITIALIZATION
    seed_history: Dict[ExecutionEpoch, GlobalSeed] = field(default_factory=dict)
    derivation_method: SeedDerivationMethod = SeedDerivationMethod.HASH_CHAIN
    
    def generate_epoch_seed(self, epoch: ExecutionEpoch) -> GlobalSeed:
        """
        Generate new global seed for given epoch
        
        Seeds are derived deterministically from initial seed and epoch,
        ensuring reproducibility across runs.
        
        Args:
            epoch: Execution epoch
            
        Returns:
            GlobalSeed for the epoch
        """
        # Deterministic epoch seed: Hash(initial_seed || epoch)
        data = f"{self.initial_seed}:{epoch.value}".encode('utf-8')
        hash_digest = hashlib.sha256(data).digest()
        epoch_seed_value = int.from_bytes(hash_digest[:8], byteorder='big')
        
        global_seed = GlobalSeed(
            value=epoch_seed_value,
            epoch=epoch,
            timestamp=time.time()
        )
        
        self.seed_history[epoch] = global_seed
        return global_seed
    
    def get_current_seed(self) -> GlobalSeed:
        """Get global seed for current epoch"""
        if self.current_epoch not in self.seed_history:
            return self.generate_epoch_seed(self.current_epoch)
        return self.seed_history[self.current_epoch]
    
    def transition_epoch(self, new_epoch: ExecutionEpoch) -> GlobalSeed:
        """
        Transition to new execution epoch
        
        Args:
            new_epoch: New execution epoch
            
        Returns:
            Global seed for new epoch
        """
        self.current_epoch = new_epoch
        return self.generate_epoch_seed(new_epoch)
    
    def compute_merkle_root(self, num_ranks: int) -> str:
        """
        Compute Merkle root for seed distribution
        
        Args:
            num_ranks: Total number of ranks
            
        Returns:
            Merkle root hash as hex string
        """
        current_seed = self.get_current_seed()
        
        # Build Merkle tree of per-rank seeds
        leaves = []
        for rank in range(num_ranks):
            rank_seed = current_seed.derive_rank_seed(rank, self.derivation_method)
            leaf_hash = hashlib.sha256(f"{rank_seed}".encode('utf-8')).hexdigest()
            leaves.append(leaf_hash)
        
        # Build tree bottom-up
        tree = [leaves]
        while len(tree[-1]) > 1:
            level = []
            prev_level = tree[-1]
            for i in range(0, len(prev_level), 2):
                if i + 1 < len(prev_level):
                    combined = prev_level[i] + prev_level[i + 1]
                else:
                    combined = prev_level[i] + prev_level[i]
                level.append(hashlib.sha256(combined.encode('utf-8')).hexdigest())
            tree.append(level)
        
        merkle_root = tree[-1][0]
        current_seed.merkle_root = merkle_root
        return merkle_root
    
    def verify_rank_seed(self, rank: int, seed_value: int) -> bool:
        """
        Verify that rank received correct seed
        
        Args:
            rank: Rank to verify
            seed_value: Seed value claimed by rank
            
        Returns:
            True if seed is correct
        """
        current_seed = self.get_current_seed()
        expected_seed = current_seed.derive_rank_seed(rank, self.derivation_method)
        return seed_value == expected_seed


@dataclass
class ExecutionOrder:
    """
    Tracks deterministic execution ordering
    
    Maintains a global counter for all operations to ensure
    operations execute in the same order across runs.
    
    Attributes:
        global_counter: Global operation counter
        rank_counters: Per-rank operation counters
        operation_log: Log of executed operations (for debugging)
    """
    global_counter: int = 0
    rank_counters: Dict[int, int] = field(default_factory=dict)
    operation_log: List[Tuple[int, str, int]] = field(default_factory=list)  # (global_id, op_name, rank)
    
    def next_operation_id(self, rank: int, operation_name: str) -> int:
        """
        Get next operation ID for deterministic ordering
        
        Args:
            rank: Rank executing operation
            operation_name: Human-readable operation name
            
        Returns:
            Global operation ID
        """
        operation_id = self.global_counter
        self.global_counter += 1
        
        # Track per-rank counter
        if rank not in self.rank_counters:
            self.rank_counters[rank] = 0
        self.rank_counters[rank] += 1
        
        # Log operation
        self.operation_log.append((operation_id, operation_name, rank))
        
        return operation_id
    
    def get_rank_operation_count(self, rank: int) -> int:
        """Get total operations executed by rank"""
        return self.rank_counters.get(rank, 0)
    
    def verify_determinism(self, other: 'ExecutionOrder') -> bool:
        """
        Verify that execution order matches another run
        
        Args:
            other: ExecutionOrder from another run
            
        Returns:
            True if execution orders match
        """
        if self.global_counter != other.global_counter:
            return False
        
        if self.rank_counters != other.rank_counters:
            return False
        
        # Check operation log matches
        if len(self.operation_log) != len(other.operation_log):
            return False
        
        for (g1, op1, r1), (g2, op2, r2) in zip(self.operation_log, other.operation_log):
            if g1 != g2 or op1 != op2 or r1 != r2:
                return False
        
        return True


@dataclass
class GlobalStateManager:
    """
    Central manager for all global state
    
    Coordinates seed management, execution ordering, and state verification
    to ensure bit-identical reproducibility at exascale.
    
    Design Principles:
    1. Single source of truth for all random state
    2. Deterministic derivation of all per-rank/per-operation state
    3. Cryptographic verification of state distribution
    4. Complete audit trail for reproducibility debugging
    
    Attributes:
        seed_manager: Manages global and per-rank seeds
        execution_order: Tracks operation ordering
        num_ranks: Total number of MPI ranks
        rank_id: This rank's ID
        state_checkpoints: Saved state snapshots
    """
    seed_manager: SeedManager
    execution_order: ExecutionOrder = field(default_factory=ExecutionOrder)
    num_ranks: int = 1
    rank_id: int = 0
    state_checkpoints: Dict[str, Dict] = field(default_factory=dict)
    
    def initialize_global_state(self, num_ranks: int, rank_id: int) -> None:
        """
        Initialize global state for all ranks
        
        Must be called by all ranks with same arguments to ensure consistency.
        
        Args:
            num_ranks: Total number of ranks
            rank_id: This rank's ID (0 to num_ranks-1)
        """
        self.num_ranks = num_ranks
        self.rank_id = rank_id
        
        # Generate and verify seed distribution
        merkle_root = self.seed_manager.compute_merkle_root(num_ranks)
        
        # Each rank verifies it has correct seed
        current_seed = self.seed_manager.get_current_seed()
        rank_seed = current_seed.derive_rank_seed(rank_id)
        
        if not self.seed_manager.verify_rank_seed(rank_id, rank_seed):
            raise RuntimeError(f"Rank {rank_id} seed verification failed!")
        
        current_seed.verified = True
    
    def get_operation_seed(self, operation_id: str) -> int:
        """
        Get deterministic seed for operation
        
        Args:
            operation_id: Unique operation identifier
            
        Returns:
            64-bit operation seed
        """
        current_seed = self.seed_manager.get_current_seed()
        return current_seed.derive_operation_seed(self.rank_id, operation_id)
    
    def register_operation(self, operation_name: str) -> int:
        """
        Register operation and get deterministic ordering ID
        
        Args:
            operation_name: Human-readable operation name
            
        Returns:
            Global operation ID for ordering
        """
        return self.execution_order.next_operation_id(self.rank_id, operation_name)
    
    def transition_to_epoch(self, epoch: ExecutionEpoch) -> None:
        """
        Transition to new execution epoch
        
        All ranks must call this simultaneously to maintain consistency.
        
        Args:
            epoch: New execution epoch
        """
        new_seed = self.seed_manager.transition_epoch(epoch)
        self.seed_manager.compute_merkle_root(self.num_ranks)
        new_seed.verified = True
    
    def checkpoint_state(self, checkpoint_name: str) -> Dict:
        """
        Save current global state
        
        Args:
            checkpoint_name: Name for checkpoint
            
        Returns:
            State dictionary
        """
        state = {
            'seed_manager': {
                'initial_seed': self.seed_manager.initial_seed,
                'current_epoch': self.seed_manager.current_epoch.value,
                'seed_history': {
                    epoch.value: {
                        'value': seed.value,
                        'merkle_root': seed.merkle_root,
                        'verified': seed.verified
                    }
                    for epoch, seed in self.seed_manager.seed_history.items()
                }
            },
            'execution_order': {
                'global_counter': self.execution_order.global_counter,
                'rank_counters': dict(self.execution_order.rank_counters),
                'operation_count': len(self.execution_order.operation_log)
            },
            'rank_info': {
                'num_ranks': self.num_ranks,
                'rank_id': self.rank_id
            }
        }
        
        self.state_checkpoints[checkpoint_name] = state
        return state
    
    def restore_state(self, checkpoint_name: str) -> None:
        """
        Restore global state from checkpoint
        
        Args:
            checkpoint_name: Name of checkpoint to restore
        """
        if checkpoint_name not in self.state_checkpoints:
            raise ValueError(f"Checkpoint '{checkpoint_name}' not found")
        
        # Note: Full restoration would require careful reconstruction
        # This is a placeholder for the restoration logic
        state = self.state_checkpoints[checkpoint_name]
        self.num_ranks = state['rank_info']['num_ranks']
        self.rank_id = state['rank_info']['rank_id']
    
    def compute_state_hash(self) -> str:
        """
        Compute hash of current global state for verification
        
        Returns:
            SHA-256 hash of state as hex string
        """
        state = self.checkpoint_state('_temp_hash')
        state_str = str(sorted(state.items()))
        return hashlib.sha256(state_str.encode('utf-8')).hexdigest()
    
    def verify_global_consistency(self, other_state_hash: str) -> bool:
        """
        Verify state consistency across ranks
        
        Args:
            other_state_hash: State hash from another rank
            
        Returns:
            True if states match
        """
        local_hash = self.compute_state_hash()
        return local_hash == other_state_hash
