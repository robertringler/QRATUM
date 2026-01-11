"""QuASIM Deterministic Reproducibility Module

Provides deterministic quantum simulation with:
- Seed-locked execution for exact replay
- Merkle-hashed quantum state fingerprinting
- Drift detection between simulation runs

This module ensures that quantum simulations can be:
1. Reproduced exactly given the same seed
2. Verified via cryptographic state fingerprints
3. Compared across runs for drift detection

Usage:
    from quasim.reproducibility import (
        DeterministicSimulator,
        QuantumStateFingerprint,
        DriftDetector,
    )
    
    # Create seed-locked simulator
    sim = DeterministicSimulator(seed=42)
    
    # Run simulation with state tracking
    result = sim.run(circuit)
    
    # Get Merkle-hashed fingerprint
    fingerprint = sim.get_state_fingerprint()
    
    # Compare runs for drift
    detector = DriftDetector(tolerance=1e-10)
    drift = detector.compare(run1_fingerprint, run2_fingerprint)
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class QuantumStateFingerprint:
    """Merkle-hashed fingerprint of quantum state.
    
    Provides cryptographic verification of quantum simulation state
    without exposing the full state vector.
    
    Attributes:
        root_hash: Merkle root of state tree
        num_qubits: Number of qubits in state
        timestamp: Simulation timestamp
        seed: RNG seed used
        depth: Merkle tree depth
        amplitude_hashes: Hash of each amplitude chunk
    """
    
    root_hash: bytes
    num_qubits: int
    timestamp: float
    seed: int
    depth: int
    amplitude_hashes: list[bytes] = field(default_factory=list)
    
    def to_hex(self) -> str:
        """Get root hash as hex string."""
        return self.root_hash.hex()
    
    def verify(self, other: "QuantumStateFingerprint") -> bool:
        """Verify fingerprint matches another."""
        return (
            self.root_hash == other.root_hash and
            self.num_qubits == other.num_qubits and
            self.seed == other.seed
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "root_hash": self.root_hash.hex(),
            "num_qubits": self.num_qubits,
            "timestamp": self.timestamp,
            "seed": self.seed,
            "depth": self.depth,
            "amplitude_hashes": [h.hex() for h in self.amplitude_hashes],
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuantumStateFingerprint":
        """Create from dictionary."""
        return cls(
            root_hash=bytes.fromhex(data["root_hash"]),
            num_qubits=data["num_qubits"],
            timestamp=data["timestamp"],
            seed=data["seed"],
            depth=data["depth"],
            amplitude_hashes=[bytes.fromhex(h) for h in data.get("amplitude_hashes", [])],
        )


class MerkleTree:
    """Merkle tree for quantum state hashing.
    
    Builds a cryptographic Merkle tree from quantum state amplitudes.
    Enables verification of state integrity without full state comparison.
    """
    
    def __init__(self, chunk_size: int = 64):
        """Initialize Merkle tree builder.
        
        Args:
            chunk_size: Number of amplitudes per leaf node
        """
        self.chunk_size = chunk_size
        self._leaves: list[bytes] = []
        self._root: bytes | None = None
        
    def add_amplitudes(self, amplitudes: np.ndarray) -> None:
        """Add amplitudes to tree.
        
        Args:
            amplitudes: Complex amplitude array
        """
        # Split amplitudes into chunks
        flat = amplitudes.flatten()
        for i in range(0, len(flat), self.chunk_size):
            chunk = flat[i:i + self.chunk_size]
            chunk_hash = self._hash_chunk(chunk)
            self._leaves.append(chunk_hash)
        
        # Rebuild tree
        self._root = self._build_tree(self._leaves)
    
    def _hash_chunk(self, chunk: np.ndarray) -> bytes:
        """Hash a chunk of amplitudes."""
        hasher = hashlib.sha3_256()
        
        # Convert complex numbers to bytes deterministically
        for c in chunk:
            hasher.update(struct.pack('dd', c.real, c.imag))
        
        return hasher.digest()
    
    def _build_tree(self, leaves: list[bytes]) -> bytes:
        """Build Merkle tree and return root."""
        if not leaves:
            return hashlib.sha3_256(b"empty").digest()
        
        current_level = leaves.copy()
        
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                
                hasher = hashlib.sha3_256()
                hasher.update(left)
                hasher.update(right)
                next_level.append(hasher.digest())
            
            current_level = next_level
        
        return current_level[0]
    
    @property
    def root(self) -> bytes:
        """Get Merkle root."""
        return self._root or hashlib.sha3_256(b"empty").digest()
    
    @property
    def leaves(self) -> list[bytes]:
        """Get leaf hashes."""
        return self._leaves.copy()
    
    @property
    def depth(self) -> int:
        """Get tree depth."""
        if not self._leaves:
            return 0
        import math
        return math.ceil(math.log2(max(len(self._leaves), 1))) + 1


@dataclass
class SimulationTrace:
    """Trace of quantum operations for reproducibility.
    
    Records all operations performed during simulation
    for exact replay capability.
    """
    
    seed: int
    operations: list[dict[str, Any]] = field(default_factory=list)
    state_snapshots: list[QuantumStateFingerprint] = field(default_factory=list)
    
    def add_operation(self, op_type: str, params: dict[str, Any]) -> None:
        """Record an operation."""
        self.operations.append({
            "type": op_type,
            "params": params,
            "index": len(self.operations),
        })
    
    def add_snapshot(self, fingerprint: QuantumStateFingerprint) -> None:
        """Record a state snapshot."""
        self.state_snapshots.append(fingerprint)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize trace to dictionary."""
        return {
            "seed": self.seed,
            "operations": self.operations,
            "state_snapshots": [s.to_dict() for s in self.state_snapshots],
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationTrace":
        """Deserialize trace from dictionary."""
        trace = cls(seed=data["seed"])
        trace.operations = data.get("operations", [])
        trace.state_snapshots = [
            QuantumStateFingerprint.from_dict(s) 
            for s in data.get("state_snapshots", [])
        ]
        return trace


class DeterministicSimulator:
    """Seed-locked quantum simulator for exact replay.
    
    Provides deterministic quantum simulation where:
    - Same seed always produces same results
    - All random operations are reproducible
    - State can be fingerprinted for verification
    
    Example:
        sim = DeterministicSimulator(seed=42)
        result = sim.apply_gate("H", qubit=0)
        fingerprint = sim.get_state_fingerprint()
    """
    
    def __init__(
        self,
        seed: int,
        num_qubits: int = 4,
        precision: str = "fp64"
    ):
        """Initialize deterministic simulator.
        
        Args:
            seed: Random seed for reproducibility
            num_qubits: Number of qubits
            precision: Floating point precision ('fp32' or 'fp64')
        """
        self.seed = seed
        self.num_qubits = num_qubits
        self.precision = precision
        
        # Initialize deterministic RNG
        self._rng = np.random.Generator(np.random.PCG64(seed))
        
        # Initialize state vector
        dtype = np.complex128 if precision == "fp64" else np.complex64
        self._state = np.zeros(2 ** num_qubits, dtype=dtype)
        self._state[0] = 1.0  # |0...0> initial state
        
        # Trace for replay
        self._trace = SimulationTrace(seed=seed)
        
        # Merkle tree for fingerprinting
        self._merkle = MerkleTree()
        
        # Simulation timestamp
        import time
        self._timestamp = time.time()
        
        # Record initial state
        self._trace.add_operation("init", {
            "num_qubits": num_qubits,
            "precision": precision,
        })
        
    def apply_gate(self, gate: str, qubit: int, params: dict[str, Any] | None = None) -> None:
        """Apply quantum gate deterministically.
        
        Args:
            gate: Gate name ('H', 'X', 'Y', 'Z', 'CNOT', 'RX', 'RY', 'RZ')
            qubit: Target qubit index
            params: Gate parameters (for parameterized gates)
        """
        params = params or {}
        
        # Record operation
        self._trace.add_operation("gate", {
            "gate": gate,
            "qubit": qubit,
            "params": params,
        })
        
        # Get gate matrix
        matrix = self._get_gate_matrix(gate, params)
        
        # Apply gate
        self._apply_single_qubit_gate(matrix, qubit)
    
    def apply_cnot(self, control: int, target: int) -> None:
        """Apply CNOT gate deterministically.
        
        Args:
            control: Control qubit index
            target: Target qubit index
        """
        self._trace.add_operation("cnot", {
            "control": control,
            "target": target,
        })
        
        # Apply CNOT
        n = self.num_qubits
        for i in range(2 ** n):
            if (i >> (n - 1 - control)) & 1:
                j = i ^ (1 << (n - 1 - target))
                self._state[i], self._state[j] = self._state[j], self._state[i]
    
    def measure(self, qubit: int) -> int:
        """Measure qubit with deterministic randomness.
        
        Args:
            qubit: Qubit index to measure
            
        Returns:
            Measurement result (0 or 1)
        """
        # Calculate probabilities
        prob_0 = self._get_probability(qubit, 0)
        prob_1 = self._get_probability(qubit, 1)
        
        # Use deterministic RNG for measurement
        rand_val = self._rng.random()
        result = 0 if rand_val < prob_0 / (prob_0 + prob_1) else 1
        
        # Record measurement
        self._trace.add_operation("measure", {
            "qubit": qubit,
            "result": result,
            "prob_0": float(prob_0),
            "prob_1": float(prob_1),
            "rand_val": float(rand_val),
        })
        
        # Collapse state
        self._collapse_state(qubit, result)
        
        return result
    
    def get_state_fingerprint(self) -> QuantumStateFingerprint:
        """Get Merkle-hashed fingerprint of current state.
        
        Returns:
            QuantumStateFingerprint with cryptographic verification data
        """
        # Build Merkle tree from current state
        merkle = MerkleTree()
        merkle.add_amplitudes(self._state)
        
        fingerprint = QuantumStateFingerprint(
            root_hash=merkle.root,
            num_qubits=self.num_qubits,
            timestamp=self._timestamp,
            seed=self.seed,
            depth=merkle.depth,
            amplitude_hashes=merkle.leaves,
        )
        
        # Record snapshot in trace
        self._trace.add_snapshot(fingerprint)
        
        return fingerprint
    
    def get_trace(self) -> SimulationTrace:
        """Get full simulation trace for replay.
        
        Returns:
            SimulationTrace with all operations
        """
        return self._trace
    
    def get_state_vector(self) -> np.ndarray:
        """Get current state vector (copy).
        
        Returns:
            Copy of state vector
        """
        return self._state.copy()
    
    def reset(self) -> None:
        """Reset to initial state with same seed."""
        # Re-initialize RNG with same seed
        self._rng = np.random.Generator(np.random.PCG64(self.seed))
        
        # Reset state to |0...0>
        self._state.fill(0)
        self._state[0] = 1.0
        
        # Clear trace
        self._trace = SimulationTrace(seed=self.seed)
        self._trace.add_operation("reset", {})
    
    def replay(self, trace: SimulationTrace) -> None:
        """Replay simulation from trace.
        
        Args:
            trace: SimulationTrace to replay
        """
        # Ensure same seed
        if trace.seed != self.seed:
            raise ValueError(f"Trace seed {trace.seed} != simulator seed {self.seed}")
        
        # Reset state
        self.reset()
        
        # Replay operations
        for op in trace.operations:
            if op["type"] == "gate":
                self.apply_gate(
                    op["params"]["gate"],
                    op["params"]["qubit"],
                    op["params"].get("params", {}),
                )
            elif op["type"] == "cnot":
                self.apply_cnot(
                    op["params"]["control"],
                    op["params"]["target"],
                )
            elif op["type"] == "measure":
                # For replay, we don't actually measure (would change state)
                pass
    
    def _get_gate_matrix(self, gate: str, params: dict[str, Any]) -> np.ndarray:
        """Get matrix for gate."""
        if gate == "H":
            return np.array([[1, 1], [1, -1]], dtype=self._state.dtype) / np.sqrt(2)
        elif gate == "X":
            return np.array([[0, 1], [1, 0]], dtype=self._state.dtype)
        elif gate == "Y":
            return np.array([[0, -1j], [1j, 0]], dtype=self._state.dtype)
        elif gate == "Z":
            return np.array([[1, 0], [0, -1]], dtype=self._state.dtype)
        elif gate == "RX":
            theta = params.get("theta", 0)
            return np.array([
                [np.cos(theta/2), -1j * np.sin(theta/2)],
                [-1j * np.sin(theta/2), np.cos(theta/2)]
            ], dtype=self._state.dtype)
        elif gate == "RY":
            theta = params.get("theta", 0)
            return np.array([
                [np.cos(theta/2), -np.sin(theta/2)],
                [np.sin(theta/2), np.cos(theta/2)]
            ], dtype=self._state.dtype)
        elif gate == "RZ":
            theta = params.get("theta", 0)
            return np.array([
                [np.exp(-1j * theta/2), 0],
                [0, np.exp(1j * theta/2)]
            ], dtype=self._state.dtype)
        else:
            raise ValueError(f"Unknown gate: {gate}")
    
    def _apply_single_qubit_gate(self, matrix: np.ndarray, qubit: int) -> None:
        """Apply single-qubit gate to state vector."""
        n = self.num_qubits
        for i in range(2 ** n):
            if not ((i >> (n - 1 - qubit)) & 1):
                j = i | (1 << (n - 1 - qubit))
                a, b = self._state[i], self._state[j]
                self._state[i] = matrix[0, 0] * a + matrix[0, 1] * b
                self._state[j] = matrix[1, 0] * a + matrix[1, 1] * b
    
    def _get_probability(self, qubit: int, value: int) -> float:
        """Get probability of measuring value on qubit."""
        n = self.num_qubits
        prob = 0.0
        for i in range(2 ** n):
            if ((i >> (n - 1 - qubit)) & 1) == value:
                prob += abs(self._state[i]) ** 2
        return prob
    
    def _collapse_state(self, qubit: int, result: int) -> None:
        """Collapse state after measurement."""
        n = self.num_qubits
        norm = 0.0
        
        # Zero out non-matching amplitudes
        for i in range(2 ** n):
            if ((i >> (n - 1 - qubit)) & 1) != result:
                self._state[i] = 0
            else:
                norm += abs(self._state[i]) ** 2
        
        # Renormalize
        if norm > 0:
            self._state /= np.sqrt(norm)


@dataclass
class DriftReport:
    """Report of drift between simulation runs.
    
    Attributes:
        is_drifted: Whether significant drift was detected
        max_drift: Maximum amplitude drift
        mean_drift: Mean amplitude drift
        drifted_positions: Positions where drift exceeded tolerance
        fingerprint_match: Whether fingerprints match exactly
    """
    
    is_drifted: bool
    max_drift: float
    mean_drift: float
    drifted_positions: list[int]
    fingerprint_match: bool
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_drifted": self.is_drifted,
            "max_drift": self.max_drift,
            "mean_drift": self.mean_drift,
            "drifted_positions": self.drifted_positions,
            "fingerprint_match": self.fingerprint_match,
        }


class DriftDetector:
    """Detector for simulation drift between runs.
    
    Compares quantum simulation results to detect:
    - Numerical drift from floating point errors
    - Non-deterministic behavior from unseeded randomness
    - State corruption or bugs
    
    Example:
        detector = DriftDetector(tolerance=1e-10)
        
        sim1 = DeterministicSimulator(seed=42)
        sim1.apply_gate("H", 0)
        fp1 = sim1.get_state_fingerprint()
        state1 = sim1.get_state_vector()
        
        sim2 = DeterministicSimulator(seed=42)
        sim2.apply_gate("H", 0)
        fp2 = sim2.get_state_fingerprint()
        state2 = sim2.get_state_vector()
        
        report = detector.compare_states(state1, state2)
        assert not report.is_drifted
    """
    
    def __init__(self, tolerance: float = 1e-10):
        """Initialize drift detector.
        
        Args:
            tolerance: Maximum allowed amplitude difference
        """
        self.tolerance = tolerance
    
    def compare_fingerprints(
        self,
        fp1: QuantumStateFingerprint,
        fp2: QuantumStateFingerprint,
    ) -> bool:
        """Compare two fingerprints for equality.
        
        Args:
            fp1: First fingerprint
            fp2: Second fingerprint
            
        Returns:
            True if fingerprints match exactly
        """
        return fp1.verify(fp2)
    
    def compare_states(
        self,
        state1: np.ndarray,
        state2: np.ndarray,
    ) -> DriftReport:
        """Compare two state vectors for drift.
        
        Args:
            state1: First state vector
            state2: Second state vector
            
        Returns:
            DriftReport with detailed comparison
        """
        if state1.shape != state2.shape:
            return DriftReport(
                is_drifted=True,
                max_drift=float('inf'),
                mean_drift=float('inf'),
                drifted_positions=[],
                fingerprint_match=False,
            )
        
        # Calculate amplitude differences
        diff = np.abs(state1 - state2)
        
        # Find drifted positions
        drifted_mask = diff > self.tolerance
        drifted_positions = list(np.where(drifted_mask)[0])
        
        max_drift = float(np.max(diff))
        mean_drift = float(np.mean(diff))
        is_drifted = max_drift > self.tolerance
        
        # Compare fingerprints
        merkle1 = MerkleTree()
        merkle1.add_amplitudes(state1)
        merkle2 = MerkleTree()
        merkle2.add_amplitudes(state2)
        fingerprint_match = merkle1.root == merkle2.root
        
        return DriftReport(
            is_drifted=is_drifted,
            max_drift=max_drift,
            mean_drift=mean_drift,
            drifted_positions=drifted_positions,
            fingerprint_match=fingerprint_match,
        )
    
    def compare_traces(
        self,
        trace1: SimulationTrace,
        trace2: SimulationTrace,
    ) -> dict[str, Any]:
        """Compare two simulation traces.
        
        Args:
            trace1: First trace
            trace2: Second trace
            
        Returns:
            Dictionary with comparison results
        """
        results = {
            "seed_match": trace1.seed == trace2.seed,
            "operation_count_match": len(trace1.operations) == len(trace2.operations),
            "snapshot_count_match": len(trace1.state_snapshots) == len(trace2.state_snapshots),
            "operation_diffs": [],
            "snapshot_matches": [],
        }
        
        # Compare operations
        for i, (op1, op2) in enumerate(zip(trace1.operations, trace2.operations)):
            if op1 != op2:
                results["operation_diffs"].append({
                    "index": i,
                    "trace1": op1,
                    "trace2": op2,
                })
        
        # Compare snapshots
        for i, (s1, s2) in enumerate(zip(trace1.state_snapshots, trace2.state_snapshots)):
            results["snapshot_matches"].append({
                "index": i,
                "match": s1.verify(s2),
            })
        
        return results


__all__ = [
    "QuantumStateFingerprint",
    "MerkleTree",
    "SimulationTrace",
    "DeterministicSimulator",
    "DriftReport",
    "DriftDetector",
]
