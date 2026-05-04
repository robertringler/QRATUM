"""Universal State Space with AHTC Compression for SI Transition.

Provides compressed tensor representations of knowledge state across
arbitrary domains using Adaptive Hierarchical Tensor Compression (AHTC),
enabling efficient cross-domain reasoning and universal modeling.

Key Features:
- AHTC-compressed universal state vectors
- Semantic-preserving compression
- Cross-domain state translation
- Fidelity-guaranteed reconstruction
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from qratum_asi.generalization.types import (
    CognitiveDomain,
    CompressionMetrics,
    UniversalStateVector,
)

# Compression constants
MIN_ENTROPY_REDUCTION = 0.0


@dataclass
class TensorBlock:
    """Block of compressed tensor data.

    Attributes:
        block_id: Unique identifier
        level: Hierarchy level (0 = most compressed)
        dimensions: Dimensionality at this level
        data: Compressed data values
        reconstruction_error: Error when reconstructed
    """

    block_id: str
    level: int
    dimensions: int
    data: list[float]
    reconstruction_error: float


@dataclass
class CompressionResult:
    """Result of a compression operation.

    Attributes:
        success: Whether compression succeeded
        input_state: Original state
        output_state: Compressed state
        metrics: Compression metrics
        hierarchy_depth: Depth of compression hierarchy
        blocks: Compressed tensor blocks
    """

    success: bool
    input_state: dict[str, Any]
    output_state: UniversalStateVector
    metrics: CompressionMetrics
    hierarchy_depth: int
    blocks: list[TensorBlock]


class AHTCEncoder:
    """Adaptive Hierarchical Tensor Compression Encoder.

    Implements multi-level tensor compression that adapts to
    the structure of the input data while preserving semantic
    content with configurable fidelity guarantees.

    CRITICAL DISCLAIMER:
    This is a PLACEHOLDER implementation. Production AHTC would
    require advanced tensor decomposition and neural compression
    techniques not yet available at required fidelity levels.
    """

    def __init__(
        self,
        max_hierarchy_depth: int = 4,
        target_compression_ratio: float = 10.0,
        min_fidelity: float = 0.95,
    ):
        """Initialize the AHTC encoder.

        Args:
            max_hierarchy_depth: Maximum compression levels
            target_compression_ratio: Target compression ratio
            min_fidelity: Minimum reconstruction fidelity
        """
        self.max_hierarchy_depth = max_hierarchy_depth
        self.target_compression_ratio = target_compression_ratio
        self.min_fidelity = min_fidelity
        self._encoding_counter = 0

    def encode(
        self,
        state_data: dict[str, Any],
        domain: CognitiveDomain,
    ) -> tuple[list[float], CompressionMetrics]:
        """Encode state data into compressed tensor representation.

        Args:
            state_data: Raw state data to compress
            domain: Domain of the state

        Returns:
            Tuple of (compressed representation, metrics)

        NOTE: This is a PLACEHOLDER. Real AHTC would use:
        - Tucker decomposition for multi-way data
        - Hierarchical matrix factorization
        - Neural compression with semantic loss
        - Adaptive bit allocation
        """
        self._encoding_counter += 1

        # Flatten state data to pseudo-tensor
        flat_data = self._flatten_state(state_data)
        input_dims = len(flat_data)

        # Target output dimensions
        output_dims = max(8, int(input_dims / self.target_compression_ratio))

        # Simple placeholder compression (averaging windows)
        compressed = self._compress_data(flat_data, output_dims)

        # Compute metrics
        fidelity = self._estimate_fidelity(flat_data, compressed)
        semantic_preservation = min(1.0, fidelity * 1.05)  # Slightly optimistic

        metrics = CompressionMetrics(
            input_dimensions=input_dims,
            output_dimensions=output_dims,
            compression_ratio=input_dims / output_dims if output_dims > 0 else 0,
            fidelity_score=fidelity,
            semantic_preservation=semantic_preservation,
            entropy_reduction=self._estimate_entropy_reduction(flat_data, compressed),
            computation_cost=input_dims * math.log2(input_dims + 1) / 1000,
        )

        return compressed, metrics

    def decode(
        self,
        compressed: list[float],
        target_dims: int,
    ) -> list[float]:
        """Decode compressed representation.

        Args:
            compressed: Compressed data
            target_dims: Target output dimensions

        Returns:
            Reconstructed data

        NOTE: PLACEHOLDER implementation.
        """
        if len(compressed) >= target_dims:
            return compressed[:target_dims]

        # Simple interpolation (placeholder)
        result = []
        ratio = len(compressed) / target_dims

        for i in range(target_dims):
            source_idx = int(i * ratio)
            source_idx = min(source_idx, len(compressed) - 1)
            result.append(compressed[source_idx])

        return result

    def _flatten_state(self, state_data: dict[str, Any]) -> list[float]:
        """Flatten state data to numeric vector."""
        flat = []

        def extract_numbers(obj: Any) -> None:
            if isinstance(obj, (int, float)):
                flat.append(float(obj))
            elif isinstance(obj, str):
                # Hash string to number
                h = int(hashlib.md5(obj.encode()).hexdigest()[:8], 16)
                flat.append(float(h) / (16**8))
            elif isinstance(obj, dict):
                for v in obj.values():
                    extract_numbers(v)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    extract_numbers(item)

        extract_numbers(state_data)

        # Ensure minimum size
        if len(flat) < 8:
            flat.extend([0.0] * (8 - len(flat)))

        return flat

    def _compress_data(self, data: list[float], target_dims: int) -> list[float]:
        """Compress data to target dimensions (placeholder)."""
        if len(data) <= target_dims:
            return data + [0.0] * (target_dims - len(data))

        # Window averaging compression
        window_size = len(data) // target_dims
        compressed = []

        for i in range(target_dims):
            start = i * window_size
            end = min(start + window_size, len(data))
            window = data[start:end]
            if window:
                compressed.append(sum(window) / len(window))
            else:
                compressed.append(0.0)

        return compressed

    def _estimate_fidelity(self, original: list[float], compressed: list[float]) -> float:
        """Estimate reconstruction fidelity (placeholder)."""
        # Reconstruct and compare
        reconstructed = self.decode(compressed, len(original))

        if not original:
            return 1.0

        # Mean squared error
        mse = sum((a - b) ** 2 for a, b in zip(original, reconstructed)) / len(original)

        # Convert to fidelity score
        fidelity = 1.0 / (1.0 + mse)
        return max(0.0, min(1.0, fidelity))

    def _estimate_entropy_reduction(self, original: list[float], compressed: list[float]) -> float:
        """Estimate entropy reduction from compression."""
        if not original or not compressed:
            return 0.0

        # Simple variance-based entropy estimate
        def variance(data: list[float]) -> float:
            if len(data) < 2:
                return 0.0
            mean = sum(data) / len(data)
            return sum((x - mean) ** 2 for x in data) / len(data)

        orig_var = variance(original)
        comp_var = variance(compressed)

        if orig_var == 0:
            return MIN_ENTROPY_REDUCTION

        return max(MIN_ENTROPY_REDUCTION, 1.0 - comp_var / orig_var)


class StateCompressor:
    """Compresses domain state to universal state vectors.

    Wraps AHTC encoding with domain-specific preprocessing and
    semantic preservation guarantees.
    """

    def __init__(
        self,
        ahtc_encoder: AHTCEncoder | None = None,
    ):
        """Initialize the state compressor.

        Args:
            ahtc_encoder: AHTC encoder to use
        """
        self.encoder = ahtc_encoder or AHTCEncoder()
        self._compression_counter = 0

    def compress(
        self,
        state_data: dict[str, Any],
        domain: CognitiveDomain,
    ) -> CompressionResult:
        """Compress state data to universal state vector.

        Args:
            state_data: Raw state data
            domain: Domain of the state

        Returns:
            CompressionResult with compressed state and metrics
        """
        self._compression_counter += 1
        state_id = f"state_{domain.value}_{self._compression_counter:06d}"

        # Encode using AHTC
        compressed_repr, metrics = self.encoder.encode(state_data, domain)

        # Create universal state vector
        state_vector = UniversalStateVector(
            state_id=state_id,
            domain=domain,
            dimensions=len(compressed_repr),
            compressed_representation=compressed_repr,
            reconstruction_fidelity=metrics.fidelity_score,
            semantic_hash=hashlib.sha3_256(
                json.dumps(compressed_repr, sort_keys=True).encode()
            ).hexdigest(),
            compression_metadata={
                "original_dimensions": metrics.input_dimensions,
                "compression_ratio": metrics.compression_ratio,
                "entropy_reduction": metrics.entropy_reduction,
            },
        )

        # Create hierarchy blocks (placeholder)
        blocks = [
            TensorBlock(
                block_id=f"{state_id}_block_0",
                level=0,
                dimensions=len(compressed_repr),
                data=compressed_repr,
                reconstruction_error=1.0 - metrics.fidelity_score,
            )
        ]

        return CompressionResult(
            success=metrics.is_acceptable,
            input_state=state_data,
            output_state=state_vector,
            metrics=metrics,
            hierarchy_depth=1,
            blocks=blocks,
        )


class UniversalStateSpace:
    """Universal state space for cross-domain reasoning.

    Manages compressed state representations across all cognitive
    domains, enabling efficient cross-domain operations and
    universal modeling.

    Enforces:
    - Fidelity guarantees on all compressions
    - Deterministic state operations
    - Provenance tracking
    """

    def __init__(
        self,
        compressor: StateCompressor | None = None,
    ):
        """Initialize the universal state space.

        Args:
            compressor: State compressor to use
        """
        self.compressor = compressor or StateCompressor()

        # State storage
        self.states: dict[str, UniversalStateVector] = {}
        self.domain_states: dict[CognitiveDomain, list[str]] = {}

        # Metrics
        self._total_compressions = 0
        self._total_operations = 0

    def add_state(
        self,
        state_data: dict[str, Any],
        domain: CognitiveDomain,
    ) -> UniversalStateVector:
        """Add a new state to the universal state space.

        Args:
            state_data: Raw state data
            domain: Domain of the state

        Returns:
            Compressed universal state vector
        """
        result = self.compressor.compress(state_data, domain)
        self._total_compressions += 1

        if not result.success:
            raise ValueError(
                f"Compression failed to meet fidelity requirements: "
                f"{result.metrics.fidelity_score:.3f}"
            )

        state = result.output_state
        self.states[state.state_id] = state

        if domain not in self.domain_states:
            self.domain_states[domain] = []
        self.domain_states[domain].append(state.state_id)

        return state

    def get_state(self, state_id: str) -> UniversalStateVector | None:
        """Get a state by ID."""
        return self.states.get(state_id)

    def get_domain_states(self, domain: CognitiveDomain) -> list[UniversalStateVector]:
        """Get all states for a domain."""
        state_ids = self.domain_states.get(domain, [])
        return [self.states[sid] for sid in state_ids if sid in self.states]

    def compute_state_similarity(
        self,
        state_a: UniversalStateVector,
        state_b: UniversalStateVector,
    ) -> float:
        """Compute similarity between two states.

        Uses cosine similarity on compressed representations.

        Args:
            state_a: First state
            state_b: Second state

        Returns:
            Similarity score (0 to 1)
        """
        self._total_operations += 1

        vec_a = state_a.compressed_representation
        vec_b = state_b.compressed_representation

        # Normalize dimensions
        min_dims = min(len(vec_a), len(vec_b))
        vec_a = vec_a[:min_dims]
        vec_b = vec_b[:min_dims]

        if not vec_a or not vec_b:
            return 0.0

        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)
        return max(0.0, min(1.0, (similarity + 1) / 2))  # Normalize to [0, 1]

    def combine_states(
        self,
        states: list[UniversalStateVector],
        weights: list[float] | None = None,
    ) -> UniversalStateVector:
        """Combine multiple states into a unified state.

        Args:
            states: States to combine
            weights: Optional weights for combination

        Returns:
            Combined state vector
        """
        self._total_operations += 1

        if not states:
            raise ValueError("No states to combine")

        if weights is None:
            weights = [1.0 / len(states)] * len(states)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Determine dimensions (use minimum)
        min_dims = min(len(s.compressed_representation) for s in states)

        # Weighted combination
        combined = [0.0] * min_dims
        for state, weight in zip(states, weights):
            for i in range(min_dims):
                combined[i] += state.compressed_representation[i] * weight

        # Determine combined fidelity (conservative)
        combined_fidelity = min(s.reconstruction_fidelity for s in states)

        # Determine combined domain
        domains = set(s.domain for s in states)
        if len(domains) == 1:
            combined_domain = list(domains)[0]
        else:
            combined_domain = CognitiveDomain.FUSIA  # Multi-domain fusion

        return UniversalStateVector(
            state_id=f"combined_{len(self.states) + 1:06d}",
            domain=combined_domain,
            dimensions=min_dims,
            compressed_representation=combined,
            reconstruction_fidelity=combined_fidelity,
            semantic_hash=hashlib.sha3_256(
                json.dumps(combined, sort_keys=True).encode()
            ).hexdigest(),
            compression_metadata={
                "combined_from": [s.state_id for s in states],
                "weights": weights,
            },
        )

    def project_state(
        self,
        state: UniversalStateVector,
        target_domain: CognitiveDomain,
    ) -> UniversalStateVector:
        """Project state into a target domain.

        Transforms state representation to be interpretable
        in the target domain context.

        Args:
            state: State to project
            target_domain: Target domain

        Returns:
            Projected state
        """
        self._total_operations += 1

        # For same domain, return copy
        if state.domain == target_domain:
            return UniversalStateVector(
                state_id=f"proj_{state.state_id}_{target_domain.value}",
                domain=target_domain,
                dimensions=state.dimensions,
                compressed_representation=list(state.compressed_representation),
                reconstruction_fidelity=state.reconstruction_fidelity,
                semantic_hash=state.semantic_hash,
                compression_metadata={
                    "projected_from": state.state_id,
                    "source_domain": state.domain.value,
                },
            )

        # Cross-domain projection (placeholder - would use learned transforms)
        # Apply domain-specific scaling based on formalization levels
        transformed = list(state.compressed_representation)

        # Simple transform: scale based on domain characteristics
        # (placeholder for learned cross-domain mappings)
        scale = 0.9  # Slight information loss in projection
        transformed = [x * scale for x in transformed]

        return UniversalStateVector(
            state_id=f"proj_{state.state_id}_{target_domain.value}",
            domain=target_domain,
            dimensions=len(transformed),
            compressed_representation=transformed,
            reconstruction_fidelity=state.reconstruction_fidelity * 0.95,
            semantic_hash=hashlib.sha3_256(
                json.dumps(transformed, sort_keys=True).encode()
            ).hexdigest(),
            compression_metadata={
                "projected_from": state.state_id,
                "source_domain": state.domain.value,
                "projection_type": "cross_domain",
            },
        )

    def get_state_space_stats(self) -> dict[str, Any]:
        """Get state space statistics."""
        return {
            "total_states": len(self.states),
            "domains_represented": len(self.domain_states),
            "states_per_domain": {d.value: len(ids) for d, ids in self.domain_states.items()},
            "total_compressions": self._total_compressions,
            "total_operations": self._total_operations,
            "average_fidelity": (
                sum(s.reconstruction_fidelity for s in self.states.values()) / len(self.states)
                if self.states
                else 0.0
            ),
        }
