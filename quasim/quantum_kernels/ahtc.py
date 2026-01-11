"""AHTC (Advanced Hierarchical Tensor Compression) Acceleration.

This module provides AHTC acceleration hooks for quantum-inspired
algorithms, enabling efficient compression of high-dimensional
data and intermediate results.

Key features:
- Hierarchical tensor decomposition
- Adaptive compression based on error bounds
- Integration with MPS/PEPS simulations
- Provenance tracking for compressed results

Applications:
- Accelerating VQE/QAOA classical analogs
- Compressing simulation states
- Reducing memory footprint for large systems
"""

from __future__ import annotations

import time
import uuid
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np


class CompressionMethod(Enum):
    """Available compression methods."""

    SVD = "svd"
    HOSVD = "hosvd"  # Higher-order SVD
    TT = "tt"  # Tensor Train
    TUCKER = "tucker"
    HIERARCHICAL = "hierarchical"


@dataclass
class CompressionConfig:
    """Configuration for AHTC compression.

    Attributes:
        method: Compression method to use
        target_rank: Target rank for truncation
        relative_error: Target relative error (alternative to rank)
        max_rank: Maximum rank to retain
        min_singular_value: Minimum singular value cutoff
        adaptive: Use adaptive rank selection
        verify_compression: Verify compression quality
    """

    method: Literal["svd", "hosvd", "tt", "tucker", "hierarchical"] = "svd"
    target_rank: int | None = None
    relative_error: float = 1e-6
    max_rank: int = 1000
    min_singular_value: float = 1e-12
    adaptive: bool = True
    verify_compression: bool = True


@dataclass
class CompressionResult:
    """Result of tensor compression.

    Attributes:
        compressed_data: Compressed representation
        compression_ratio: Ratio of original to compressed size
        actual_error: Actual reconstruction error
        rank_used: Rank after truncation
        execution_time: Time for compression
        method_used: Compression method used
        execution_id: Unique identifier
    """

    compressed_data: Any
    compression_ratio: float
    actual_error: float
    rank_used: int
    execution_time: float
    method_used: str
    execution_id: str = ""

    def __post_init__(self) -> None:
        if not self.execution_id:
            self.execution_id = str(uuid.uuid4())


class AHTCAccelerator:
    """AHTC Accelerator for quantum-inspired computations.

    This accelerator provides tensor compression and acceleration
    capabilities for quantum-inspired algorithms.

    Key operations:
    - Compress high-dimensional tensors
    - Accelerate matrix-vector products
    - Reduce memory for iterative algorithms
    - Track provenance of compressed results

    Example:
        >>> accelerator = AHTCAccelerator(CompressionConfig(target_rank=50))
        >>> # Compress large matrix
        >>> result = accelerator.compress(large_matrix)
        >>> print(f"Compression ratio: {result.compression_ratio:.2f}x")
    """

    def __init__(self, config: CompressionConfig):
        """Initialize AHTC accelerator.

        Args:
            config: Compression configuration
        """
        self.config = config
        self._compression_history: list[CompressionResult] = []

    def compress(
        self,
        tensor: np.ndarray,
        target_rank: int | None = None,
    ) -> CompressionResult:
        """Compress tensor using configured method.

        Args:
            tensor: Tensor to compress
            target_rank: Override target rank

        Returns:
            CompressionResult with compressed data
        """
        start_time = time.time()

        rank = target_rank or self.config.target_rank

        if self.config.method == "svd":
            result = self._compress_svd(tensor, rank)
        elif self.config.method == "hosvd":
            result = self._compress_hosvd(tensor, rank)
        elif self.config.method == "tt":
            result = self._compress_tt(tensor, rank)
        elif self.config.method == "tucker":
            result = self._compress_tucker(tensor, rank)
        elif self.config.method == "hierarchical":
            result = self._compress_hierarchical(tensor, rank)
        else:
            raise ValueError(f"Unknown compression method: {self.config.method}")

        result.execution_time = time.time() - start_time
        result.method_used = self.config.method

        # Verify if requested
        if self.config.verify_compression:
            self._verify_compression(tensor, result)

        self._compression_history.append(result)
        return result

    def decompress(self, result: CompressionResult) -> np.ndarray:
        """Decompress compressed tensor.

        Args:
            result: CompressionResult from compress()

        Returns:
            Reconstructed tensor
        """
        data = result.compressed_data

        if result.method_used == "svd":
            U, s, Vh = data["U"], data["s"], data["Vh"]
            return U @ np.diag(s) @ Vh

        elif result.method_used == "tt":
            # Reconstruct from TT cores
            cores = data["cores"]
            tensor = cores[0]
            for core in cores[1:]:
                tensor = np.tensordot(tensor, core, axes=(-1, 0))
            return tensor.reshape(data["original_shape"])

        elif result.method_used == "tucker":
            core = data["core"]
            factors = data["factors"]
            # Tucker decomposition reconstruction
            tensor = core
            for i, factor in enumerate(factors):
                tensor = np.tensordot(tensor, factor, axes=(0, 1))
            return tensor

        elif result.method_used in ["hosvd", "hierarchical"]:
            # Fallback to stored reconstruction
            if "reconstructed" in data:
                return data["reconstructed"]
            return data.get("tensor", np.zeros(1))

        else:
            warnings.warn(
                f"Unknown method {result.method_used}, cannot decompress",
                UserWarning,
                stacklevel=2,
            )
            return np.zeros(1)

    def accelerated_matvec(
        self,
        compressed_matrix: CompressionResult,
        vector: np.ndarray,
    ) -> np.ndarray:
        """Compute matrix-vector product using compressed representation.

        For low-rank matrices, this is O(rank * n) instead of O(n²).

        Args:
            compressed_matrix: Compressed matrix from compress()
            vector: Vector to multiply

        Returns:
            Result of matrix-vector product
        """
        data = compressed_matrix.compressed_data

        if compressed_matrix.method_used == "svd":
            U, s, Vh = data["U"], data["s"], data["Vh"]
            # y = U @ diag(s) @ Vh @ x
            # Compute right-to-left for efficiency
            temp = Vh @ vector
            temp = s * temp
            return U @ temp

        else:
            # Fall back to decompression
            matrix = self.decompress(compressed_matrix)
            return matrix @ vector

    def estimate_memory_savings(
        self,
        original_shape: tuple[int, ...],
        rank: int,
    ) -> dict[str, Any]:
        """Estimate memory savings from compression.

        Args:
            original_shape: Shape of original tensor
            rank: Target rank

        Returns:
            Dictionary with memory analysis
        """
        original_size = np.prod(original_shape)

        if len(original_shape) == 2:
            # Matrix SVD
            m, n = original_shape
            compressed_size = m * rank + rank + rank * n
        else:
            # Higher-order tensor
            compressed_size = rank * sum(original_shape)

        return {
            "original_size": int(original_size),
            "compressed_size": int(compressed_size),
            "compression_ratio": float(original_size / compressed_size),
            "memory_reduction_percent": float(100 * (1 - compressed_size / original_size)),
        }

    def _compress_svd(
        self,
        tensor: np.ndarray,
        target_rank: int | None,
    ) -> CompressionResult:
        """Compress using truncated SVD."""
        if tensor.ndim != 2:
            # Reshape to matrix
            original_shape = tensor.shape
            tensor = tensor.reshape(tensor.shape[0], -1)
        else:
            original_shape = tensor.shape

        U, s, Vh = np.linalg.svd(tensor, full_matrices=False)

        # Determine rank
        if target_rank is not None:
            rank = min(target_rank, len(s))
        elif self.config.adaptive:
            # Adaptive: keep singular values above threshold
            total_sq = np.sum(s**2)
            cumsum_sq = np.cumsum(s**2)
            relative_error = 1 - cumsum_sq / total_sq
            rank = np.searchsorted(-relative_error, -self.config.relative_error) + 1
            rank = min(rank, self.config.max_rank)
        else:
            rank = min(self.config.max_rank, len(s))

        # Truncate
        U_trunc = U[:, :rank]
        s_trunc = s[:rank]
        Vh_trunc = Vh[:rank, :]

        # Compute error
        actual_error = np.sqrt(np.sum(s[rank:] ** 2)) / np.sqrt(np.sum(s**2)) if len(s) > rank else 0.0

        # Compute compression ratio
        original_size = np.prod(original_shape)
        compressed_size = U_trunc.size + s_trunc.size + Vh_trunc.size
        compression_ratio = original_size / compressed_size

        return CompressionResult(
            compressed_data={
                "U": U_trunc,
                "s": s_trunc,
                "Vh": Vh_trunc,
                "original_shape": original_shape,
            },
            compression_ratio=compression_ratio,
            actual_error=actual_error,
            rank_used=rank,
            execution_time=0.0,
            method_used="svd",
        )

    def _compress_hosvd(
        self,
        tensor: np.ndarray,
        target_rank: int | None,
    ) -> CompressionResult:
        """Compress using Higher-Order SVD (Tucker decomposition)."""
        if tensor.ndim < 2:
            return self._compress_svd(tensor.reshape(-1, 1), target_rank)

        original_shape = tensor.shape
        rank = target_rank or self.config.max_rank

        # Compute factor matrices via SVD of each unfolding
        factors = []
        core = tensor

        for mode in range(tensor.ndim):
            # Mode-n unfolding
            unfolded = self._unfold(tensor, mode)

            # SVD
            U, s, _ = np.linalg.svd(unfolded, full_matrices=False)

            # Truncate
            mode_rank = min(rank, len(s), tensor.shape[mode])
            factors.append(U[:, :mode_rank])

            # Update core (mode-n product with U.T)
            core = self._mode_n_product(core, U[:, :mode_rank].T, mode)

        # Compute error (approximate)
        reconstructed = core
        for mode, factor in enumerate(factors):
            reconstructed = self._mode_n_product(reconstructed, factor, mode)

        actual_error = np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)

        # Compression ratio
        original_size = np.prod(original_shape)
        compressed_size = np.prod(core.shape) + sum(f.size for f in factors)
        compression_ratio = original_size / compressed_size

        return CompressionResult(
            compressed_data={
                "core": core,
                "factors": factors,
                "original_shape": original_shape,
            },
            compression_ratio=compression_ratio,
            actual_error=actual_error,
            rank_used=rank,
            execution_time=0.0,
            method_used="hosvd",
        )

    def _compress_tt(
        self,
        tensor: np.ndarray,
        target_rank: int | None,
    ) -> CompressionResult:
        """Compress using Tensor Train decomposition."""
        original_shape = tensor.shape
        rank = target_rank or self.config.max_rank

        # TT-SVD algorithm
        cores = []
        remaining = tensor.flatten()
        remaining_shape = list(original_shape)

        n_dims = len(original_shape)
        r_prev = 1

        for i in range(n_dims - 1):
            # Reshape remaining tensor
            rows = r_prev * remaining_shape[0]
            cols = np.prod(remaining_shape[1:])

            mat = remaining.reshape(rows, cols)

            # SVD
            U, s, Vh = np.linalg.svd(mat, full_matrices=False)

            # Truncate
            r_new = min(rank, len(s))

            # Core for dimension i
            core = U[:, :r_new].reshape(r_prev, remaining_shape[0], r_new)
            cores.append(core)

            # Update remaining
            remaining = (np.diag(s[:r_new]) @ Vh[:r_new, :]).flatten()
            remaining_shape = remaining_shape[1:]
            r_prev = r_new

        # Last core
        cores.append(remaining.reshape(r_prev, remaining_shape[0], 1))

        # Compute error
        reconstructed = cores[0]
        for core in cores[1:]:
            reconstructed = np.tensordot(reconstructed, core, axes=(-1, 0))
        reconstructed = reconstructed.reshape(original_shape)
        actual_error = np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)

        # Compression ratio
        original_size = np.prod(original_shape)
        compressed_size = sum(c.size for c in cores)
        compression_ratio = original_size / max(compressed_size, 1)

        return CompressionResult(
            compressed_data={
                "cores": cores,
                "original_shape": original_shape,
            },
            compression_ratio=compression_ratio,
            actual_error=actual_error,
            rank_used=rank,
            execution_time=0.0,
            method_used="tt",
        )

    def _compress_tucker(
        self,
        tensor: np.ndarray,
        target_rank: int | None,
    ) -> CompressionResult:
        """Compress using Tucker decomposition."""
        # Tucker is same as HOSVD for our purposes
        return self._compress_hosvd(tensor, target_rank)

    def _compress_hierarchical(
        self,
        tensor: np.ndarray,
        target_rank: int | None,
    ) -> CompressionResult:
        """Compress using hierarchical decomposition."""
        # For now, use recursive SVD
        # Full hierarchical implementation planned for Phase 2

        if tensor.ndim <= 2:
            return self._compress_svd(tensor, target_rank)

        # Split tensor in half and compress recursively
        mid = tensor.shape[0] // 2
        left = tensor[:mid]
        right = tensor[mid:]

        left_result = self._compress_svd(left.reshape(left.shape[0], -1), target_rank)
        right_result = self._compress_svd(right.reshape(right.shape[0], -1), target_rank)

        # Combined compression ratio
        original_size = tensor.size

        # Calculate compressed sizes for both parts
        left_arrays = [v for v in left_result.compressed_data.values() if isinstance(v, np.ndarray)]
        right_arrays = [v for v in right_result.compressed_data.values() if isinstance(v, np.ndarray)]
        compressed_size = sum(arr.size for arr in left_arrays) + sum(arr.size for arr in right_arrays)

        return CompressionResult(
            compressed_data={
                "left": left_result.compressed_data,
                "right": right_result.compressed_data,
                "split_index": mid,
                "original_shape": tensor.shape,
            },
            compression_ratio=original_size / max(compressed_size, 1),
            actual_error=max(left_result.actual_error, right_result.actual_error),
            rank_used=target_rank or self.config.max_rank,
            execution_time=0.0,
            method_used="hierarchical",
        )

    def _unfold(self, tensor: np.ndarray, mode: int) -> np.ndarray:
        """Unfold tensor along specified mode."""
        return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)

    def _mode_n_product(
        self,
        tensor: np.ndarray,
        matrix: np.ndarray,
        mode: int,
    ) -> np.ndarray:
        """Compute mode-n product of tensor with matrix."""
        return np.tensordot(tensor, matrix, axes=(mode, 1)).swapaxes(mode, -1)

    def _verify_compression(
        self,
        original: np.ndarray,
        result: CompressionResult,
    ) -> None:
        """Verify compression quality."""
        reconstructed = self.decompress(result)

        actual_error = np.linalg.norm(original - reconstructed) / np.linalg.norm(original)

        if actual_error > self.config.relative_error * 10:
            warnings.warn(
                f"Compression error {actual_error:.6f} exceeds threshold",
                UserWarning,
                stacklevel=3,
            )

    def get_compression_history(self) -> list[CompressionResult]:
        """Get history of compression operations.

        Returns:
            List of CompressionResult objects
        """
        return self._compression_history.copy()
