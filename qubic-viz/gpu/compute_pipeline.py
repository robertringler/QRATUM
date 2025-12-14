"""GPU compute pipeline orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .kernels import GPUKernels
from .memory_manager import GPUMemoryManager


class ComputePipeline:
    """Orchestrate GPU compute operations.

    Args:
        device: Compute device (cuda, cpu)
        memory_limit_mb: Memory limit in MB
    """

    def __init__(self, device: str = "cpu", memory_limit_mb: int = 4096) -> None:
        """Initialize compute pipeline."""
        self.device = device
        self.kernels = GPUKernels(device)
        self.memory_manager = GPUMemoryManager(memory_limit_mb)
        self.operations = []

    def add_operation(self, operation: Callable, name: str, inputs: list | None = None) -> None:
        """Add operation to pipeline.

        Args:
            operation: Operation function to execute
            name: Operation name
            inputs: Input dependencies
        """
        self.operations.append({"operation": operation, "name": name, "inputs": inputs or []})

    def execute(self, initial_data: dict[str, Any]) -> dict[str, Any]:
        """Execute pipeline.

        Args:
            initial_data: Initial data dictionary

        Returns:
            Results dictionary
        """
        results = initial_data.copy()

        for op in self.operations:
            # Get input data
            inputs = [results.get(inp) for inp in op["inputs"]]

            # Execute operation
            try:
                output = op["operation"](*inputs)
                results[op["name"]] = output
            except Exception as e:
                print(f"Operation {op['name']} failed: {e}")
                results[op["name"]] = None

        return results

    def clear(self) -> None:
        """Clear pipeline operations."""
        self.operations = []

    def optimize_mesh_rendering(
        self, vertices: np.ndarray, faces: np.ndarray, field_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Optimize mesh for rendering.

        Args:
            vertices: Mesh vertices
            faces: Mesh faces
            field_data: Field values at vertices

        Returns:
            Tuple of (optimized_vertices, optimized_faces, optimized_field)
        """
        # Check memory constraints
        vertex_memory = vertices.nbytes
        face_memory = faces.nbytes
        field_memory = field_data.nbytes
        total_memory = vertex_memory + face_memory + field_memory

        if not self.memory_manager.can_allocate(total_memory):
            # Decimate mesh
            decimation_factor = 0.5
            stride = int(1.0 / decimation_factor)

            # Subsample vertices
            vertex_indices = np.arange(0, len(vertices), stride)
            optimized_vertices = vertices[vertex_indices]
            optimized_field = field_data[vertex_indices]

            # Rebuild faces
            index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(vertex_indices)}
            optimized_faces = []
            for face in faces:
                if all(v in index_map for v in face):
                    new_face = [index_map[v] for v in face]
                    optimized_faces.append(new_face)

            optimized_faces = np.array(optimized_faces)
        else:
            optimized_vertices = vertices
            optimized_faces = faces
            optimized_field = field_data

        return optimized_vertices, optimized_faces, optimized_field

    def batch_process_fields(
        self, field_list: list[np.ndarray], operation: Callable
    ) -> list[np.ndarray]:
        """Batch process multiple fields.

        Args:
            field_list: List of field arrays
            operation: Operation to apply to each field

        Returns:
            List of processed fields
        """
        results = []

        for field in field_list:
            try:
                result = operation(field)
                results.append(result)
            except Exception as e:
                print(f"Field processing failed: {e}")
                results.append(None)

        return results
