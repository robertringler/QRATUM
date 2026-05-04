"""GPU compute kernels with CPU fallback."""

from __future__ import annotations

import numpy as np


class GPUKernels:
    """GPU acceleration kernels with CPU fallback.

    Args:
        device: Compute device (cuda, cpu)
    """

    def __init__(self, device: str = "cpu") -> None:
        """Initialize GPU kernels."""

        self.device = device
        self.cuda_available = self._check_cuda()

        if device == "cuda" and not self.cuda_available:
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"

    def _check_cuda(self) -> bool:
        """Check if CUDA is available.

        Returns:
            True if CUDA is available
        """

        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def mesh_deformation_kernel(
        self,
        vertices: np.ndarray,
        forces: np.ndarray,
        stiffness: float,
    ) -> np.ndarray:
        """Apply deformation to mesh vertices.

        Args:
            vertices: Vertex positions (N, 3)
            forces: Forces at each vertex (N, 3)
            stiffness: Material stiffness

        Returns:
            Deformed vertex positions
        """

        if self.device == "cuda" and self.cuda_available:
            return self._mesh_deformation_cuda(vertices, forces, stiffness)
        else:
            return self._mesh_deformation_cpu(vertices, forces, stiffness)

    def _mesh_deformation_cuda(
        self, vertices: np.ndarray, forces: np.ndarray, stiffness: float
    ) -> np.ndarray:
        """CUDA implementation of mesh deformation.

        Args:
            vertices: Vertex positions
            forces: Forces at vertices
            stiffness: Material stiffness

        Returns:
            Deformed vertices
        """

        try:
            import torch

            # Convert to torch tensors
            v_tensor = torch.from_numpy(vertices).cuda()
            f_tensor = torch.from_numpy(forces).cuda()

            # Apply deformation
            displacement = f_tensor / stiffness
            deformed = v_tensor + displacement

            return deformed.cpu().numpy()
        except Exception as e:
            print(f"CUDA deformation failed: {e}, falling back to CPU")
            return self._mesh_deformation_cpu(vertices, forces, stiffness)

    def _mesh_deformation_cpu(
        self, vertices: np.ndarray, forces: np.ndarray, stiffness: float
    ) -> np.ndarray:
        """CPU implementation of mesh deformation.

        Args:
            vertices: Vertex positions
            forces: Forces at vertices
            stiffness: Material stiffness

        Returns:
            Deformed vertices
        """

        displacement = forces / stiffness
        return vertices + displacement

    def field_interpolation_kernel(
        self,
        positions: np.ndarray,
        field_values: np.ndarray,
        query_positions: np.ndarray,
    ) -> np.ndarray:
        """Interpolate field values at query positions.

        Args:
            positions: Known positions (N, 3)
            field_values: Field values at known positions (N,)
            query_positions: Query positions (M, 3)

        Returns:
            Interpolated field values (M,)
        """

        if self.device == "cuda" and self.cuda_available:
            return self._field_interpolation_cuda(positions, field_values, query_positions)
        else:
            return self._field_interpolation_cpu(positions, field_values, query_positions)

    def _field_interpolation_cuda(
        self,
        positions: np.ndarray,
        field_values: np.ndarray,
        query_positions: np.ndarray,
    ) -> np.ndarray:
        """CUDA implementation of field interpolation.

        Args:
            positions: Known positions
            field_values: Field values
            query_positions: Query positions

        Returns:
            Interpolated values
        """

        try:
            import torch

            pos_tensor = torch.from_numpy(positions).cuda()
            val_tensor = torch.from_numpy(field_values).cuda()
            query_tensor = torch.from_numpy(query_positions).cuda()

            # Simple nearest neighbor interpolation
            # Compute distances
            diff = query_tensor.unsqueeze(1) - pos_tensor.unsqueeze(0)
            distances = torch.norm(diff, dim=2)

            # Find nearest neighbors
            nearest_idx = torch.argmin(distances, dim=1)
            interpolated = val_tensor[nearest_idx]

            return interpolated.cpu().numpy()
        except Exception as e:
            print(f"CUDA interpolation failed: {e}, falling back to CPU")
            return self._field_interpolation_cpu(positions, field_values, query_positions)

    def _field_interpolation_cpu(
        self,
        positions: np.ndarray,
        field_values: np.ndarray,
        query_positions: np.ndarray,
    ) -> np.ndarray:
        """CPU implementation of field interpolation.

        Args:
            positions: Known positions
            field_values: Field values
            query_positions: Query positions

        Returns:
            Interpolated values
        """

        interpolated = np.zeros(len(query_positions))

        for i, query_pos in enumerate(query_positions):
            # Compute distances to all known positions
            distances = np.linalg.norm(positions - query_pos, axis=1)

            # Find nearest neighbor
            nearest_idx = np.argmin(distances)
            interpolated[i] = field_values[nearest_idx]

        return interpolated

    def thermal_diffusion_kernel(
        self,
        temperature: np.ndarray,
        conductivity: float,
        dt: float,
    ) -> np.ndarray:
        """Simulate thermal diffusion.

        Args:
            temperature: Temperature field
            conductivity: Thermal conductivity
            dt: Time step

        Returns:
            Updated temperature field
        """

        if self.device == "cuda" and self.cuda_available:
            return self._thermal_diffusion_cuda(temperature, conductivity, dt)
        else:
            return self._thermal_diffusion_cpu(temperature, conductivity, dt)

    def _thermal_diffusion_cuda(
        self, temperature: np.ndarray, conductivity: float, dt: float
    ) -> np.ndarray:
        """CUDA implementation of thermal diffusion.

        Args:
            temperature: Temperature field
            conductivity: Thermal conductivity
            dt: Time step

        Returns:
            Updated temperature
        """

        try:
            import torch

            temp_tensor = torch.from_numpy(temperature).cuda()

            # Simple diffusion (Laplacian)
            # This is a simplified version; full implementation would use proper stencils
            laplacian = torch.zeros_like(temp_tensor)
            diffusion = conductivity * dt * laplacian

            return (temp_tensor + diffusion).cpu().numpy()
        except Exception as e:
            print(f"CUDA diffusion failed: {e}, falling back to CPU")
            return self._thermal_diffusion_cpu(temperature, conductivity, dt)

    def _thermal_diffusion_cpu(
        self, temperature: np.ndarray, conductivity: float, dt: float
    ) -> np.ndarray:
        """CPU implementation of thermal diffusion.

        Args:
            temperature: Temperature field
            conductivity: Thermal conductivity
            dt: Time step

        Returns:
            Updated temperature

        Note:
            This is a simplified implementation without spatial derivatives.
            For production use, implement proper finite difference stencils.
        """

        # Simplified diffusion (no spatial derivatives for now)
        return temperature
