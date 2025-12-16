"""GPU-accelerated rendering backend with CPU fallback."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from qubic.visualization.backends.matplotlib_backend import MatplotlibBackend
from qubic.visualization.core.camera import Camera
from qubic.visualization.core.data_model import VisualizationData

logger = logging.getLogger(__name__)


class GPUBackend:
    """GPU-accelerated rendering with automatic CPU fallback.

    Attempts to use PyTorch with CUDA for GPU acceleration. Falls back
    to CPU-based MatplotlibBackend if GPU is unavailable.
    """

    def __init__(
        self, figsize: tuple[int, int] = (10, 8), dpi: int = 100, force_cpu: bool = False
    ) -> None:
        """Initialize GPU backend.

        Args:
            figsize: Figure size in inches
            dpi: Resolution in dots per inch
            force_cpu: Force CPU fallback even if GPU is available
        """
        self.figsize = figsize
        self.dpi = dpi
        self.use_gpu = False
        self.device = "cpu"

        # Try to initialize GPU
        if not force_cpu:
            self.use_gpu = self._init_gpu()

        if not self.use_gpu:
            logger.info("GPU not available or disabled, using CPU fallback")
            self.backend = MatplotlibBackend(figsize=figsize, dpi=dpi)
        else:
            logger.info(f"GPU rendering enabled on device: {self.device}")
            # Initialize GPU-specific resources
            self.backend = MatplotlibBackend(figsize=figsize, dpi=dpi)

    def _init_gpu(self) -> bool:
        """Initialize GPU support.

        Returns:
            True if GPU is available and initialized successfully
        """
        try:
            import torch

            if torch.cuda.is_available():
                self.device = "cuda"
                self.torch = torch
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                return True
            else:
                logger.info("PyTorch available but CUDA not found")
                return False

        except ImportError:
            logger.info("PyTorch not installed, GPU acceleration unavailable")
            return False

    def render(
        self,
        data: VisualizationData,
        scalar_field: str | None = None,
        camera: Camera | None = None,
        colormap: str = "viridis",
        show_edges: bool = False,
        alpha: float = 1.0,
    ):
        """Render visualization data.

        If GPU is available, preprocessing is done on GPU before CPU rendering.

        Args:
            data: Visualization data to render
            scalar_field: Name of scalar field for color mapping
            camera: Camera settings
            colormap: Matplotlib colormap name
            show_edges: Whether to show mesh edges
            alpha: Transparency

        Returns:
            Matplotlib figure with rendered visualization
        """
        # Preprocess data on GPU if available
        if self.use_gpu:
            data = self._preprocess_gpu(data)

        # Render using backend (currently matplotlib)
        return self.backend.render(
            data=data,
            scalar_field=scalar_field,
            camera=camera,
            colormap=colormap,
            show_edges=show_edges,
            alpha=alpha,
        )

    def _preprocess_gpu(self, data: VisualizationData) -> VisualizationData:
        """Preprocess data on GPU.

        Performs operations like normal computation, field transformations
        on GPU for acceleration.

        Args:
            data: Input visualization data

        Returns:
            Processed visualization data
        """
        # Example: compute normals on GPU
        try:
            vertices_gpu = self.torch.tensor(
                data.vertices, device=self.device, dtype=self.torch.float32
            )
            faces_gpu = self.torch.tensor(data.faces, device=self.device, dtype=self.torch.long)

            # Compute face normals on GPU
            v0 = vertices_gpu[faces_gpu[:, 0]]
            v1 = vertices_gpu[faces_gpu[:, 1]]
            v2 = vertices_gpu[faces_gpu[:, 2]]

            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normals = self.torch.cross(edge1, edge2)

            # Accumulate at vertices
            vertex_normals = self.torch.zeros_like(vertices_gpu)
            vertex_normals.index_add_(0, faces_gpu[:, 0], face_normals)
            vertex_normals.index_add_(0, faces_gpu[:, 1], face_normals)
            vertex_normals.index_add_(0, faces_gpu[:, 2], face_normals)

            # Normalize
            norms = self.torch.norm(vertex_normals, dim=1, keepdim=True)
            norms = self.torch.where(norms > 0, norms, self.torch.ones_like(norms))
            vertex_normals = vertex_normals / norms

            # Update data with GPU-computed normals
            data.normals = vertex_normals.cpu().numpy()

        except Exception as e:
            logger.warning(f"GPU preprocessing failed: {e}, using CPU fallback")

        return data

    def save(self, output_path: Path, **kwargs) -> None:
        """Save rendered figure to file.

        Args:
            output_path: Output file path
            **kwargs: Additional arguments for save
        """
        self.backend.save(output_path, **kwargs)

    def show(self) -> None:
        """Display the rendered figure interactively."""
        self.backend.show()

    def close(self) -> None:
        """Close the backend and free resources."""
        self.backend.close()

        # Clean up GPU resources if applicable
        if self.use_gpu:
            with contextlib.suppress(Exception):
                self.torch.cuda.empty_cache()
