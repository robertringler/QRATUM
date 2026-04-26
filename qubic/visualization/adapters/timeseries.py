"""Time-series simulation adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from qubic.visualization.adapters.base import SimulationAdapter
from qubic.visualization.core.data_model import VisualizationData


class TimeSeriesAdapter(SimulationAdapter):
    """Adapter for time-dependent simulation data.

    Handles sequences of simulation states over time, useful for
    animations and transient analysis.
    """

    def __init__(self) -> None:
        """Initialize time-series adapter."""

        self.timesteps: list[VisualizationData] = []
        self.times: list[float] = []

    def load_data(
        self, source: str | Path | dict[str, Any] | list[dict[str, Any]]
    ) -> VisualizationData:
        """Load time-series data.

        Args:
            source: File path, dictionary, or list of dictionaries with time-series data

        Returns:
            VisualizationData for the first timestep

        Raises:
            ValueError: If source format is invalid
        """

        if isinstance(source, (str, Path)):
            return self._load_from_file(Path(source))
        elif isinstance(source, list):
            return self._load_from_list(source)
        elif isinstance(source, dict):
            # Single timestep
            return self._load_single_timestep(source)
        else:
            raise ValueError(
                f"Unsupported source type: {type(source)}. Expected file path, dictionary, or list."
            )

    def _load_from_file(self, path: Path) -> VisualizationData:
        """Load time-series from file.

        Args:
            path: Path to time-series file

        Returns:
            VisualizationData for first timestep

        Raises:
            FileNotFoundError: If file doesn't exist
        """

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        raise NotImplementedError("File-based loading not yet implemented. Use list input.")

    def _load_from_list(self, data_list: list[dict[str, Any]]) -> VisualizationData:
        """Load time-series from list of dictionaries.

        Args:
            data_list: List of dictionaries, each representing one timestep

        Returns:
            VisualizationData for first timestep
        """

        self.timesteps = []
        self.times = []

        for i, data in enumerate(data_list):
            vis_data = self._load_single_timestep(data)
            self.timesteps.append(vis_data)

            # Extract time if available, otherwise use index
            time = data.get("time", float(i))
            self.times.append(time)

        if not self.timesteps:
            raise ValueError("No timesteps loaded")

        return self.timesteps[0]

    def _load_single_timestep(self, data: dict[str, Any]) -> VisualizationData:
        """Load single timestep.

        Args:
            data: Dictionary with mesh and field data

        Returns:
            VisualizationData object
        """

        if "vertices" not in data or "faces" not in data:
            raise ValueError("Each timestep must contain 'vertices' and 'faces'")

        vertices = np.asarray(data["vertices"])
        faces = np.asarray(data["faces"])

        # Extract fields
        scalar_fields = {}
        vector_fields = {}

        for key, value in data.items():
            if key in ("vertices", "faces", "normals", "time"):
                continue

            field_data = np.asarray(value)

            if field_data.ndim == 1:
                scalar_fields[key] = field_data
            elif field_data.ndim == 2 and field_data.shape[1] == 3:
                vector_fields[key] = field_data

        metadata = {
            "adapter_type": "timeseries",
            "time": data.get("time", 0.0),
        }

        return VisualizationData(
            vertices=vertices,
            faces=faces,
            scalar_fields=scalar_fields,
            vector_fields=vector_fields,
            metadata=metadata,
        )

    def get_timestep(self, index: int) -> VisualizationData:
        """Get data for specific timestep.

        Args:
            index: Timestep index

        Returns:
            VisualizationData for the timestep

        Raises:
            IndexError: If index is out of range
        """

        if not self.timesteps:
            raise RuntimeError("No timesteps loaded. Call load_data first.")

        if index < 0 or index >= len(self.timesteps):
            raise IndexError(f"Timestep index {index} out of range [0, {len(self.timesteps)})")

        return self.timesteps[index]

    def get_num_timesteps(self) -> int:
        """Get number of timesteps.

        Returns:
            Number of loaded timesteps
        """

        return len(self.timesteps)

    def get_time_range(self) -> tuple[float, float]:
        """Get time range of the simulation.

        Returns:
            Tuple of (start_time, end_time)

        Raises:
            RuntimeError: If no timesteps are loaded
        """

        if not self.times:
            raise RuntimeError("No timesteps loaded")

        return min(self.times), max(self.times)

    def validate_source(self, source: Any) -> bool:
        """Validate time-series data source.

        Args:
            source: Data source to validate

        Returns:
            True if source is valid
        """

        if isinstance(source, (str, Path)):
            return Path(source).exists()
        elif isinstance(source, list):
            # Validate first element
            if source and isinstance(source[0], dict):
                return "vertices" in source[0] and "faces" in source[0]
            return False
        elif isinstance(source, dict):
            return "vertices" in source and "faces" in source
        return False

    def create_synthetic_timeseries(
        self, n_steps: int = 10, mesh_type: str = "deforming_sphere"
    ) -> VisualizationData:
        """Create synthetic time-series for testing.

        Args:
            n_steps: Number of timesteps
            mesh_type: Type of animation

        Returns:
            VisualizationData for first timestep
        """

        data_list = []

        for i in range(n_steps):
            t = i / (n_steps - 1)  # Normalized time [0, 1]

            if mesh_type == "deforming_sphere":
                # Sphere that pulses
                resolution = 20
                phi = np.linspace(0, np.pi, resolution)
                theta = np.linspace(0, 2 * np.pi, resolution)
                phi, theta = np.meshgrid(phi, theta)

                # Radius varies with time
                radius = 1.0 + 0.2 * np.sin(2 * np.pi * t)

                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.sin(phi) * np.sin(theta)
                z = radius * np.cos(phi)

                vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

                # Generate faces (same for all timesteps)
                if i == 0:
                    faces = []
                    for ii in range(resolution - 1):
                        for jj in range(resolution - 1):
                            v0 = ii * resolution + jj
                            v1 = ii * resolution + (jj + 1)
                            v2 = (ii + 1) * resolution + (jj + 1)
                            v3 = (ii + 1) * resolution + jj
                            faces.append([v0, v1, v2])
                            faces.append([v0, v2, v3])
                    faces = np.array(faces)

                # Synthetic field that varies with time
                field_value = np.sin(2 * np.pi * t + vertices[:, 2])

                data_list.append(
                    {
                        "vertices": vertices,
                        "faces": faces,
                        "time": t,
                        "field": field_value,
                    }
                )

        return self._load_from_list(data_list)
