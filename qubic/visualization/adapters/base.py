"""Base adapter interface for simulation data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from qubic.visualization.core.data_model import VisualizationData


class SimulationAdapter(ABC):
    """Abstract base class for simulation data adapters.

    Adapters convert simulation-specific data formats into the unified
    VisualizationData format for rendering.
    """

    @abstractmethod
    def load_data(self, source: Any) -> VisualizationData:
        """Load simulation data and convert to visualization format.

        Args:
            source: Simulation data source (file path, data structure, etc.)

        Returns:
            VisualizationData object ready for rendering

        Raises:
            ValueError: If data cannot be loaded or converted
        """

        pass

    @abstractmethod
    def validate_source(self, source: Any) -> bool:
        """Validate that the source can be loaded by this adapter.

        Args:
            source: Data source to validate

        Returns:
            True if source is valid, False otherwise
        """

        pass
