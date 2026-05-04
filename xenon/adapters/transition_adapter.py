"""Adapter for converting Transition events to visualization format."""

from __future__ import annotations

import numpy as np

from qubic.visualization.core.data_model import VisualizationData
from xenon.core.mechanism import Transition

# Constants for energy barrier calculation
BARRIER_PEAK_POSITION = 0.5  # Position of energy barrier maximum (0-1)
BARRIER_WIDTH = 0.1  # Width parameter for Gaussian barrier

# Constants for thickness calculation
MIN_RATE_LOG = 1e-10  # Minimum rate constant for log scaling
LOG_OFFSET = 10  # Offset for log normalization
LOG_SCALE = 20  # Scale factor for log normalization


class TransitionAdapter:
    """Adapter to convert Transition events to visualization format.

    Converts state transitions into visual representations such as
    arrows, tubes, or energy barrier diagrams.
    """

    def __init__(self, transition: Transition):
        """Initialize adapter with a transition.

        Args:
            transition: Transition instance to adapt
        """

        self.transition = transition

    def to_viz_model(self) -> dict[str, any]:
        """Convert Transition into visualization-ready format.

        Returns:
            Dictionary with transition properties for visualization
        """

        return {
            "source": self.transition.source_state,
            "target": self.transition.target_state,
            "rate": self.transition.rate_constant,
            "delta_g": self.transition.delta_g,
            "activation_energy": self.transition.activation_energy,
            "metadata": self.transition.metadata,
        }

    def to_arrow(
        self,
        source_pos: np.ndarray,
        target_pos: np.ndarray,
        arrow_width: float = 0.1,
    ) -> VisualizationData:
        """Create arrow visualization for the transition.

        Generates a 3D arrow from source to target position.

        Args:
            source_pos: 3D position of source state
            target_pos: 3D position of target state
            arrow_width: Width of the arrow shaft

        Returns:
            VisualizationData with arrow mesh
        """

        direction = target_pos - source_pos
        length = np.linalg.norm(direction)
        direction_norm = direction / length if length > 0 else np.array([0, 0, 1])

        # Create simple arrow as a line with thickness
        # For simplicity, create a cylinder-like structure
        num_segments = 10
        vertices = []

        for i in range(num_segments):
            t = i / (num_segments - 1)
            pos = source_pos + t * direction
            # Add thickness perpendicular to direction
            perp = np.array([-direction_norm[1], direction_norm[0], 0])
            if np.linalg.norm(perp) < 0.1:
                perp = np.array([0, -direction_norm[2], direction_norm[1]])
            perp = perp / np.linalg.norm(perp) * arrow_width

            vertices.append(pos + perp)
            vertices.append(pos - perp)

        vertices = np.array(vertices)

        # Create faces connecting the segments
        faces = []
        for i in range(num_segments - 1):
            v0 = i * 2
            v1 = i * 2 + 1
            v2 = (i + 1) * 2
            v3 = (i + 1) * 2 + 1

            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

        faces = np.array(faces) if faces else np.array([[0, 0, 0]])

        # Add scalar fields for transition properties
        scalar_fields = {
            "rate_constant": np.full(len(vertices), self.transition.rate_constant),
            "delta_g": np.full(len(vertices), self.transition.delta_g),
            "activation_energy": np.full(len(vertices), self.transition.activation_energy),
        }

        return VisualizationData(
            vertices=vertices,
            faces=faces,
            scalar_fields=scalar_fields,
            metadata={
                "source_state": self.transition.source_state,
                "target_state": self.transition.target_state,
                "transition_type": "arrow",
            },
        )

    def to_energy_barrier(self, num_points: int = 50) -> VisualizationData:
        """Create energy barrier diagram for the transition.

        Generates a curve showing the energy profile along the reaction coordinate.

        Args:
            num_points: Number of points along the reaction path

        Returns:
            VisualizationData with energy barrier curve
        """

        # Create reaction coordinate (0 to 1)
        reaction_coord = np.linspace(0, 1, num_points)

        # Model energy barrier as a simple curve
        # E = e0 + ea * exp(-((x-peak)/width)^2) + ΔG * x
        e0 = 0.0  # Initial energy (reference)
        ea = self.transition.activation_energy
        delta_g = self.transition.delta_g

        # Energy along path (Gaussian barrier + linear ΔG)
        energy = (
            e0
            + ea * np.exp(-(((reaction_coord - BARRIER_PEAK_POSITION) / BARRIER_WIDTH) ** 2))
            + delta_g * reaction_coord
        )

        # Create 3D curve (x = reaction coordinate, y = 0, z = energy)
        vertices = np.column_stack([reaction_coord * 10, np.zeros(num_points), energy])

        # Create degenerate faces for line rendering
        faces = []
        for i in range(num_points - 1):
            faces.append([i, i, i + 1])
        faces = np.array(faces) if faces else np.array([[0, 0, 0]])

        # Add scalar fields
        scalar_fields = {
            "energy": energy,
            "reaction_coordinate": reaction_coord,
            "rate_constant": np.full(num_points, self.transition.rate_constant),
        }

        return VisualizationData(
            vertices=vertices,
            faces=faces,
            scalar_fields=scalar_fields,
            metadata={
                "source_state": self.transition.source_state,
                "target_state": self.transition.target_state,
                "transition_type": "energy_barrier",
                "activation_energy": ea,
                "delta_g": delta_g,
            },
        )

    def get_color_by_rate(
        self, min_rate: float = 0.0, max_rate: float = 1.0
    ) -> tuple[float, float, float]:
        """Get RGB color based on transition rate constant.

        Maps rate constant to a color for visualization (slow=blue, fast=red).

        Args:
            min_rate: Minimum rate for color mapping
            max_rate: Maximum rate for color mapping

        Returns:
            RGB tuple (0-1 range)
        """

        # Normalize rate to 0-1 range
        rate_norm = (self.transition.rate_constant - min_rate) / (max_rate - min_rate)
        rate_norm = np.clip(rate_norm, 0, 1)

        # Blue (slow) to Red (fast)
        r = rate_norm
        g = 0.0
        b = 1.0 - rate_norm

        return (r, g, b)

    def get_thickness_by_rate(
        self, min_thickness: float = 0.1, max_thickness: float = 1.0
    ) -> float:
        """Get edge thickness based on transition rate constant.

        Args:
            min_thickness: Minimum thickness value
            max_thickness: Maximum thickness value

        Returns:
            Thickness value for visualization
        """

        # Log scale for rate constants (often span many orders of magnitude)
        log_rate = np.log10(max(self.transition.rate_constant, 1e-10))
        # Normalize to thickness range (heuristic)
        thickness = min_thickness + (max_thickness - min_thickness) * np.clip(
            (log_rate + 10) / 20, 0, 1
        )
        return thickness
