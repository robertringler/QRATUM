"""Adapter for converting BioMechanism to visualization format."""

from __future__ import annotations

import numpy as np

from qubic.visualization.core.data_model import VisualizationData
from xenon.core.mechanism import BioMechanism


class BioMechanismAdapter:
    """Adapter to convert BioMechanism DAG into visualization format.

    Converts biochemical reaction networks into graph/network visualizations
    where nodes represent molecular states and edges represent transitions.
    """

    def __init__(self, mechanism: BioMechanism):
        """Initialize adapter with a bio-mechanism.

        Args:
            mechanism: BioMechanism instance to adapt
        """

        self.mechanism = mechanism

    def to_viz_model(self) -> dict[str, any]:
        """Convert BioMechanism DAG into graph data for visualization.

        Returns:
            Dictionary with 'nodes' and 'edges' suitable for network visualization
        """

        nodes = []
        edges = []

        for state in self.mechanism.states:
            nodes.append(
                {
                    "id": state.state_id,
                    "protein": state.protein_name,
                    "free_energy": state.free_energy,
                    "concentration": state.concentration,
                }
            )

        for t in self.mechanism.transitions:
            edges.append(
                {
                    "source": t.source_state,
                    "target": t.target_state,
                    "rate": t.rate_constant,
                    "delta_g": t.delta_g,
                    "activation_energy": t.activation_energy,
                }
            )

        return {"nodes": nodes, "edges": edges, "evidence_score": self.mechanism.evidence_score}

    def to_3d_network(self, layout: str = "spring", scale: float = 10.0) -> VisualizationData:
        """Convert BioMechanism to 3D network visualization.

        Creates a 3D spatial layout of the mechanism network suitable for
        rendering with the visualization pipeline.

        Args:
            layout: Layout algorithm ('spring', 'circular', 'hierarchical')
            scale: Spatial scale factor for node positions

        Returns:
            VisualizationData with network geometry
        """

        num_states = len(self.mechanism.states)

        # Generate 3D positions for nodes based on layout
        if layout == "spring":
            positions = self._spring_layout_3d(num_states, scale)
        elif layout == "circular":
            positions = self._circular_layout_3d(num_states, scale)
        elif layout == "hierarchical":
            positions = self._hierarchical_layout_3d(scale)
        else:
            raise ValueError(f"Unknown layout: {layout}")

        # Create vertices (nodes as small spheres)
        vertices = positions

        # Create edges as line segments (each edge = 2 vertices)
        edge_vertices = []
        for t in self.mechanism.transitions:
            src_idx = self._get_state_index(t.source_state)
            tgt_idx = self._get_state_index(t.target_state)
            if src_idx is not None and tgt_idx is not None:
                edge_vertices.append(positions[src_idx])
                edge_vertices.append(positions[tgt_idx])

        # For visualization, create dummy faces (triangles) for node spheres
        # In a full implementation, this would create actual sphere meshes
        faces = self._create_dummy_faces(num_states)

        # Add scalar fields for coloring
        scalar_fields = {
            "free_energy": np.array([s.free_energy for s in self.mechanism.states]),
            "concentration": np.array([s.concentration for s in self.mechanism.states]),
        }

        return VisualizationData(
            vertices=vertices,
            faces=faces,
            scalar_fields=scalar_fields,
            metadata={
                "mechanism_id": self.mechanism.mechanism_id,
                "evidence_score": self.mechanism.evidence_score,
                "num_states": num_states,
                "num_transitions": len(self.mechanism.transitions),
            },
        )

    def _get_state_index(self, state_id: str) -> int | None:
        """Get index of a state by ID.

        Args:
            state_id: State identifier

        Returns:
            Index of the state, or None if not found
        """

        for i, state in enumerate(self.mechanism.states):
            if state.state_id == state_id:
                return i
        return None

    def _spring_layout_3d(self, num_nodes: int, scale: float) -> np.ndarray:
        """Generate 3D spring layout positions.

        Args:
            num_nodes: Number of nodes
            scale: Spatial scale

        Returns:
            (N, 3) array of node positions
        """

        # Simple spring layout: random positions with slight structure
        np.random.seed(42)  # Deterministic for testing
        positions = np.random.randn(num_nodes, 3) * scale
        return positions

    def _circular_layout_3d(self, num_nodes: int, scale: float) -> np.ndarray:
        """Generate circular layout in 3D.

        Args:
            num_nodes: Number of nodes
            scale: Spatial scale

        Returns:
            (N, 3) array of node positions
        """

        angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
        positions = np.zeros((num_nodes, 3))
        positions[:, 0] = scale * np.cos(angles)
        positions[:, 1] = scale * np.sin(angles)
        positions[:, 2] = 0  # Flat circle in XY plane
        return positions

    def _hierarchical_layout_3d(self, scale: float) -> np.ndarray:
        """Generate hierarchical layout based on transition structure.

        Args:
            scale: Spatial scale

        Returns:
            (N, 3) array of node positions
        """

        # Simple hierarchical: assign levels based on connectivity
        num_nodes = len(self.mechanism.states)
        positions = np.zeros((num_nodes, 3))

        # Assign levels (simple heuristic: count incoming transitions)
        for i, state in enumerate(self.mechanism.states):
            level = len(self.mechanism.get_transitions_to(state.state_id))
            positions[i] = [i * scale / num_nodes, level * scale, 0]

        return positions

    def _create_dummy_faces(self, num_nodes: int) -> np.ndarray:
        """Create dummy face array for nodes.

        In a full implementation, this would create proper sphere meshes.

        Args:
            num_nodes: Number of nodes

        Returns:
            (M, 3) array of face indices
        """

        # Create minimal dummy triangles (just to satisfy VisualizationData)
        # Each node gets one triangle (not realistic but sufficient for placeholder)
        if num_nodes < 3:
            # Need at least 3 vertices for a triangle
            return np.array([[0, 0, 0]])

        faces = []
        for i in range(num_nodes - 2):
            faces.append([i, i + 1, i + 2])
        return np.array(faces)
