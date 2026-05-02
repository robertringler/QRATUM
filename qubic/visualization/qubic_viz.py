"""QUBIC Unified Visualization Module.

This module provides a high-level API for visualizing simulation data
from multiple sources including XENON bio-mechanisms, quantum states,
and mesh simulations.

Usage:
    python -m qubic.visualization.qubic_viz demo
    python -m qubic.visualization.qubic_viz render --source xenon --output output.png
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class VizConfig:
    """Configuration for visualization rendering."""

    figsize: tuple[int, int] = (12, 10)
    dpi: int = 150
    colormap: str = "viridis"
    style: str = "dark_background"
    title: str = "QUBIC Visualization"
    show_colorbar: bool = True
    show_axes: bool = True
    alpha: float = 0.8


class QubicVisualizer:
    """High-level visualizer for QUBIC/XENON simulation data.

    Supports:
    - XENON bio-mechanism networks
    - Quantum state visualizations
    - 3D mesh/field visualizations
    - Time-series animations
    """

    def __init__(self, config: Optional[VizConfig] = None):
        """Initialize visualizer.

        Args:
            config: Visualization configuration
        """
        self.config = config or VizConfig()
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check required dependencies are available."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Install with: pip install matplotlib")

    def visualize_mechanism(
        self,
        nodes: list[dict],
        edges: list[dict],
        title: str = "Bio-Mechanism Network",
        output_path: Optional[str] = None,
    ) -> Optional[Figure]:
        """Visualize a bio-mechanism network.

        Args:
            nodes: List of node dictionaries with id, protein, free_energy, concentration
            edges: List of edge dictionaries with source, target, rate, delta_g
            title: Plot title
            output_path: Path to save figure (None to display)

        Returns:
            Matplotlib figure if successful
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib required for visualization")
            return None

        plt.style.use(self.config.style)
        fig, axes = plt.subplots(1, 2, figsize=self.config.figsize, dpi=self.config.dpi)

        ax_network = axes[0]
        ax_energy = axes[1]

        # Layout nodes in a circle
        n_nodes = len(nodes)
        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        radius = 5.0
        positions = {
            node["id"]: (radius * np.cos(angles[i]), radius * np.sin(angles[i]))
            for i, node in enumerate(nodes)
        }

        # Draw edges
        for edge in edges:
            src_pos = positions.get(edge["source"])
            tgt_pos = positions.get(edge["target"])
            if src_pos and tgt_pos:
                # Draw arrow
                ax_network.annotate(
                    "",
                    xy=tgt_pos,
                    xytext=src_pos,
                    arrowprops=dict(
                        arrowstyle="->",
                        color="cyan",
                        lw=1.5,
                        alpha=0.7,
                    ),
                )

        # Draw nodes
        energies = [node.get("free_energy", 0) for node in nodes]
        min_e, max_e = min(energies), max(energies)
        norm_energies = [(e - min_e) / (max_e - min_e + 1e-10) for e in energies]

        cmap = plt.get_cmap(self.config.colormap)
        for i, node in enumerate(nodes):
            x, y = positions[node["id"]]
            color = cmap(norm_energies[i])
            circle = plt.Circle((x, y), 0.8, color=color, ec="white", lw=2, alpha=self.config.alpha)
            ax_network.add_patch(circle)
            ax_network.text(x, y, node["id"], ha="center", va="center", fontsize=8, color="white")

        ax_network.set_xlim(-8, 8)
        ax_network.set_ylim(-8, 8)
        ax_network.set_aspect("equal")
        ax_network.set_title("Mechanism Network", fontsize=12)
        ax_network.axis("off")

        # Energy landscape
        node_ids = [node["id"] for node in nodes]
        concentrations = [node.get("concentration", 0.5) for node in nodes]

        bars = ax_energy.bar(
            range(len(nodes)),
            energies,
            color=[cmap(e) for e in norm_energies],
            alpha=self.config.alpha,
            edgecolor="white",
            linewidth=1,
        )

        ax_energy.set_xticks(range(len(nodes)))
        ax_energy.set_xticklabels(node_ids, rotation=45, ha="right", fontsize=8)
        ax_energy.set_ylabel("Free Energy (kJ/mol)", fontsize=10)
        ax_energy.set_title("Energy Landscape", fontsize=12)
        ax_energy.grid(True, alpha=0.3)

        # Add colorbar
        if self.config.show_colorbar:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min_e, max_e))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax_energy, shrink=0.8)
            cbar.set_label("Free Energy (kJ/mol)", fontsize=9)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.config.dpi, bbox_inches="tight")
            logger.info(f"Saved visualization to {output_path}")

        return fig

    def visualize_quantum_state(
        self,
        amplitudes: np.ndarray,
        title: str = "Quantum State",
        output_path: Optional[str] = None,
    ) -> Optional[Figure]:
        """Visualize quantum state amplitudes.

        Args:
            amplitudes: Complex amplitude array
            title: Plot title
            output_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib required for visualization")
            return None

        plt.style.use(self.config.style)
        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize, dpi=self.config.dpi)

        n_states = len(amplitudes)
        probabilities = np.abs(amplitudes) ** 2
        phases = np.angle(amplitudes)

        # Probability distribution
        ax = axes[0, 0]
        colors = plt.get_cmap(self.config.colormap)(np.linspace(0, 1, n_states))
        ax.bar(range(n_states), probabilities, color=colors, alpha=self.config.alpha)
        ax.set_xlabel("Basis State")
        ax.set_ylabel("Probability")
        ax.set_title("Probability Distribution")
        ax.grid(True, alpha=0.3)

        # Phase diagram (polar)
        ax = axes[0, 1]
        ax = plt.subplot(2, 2, 2, projection="polar")
        for i, (amp, phase) in enumerate(zip(np.abs(amplitudes), phases)):
            ax.plot([0, phase], [0, amp], color=colors[i], lw=2, alpha=0.8)
            ax.scatter([phase], [amp], color=colors[i], s=50, zorder=5)
        ax.set_title("Phase Diagram")

        # Real/Imaginary components
        ax = axes[1, 0]
        ax.bar(np.arange(n_states) - 0.2, np.real(amplitudes), 0.4, label="Real", alpha=0.7)
        ax.bar(np.arange(n_states) + 0.2, np.imag(amplitudes), 0.4, label="Imaginary", alpha=0.7)
        ax.set_xlabel("Basis State")
        ax.set_ylabel("Amplitude Component")
        ax.set_title("Real & Imaginary Components")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Bloch sphere approximation (for single qubit)
        ax = axes[1, 1]
        ax = plt.subplot(2, 2, 4, projection="3d")

        # Draw Bloch sphere wireframe
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(u.size), np.cos(v))
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.2, linewidth=0.5)

        # Plot state vector (simplified)
        if n_states == 2:
            theta = 2 * np.arccos(np.abs(amplitudes[0]))
            phi = phases[1] - phases[0]
            sx = np.sin(theta) * np.cos(phi)
            sy = np.sin(theta) * np.sin(phi)
            sz = np.cos(theta)
            ax.quiver(0, 0, 0, sx, sy, sz, color="cyan", arrow_length_ratio=0.1, lw=2)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Bloch Sphere")

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.config.dpi, bbox_inches="tight")
            logger.info(f"Saved visualization to {output_path}")

        return fig

    def visualize_3d_field(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        values: np.ndarray,
        title: str = "3D Scalar Field",
        output_path: Optional[str] = None,
    ) -> Optional[Figure]:
        """Visualize 3D scalar field.

        Args:
            x, y, z: Coordinate arrays
            values: Scalar values at each point
            title: Plot title
            output_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib required for visualization")
            return None

        plt.style.use(self.config.style)
        fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        ax = fig.add_subplot(111, projection="3d")

        cmap = plt.get_cmap(self.config.colormap)
        norm_values = (values - values.min()) / (values.max() - values.min() + 1e-10)
        colors = cmap(norm_values)

        scatter = ax.scatter(
            x, y, z, c=values, cmap=self.config.colormap, alpha=self.config.alpha, s=20
        )

        if self.config.show_colorbar:
            cbar = fig.colorbar(scatter, shrink=0.6, pad=0.1)
            cbar.set_label("Field Value")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title, fontsize=14, fontweight="bold")

        if output_path:
            fig.savefig(output_path, dpi=self.config.dpi, bbox_inches="tight")
            logger.info(f"Saved visualization to {output_path}")

        return fig


def create_demo_mechanism() -> tuple[list[dict], list[dict]]:
    """Create demo bio-mechanism data for visualization."""
    nodes = [
        {"id": "S0", "protein": "Inactive", "free_energy": -10.0, "concentration": 0.8},
        {"id": "S1", "protein": "Activated", "free_energy": -15.0, "concentration": 0.1},
        {"id": "S2", "protein": "Bound", "free_energy": -25.0, "concentration": 0.05},
        {"id": "S3", "protein": "Catalytic", "free_energy": -30.0, "concentration": 0.03},
        {"id": "S4", "protein": "Product", "free_energy": -40.0, "concentration": 0.02},
    ]

    edges = [
        {"source": "S0", "target": "S1", "rate": 1.5, "delta_g": -5.0},
        {"source": "S1", "target": "S2", "rate": 2.0, "delta_g": -10.0},
        {"source": "S2", "target": "S3", "rate": 0.8, "delta_g": -5.0},
        {"source": "S3", "target": "S4", "rate": 3.0, "delta_g": -10.0},
        {"source": "S1", "target": "S0", "rate": 0.2, "delta_g": 5.0},  # Reverse
        {"source": "S4", "target": "S0", "rate": 0.1, "delta_g": 30.0},  # Recycling
    ]

    return nodes, edges


def create_demo_quantum_state(n_qubits: int = 3) -> np.ndarray:
    """Create demo quantum state for visualization."""
    n_states = 2**n_qubits
    # Create a superposition state with varying amplitudes
    amplitudes = np.random.randn(n_states) + 1j * np.random.randn(n_states)
    amplitudes /= np.linalg.norm(amplitudes)  # Normalize
    return amplitudes


def run_demo() -> None:
    """Run visualization demo."""
    print("=" * 60)
    print("QUBIC Visualization Module Demo")
    print("=" * 60)
    print()

    viz = QubicVisualizer(VizConfig(style="dark_background"))

    # Demo 1: Bio-mechanism network
    print("1. Generating Bio-Mechanism Network Visualization...")
    nodes, edges = create_demo_mechanism()
    fig1 = viz.visualize_mechanism(
        nodes=nodes,
        edges=edges,
        title="XENON Bio-Mechanism Network",
        output_path="qubic_mechanism_viz.png",
    )
    if fig1:
        print("   ✅ Saved: qubic_mechanism_viz.png")

    # Demo 2: Quantum state
    print("\n2. Generating Quantum State Visualization...")
    amplitudes = create_demo_quantum_state(3)
    fig2 = viz.visualize_quantum_state(
        amplitudes=amplitudes,
        title="Quantum State |ψ⟩",
        output_path="qubic_quantum_viz.png",
    )
    if fig2:
        print("   ✅ Saved: qubic_quantum_viz.png")

    # Demo 3: 3D scalar field
    print("\n3. Generating 3D Scalar Field Visualization...")
    np.random.seed(42)
    n_points = 200
    x = np.random.randn(n_points)
    y = np.random.randn(n_points)
    z = np.random.randn(n_points)
    values = np.sin(x) * np.cos(y) * np.exp(-(z**2))

    fig3 = viz.visualize_3d_field(
        x=x,
        y=y,
        z=z,
        values=values,
        title="3D Energy Field",
        output_path="qubic_field_viz.png",
    )
    if fig3:
        print("   ✅ Saved: qubic_field_viz.png")

    print()
    print("=" * 60)
    print("Demo complete! Generated visualizations:")
    print("  - qubic_mechanism_viz.png")
    print("  - qubic_quantum_viz.png")
    print("  - qubic_field_viz.png")
    print("=" * 60)


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="QUBIC Visualization Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m qubic.visualization.qubic_viz demo
  python -m qubic.visualization.qubic_viz --help
        """,
    )

    parser.add_argument(
        "command",
        choices=["demo", "version"],
        help="Command to run",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output directory for visualizations",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.command == "demo":
        run_demo()
    elif args.command == "version":
        print("QUBIC Visualization Module v0.1.0")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
