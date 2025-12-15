"""Tire-specific rendering engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..core.camera import Camera
from ..core.lighting import Light, LightType, PBRMaterial
from ..core.renderer import RenderConfig, SceneRenderer
from ..core.scene_graph import SceneGraph
from .mesh_generator import TireMesh, TireMeshGenerator


class TireRenderer:
    """Specialized renderer for tire visualization.

    Args:
        config: Rendering configuration
    """

    def __init__(self, config: RenderConfig | None = None) -> None:
        """Initialize tire renderer."""
        self.config = config or RenderConfig()
        self.renderer = SceneRenderer(self.config)
        self.mesh_generator = TireMeshGenerator(resolution=32)

    def render_tire_3d(
        self,
        tire_mesh: TireMesh,
        camera: Camera | None = None,
        lights: list[Light] | None = None,
        material: PBRMaterial | None = None,
    ) -> np.ndarray:
        """Render 3D tire model with lighting.

        Args:
            tire_mesh: Tire mesh to render
            camera: Camera (defaults to standard view)
            lights: Light sources (defaults to basic lighting)
            material: Surface material (defaults to rubber material)

        Returns:
            Rendered image as RGB array
        """
        if camera is None:
            camera = Camera(
                position=np.array([2.0, 1.5, 2.0]),
                target=np.array([0.0, 0.0, 0.0]),
            )

        if lights is None:
            lights = [
                Light(type=LightType.DIRECTIONAL, direction=np.array([1, -1, 1]), intensity=0.8),
                Light(type=LightType.AMBIENT, intensity=0.2),
            ]

        if material is None:
            # Default rubber material
            material = PBRMaterial(
                albedo=np.array([0.1, 0.1, 0.1]),
                metallic=0.0,
                roughness=0.8,
            )

        # Create scene
        scene = SceneGraph()

        # Render using matplotlib 3D
        fig = plt.figure(figsize=(self.config.width / 100, self.config.height / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")

        # Plot mesh
        vertices = tire_mesh.vertices
        faces = tire_mesh.faces

        # Simple shading based on normals
        face_normals = tire_mesh.compute_face_normals()
        light_dir = lights[0].direction / np.linalg.norm(lights[0].direction)
        shading = np.abs(face_normals @ light_dir)

        # Plot triangles
        for i, face in enumerate(faces[::10]):  # Subsample for performance
            triangle = vertices[face]
            color = material.albedo * shading[i]
            ax.plot_trisurf(
                triangle[:, 0],
                triangle[:, 1],
                triangle[:, 2],
                color=color,
                alpha=material.alpha,
                antialiased=True,
            )

        # Set view
        ax.view_init(elev=30, azim=45)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)

        return image

    def render_thermal_map(
        self, tire_mesh: TireMesh, thermal_data: np.ndarray, output_path: Path | None = None
    ) -> np.ndarray:
        """Render thermal gradient visualization.

        Args:
            tire_mesh: Tire mesh
            thermal_data: Temperature values at vertices
            output_path: Optional output path

        Returns:
            Rendered thermal map
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        vertices = tire_mesh.vertices

        # Normalize thermal data for color mapping
        temp_min = thermal_data.min()
        temp_max = thermal_data.max()
        temp_norm = (thermal_data - temp_min) / (temp_max - temp_min + 1e-10)

        # Plot with thermal colors
        scatter = ax.scatter(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            c=temp_norm,
            cmap="hot",
            s=5,
            alpha=0.8,
        )

        plt.colorbar(scatter, ax=ax, label="Temperature (°C)")
        ax.set_title("Tire Thermal Distribution")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")

        plt.close(fig)

        return image

    def render_stress_distribution(
        self, tire_mesh: TireMesh, stress_data: np.ndarray, output_path: Path | None = None
    ) -> np.ndarray:
        """Render stress field visualization.

        Args:
            tire_mesh: Tire mesh
            stress_data: Stress values at vertices
            output_path: Optional output path

        Returns:
            Rendered stress distribution
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        vertices = tire_mesh.vertices

        # Normalize stress data
        stress_min = stress_data.min()
        stress_max = stress_data.max()
        stress_norm = (stress_data - stress_min) / (stress_max - stress_min + 1e-10)

        # Plot with stress colors
        scatter = ax.scatter(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            c=stress_norm,
            cmap="viridis",
            s=5,
            alpha=0.8,
        )

        plt.colorbar(scatter, ax=ax, label="Stress (MPa)")
        ax.set_title("Tire Stress Distribution")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")

        plt.close(fig)

        return image

    def render_wear_pattern(
        self, tire_mesh: TireMesh, wear_data: np.ndarray, output_path: Path | None = None
    ) -> np.ndarray:
        """Render tread wear visualization.

        Args:
            tire_mesh: Tire mesh
            wear_data: Wear depth values at vertices
            output_path: Optional output path

        Returns:
            Rendered wear pattern
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        vertices = tire_mesh.vertices

        # Normalize wear data
        wear_min = wear_data.min()
        wear_max = wear_data.max()
        wear_norm = (wear_data - wear_min) / (wear_max - wear_min + 1e-10)

        # Plot with wear colors
        scatter = ax.scatter(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            c=wear_norm,
            cmap="plasma",
            s=5,
            alpha=0.8,
        )

        plt.colorbar(scatter, ax=ax, label="Wear Depth (mm)")
        ax.set_title("Tire Wear Pattern")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")

        plt.close(fig)

        return image

    def render_performance_dashboard(
        self, simulation_result: Any, output_path: Path | None = None
    ) -> np.ndarray:
        """Render comprehensive multi-panel performance dashboard.

        Args:
            simulation_result: TireSimulationResult object
            output_path: Optional output path

        Returns:
            Rendered dashboard image
        """
        fig = plt.figure(figsize=(16, 12))

        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Extract metrics
        metrics = simulation_result.performance_metrics

        # Panel 1: Grip coefficients
        ax1 = fig.add_subplot(gs[0, 0])
        grip_types = ["Dry", "Wet", "Snow", "Ice"]
        grip_values = [
            metrics.dry_grip,
            metrics.wet_grip,
            metrics.snow_grip,
            metrics.ice_grip,
        ]
        ax1.bar(grip_types, grip_values, color=["red", "blue", "cyan", "lightblue"])
        ax1.set_ylabel("Grip Coefficient")
        ax1.set_title("Traction Performance")
        ax1.set_ylim(0, 1)

        # Panel 2: Performance indices
        ax2 = fig.add_subplot(gs[0, 1])
        indices = ["Thermal", "Durability", "Comfort"]
        index_values = [
            metrics.thermal_performance,
            metrics.durability_index,
            metrics.comfort_index,
        ]
        ax2.bar(indices, index_values, color=["orange", "green", "purple"])
        ax2.set_ylabel("Index (0-1)")
        ax2.set_title("Performance Indices")
        ax2.set_ylim(0, 1)

        # Panel 3: Optimization score
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.text(
            0.5,
            0.5,
            f"{metrics.optimization_score:.2f}",
            ha="center",
            va="center",
            fontsize=48,
            fontweight="bold",
        )
        ax3.set_title("Optimization Score")
        ax3.axis("off")

        # Panel 4: Rolling resistance & wear
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.bar(
            ["Rolling\nResistance", "Wear Rate"],
            [metrics.rolling_resistance, metrics.wear_rate / 10.0],
            color=["brown", "gray"],
        )
        ax4.set_ylabel("Value")
        ax4.set_title("Efficiency Metrics")

        # Panel 5: Environmental conditions
        ax5 = fig.add_subplot(gs[1, 1])
        env = simulation_result.environment
        env_text = (
            f"Temperature: {env.temperature_celsius:.1f}°C\n"
            f"Pressure: {simulation_result.pressure_kpa:.0f} kPa\n"
            f"Speed: {simulation_result.speed_kmh:.0f} km/h\n"
            f"Load: {simulation_result.load_kg:.0f} kg"
        )
        ax5.text(0.1, 0.5, env_text, fontsize=12, va="center", family="monospace")
        ax5.set_title("Operating Conditions")
        ax5.axis("off")

        # Panel 6: Predicted lifetime
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.text(
            0.5,
            0.5,
            f"{metrics.predicted_lifetime_km/1000:.1f}k km",
            ha="center",
            va="center",
            fontsize=36,
            fontweight="bold",
            color="darkgreen",
        )
        ax6.set_title("Predicted Lifetime")
        ax6.axis("off")

        # Panel 7-9: Visualization placeholders
        ax7 = fig.add_subplot(gs[2, :])
        suggestions = simulation_result.optimization_suggestions[:3] if simulation_result.optimization_suggestions else ["No suggestions available"]
        suggestions_text = "\n".join([f"• {s}" for s in suggestions])
        ax7.text(0.05, 0.5, suggestions_text, fontsize=11, va="center", wrap=True)
        ax7.set_title("Optimization Suggestions")
        ax7.axis("off")

        # Main title
        fig.suptitle(
            f"Tire Performance Dashboard - {simulation_result.simulation_id}",
            fontsize=16,
            fontweight="bold",
        )

        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")

        plt.close(fig)

        return image
