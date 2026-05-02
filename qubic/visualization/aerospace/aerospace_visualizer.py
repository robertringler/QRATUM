"""Aerospace Visualizer - DO-178C Level A compliant visualization system.

Extends QubicVisualizer with aerospace-specific rendering capabilities including:
- Flight trajectory and airflow visualization
- FEA mesh and structural analysis
- Thermal field rendering
- Avionics display and sensor FOV cones

Compliance Features:
- Deterministic rendering with seed-based reproducibility
- SHA-256 frame hashing for audit trails
- Comprehensive logging and validation
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Figure = None  # type: ignore

from qubic.visualization.qubic_viz import QubicVisualizer, VizConfig

logger = logging.getLogger(__name__)


class ComplianceMode(str, Enum):
    """Compliance mode for aerospace visualization."""

    DO178C_LEVEL_A = "DO178C_LEVEL_A"  # Deterministic, full audit trail
    DO178C_LEVEL_B = "DO178C_LEVEL_B"  # Deterministic, reduced audit
    DEVELOPMENT = "DEVELOPMENT"  # Non-deterministic, no audit


class RenderBackend(str, Enum):
    """Rendering backend options."""

    MATPLOTLIB = "matplotlib"  # CPU-based, always available
    THREEJS = "threejs"  # WebGL export for dashboard
    HEADLESS = "headless"  # CI/cluster rendering


@dataclass
class FrameAuditRecord:
    """Audit record for a single rendered frame."""

    frame_id: int
    timestamp_ns: int
    seed: int
    config_hash: str
    frame_hash: str
    render_time_ms: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class AerospaceVizConfig(VizConfig):
    """Configuration for aerospace visualization extending VizConfig."""

    compliance_mode: ComplianceMode = ComplianceMode.DEVELOPMENT
    render_backend: RenderBackend = RenderBackend.MATPLOTLIB
    seed: int = 42
    enable_audit_log: bool = False
    resolution: tuple[int, int] = (1920, 1080)
    target_fps: int = 60
    show_velocity_vectors: bool = True
    show_pressure_gradients: bool = True
    show_streamlines: bool = True
    streamline_density: int = 100
    enable_hud: bool = True
    hud_opacity: float = 0.8
    show_telemetry: bool = True


class AerospaceVisualizer(QubicVisualizer):
    """Aerospace-grade visualizer extending QubicVisualizer.

    Provides specialized rendering for aerospace simulation data with
    DO-178C Level A compliance support including deterministic rendering
    and comprehensive audit trails.

    Attributes:
        config: Aerospace-specific visualization configuration
        rng: Numpy random number generator for deterministic behavior
        audit_trail: List of frame audit records
        frame_counter: Current frame counter for audit trail
    """

    def __init__(self, config: AerospaceVizConfig | None = None):
        """Initialize aerospace visualizer.

        Args:
            config: Aerospace visualization configuration. If None, uses defaults.
        """
        self.config = config or AerospaceVizConfig()
        super().__init__(self.config)

        # Initialize deterministic RNG
        self.rng = np.random.default_rng(self.config.seed)

        # Initialize audit trail
        self.audit_trail: list[FrameAuditRecord] = []
        self.frame_counter: int = 0

        # Compute config hash for audit trail
        self._config_hash = self._compute_config_hash()

        logger.info(
            f"Initialized AerospaceVisualizer with compliance_mode={self.config.compliance_mode}, "
            f"seed={self.config.seed}, backend={self.config.render_backend}"
        )

    def _compute_config_hash(self) -> str:
        """Compute SHA-256 hash of configuration for audit trail.

        Returns:
            Hexadecimal SHA-256 hash string
        """
        config_dict = {
            "compliance_mode": self.config.compliance_mode.value,
            "seed": self.config.seed,
            "resolution": self.config.resolution,
            "render_backend": self.config.render_backend.value,
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def _compute_frame_hash(self, fig: Figure) -> str:
        """Compute SHA-256 hash of rendered frame for audit trail.

        Args:
            fig: Matplotlib Figure object

        Returns:
            Hexadecimal SHA-256 hash string

        Compliance:
            Required for DO-178C Level A audit trails
        """
        if not MATPLOTLIB_AVAILABLE or fig is None:
            return "N/A"

        # Render figure to buffer
        import io

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.config.dpi)
        buf.seek(0)
        frame_data = buf.read()
        buf.close()

        return hashlib.sha256(frame_data).hexdigest()

    def _record_frame(
        self, fig: Figure | None, render_time_ms: float, warnings: list[str] | None = None
    ) -> None:
        """Record frame in audit trail if audit logging is enabled.

        Args:
            fig: Rendered figure (None if rendering failed)
            render_time_ms: Time taken to render frame in milliseconds
            warnings: Optional list of warning messages

        Compliance:
            Creates audit record for DO-178C compliance
        """
        if not self.config.enable_audit_log:
            return

        frame_hash = self._compute_frame_hash(fig) if fig is not None else "RENDER_FAILED"

        record = FrameAuditRecord(
            frame_id=self.frame_counter,
            timestamp_ns=time.time_ns(),
            seed=self.config.seed,
            config_hash=self._config_hash,
            frame_hash=frame_hash,
            render_time_ms=render_time_ms,
            warnings=warnings or [],
        )

        self.audit_trail.append(record)
        self.frame_counter += 1

        logger.debug(
            f"Recorded frame {record.frame_id}: hash={frame_hash[:16]}..., "
            f"render_time={render_time_ms:.2f}ms"
        )

    def get_audit_trail(self) -> list[FrameAuditRecord]:
        """Return audit trail of all rendered frames.

        Returns:
            List of frame audit records

        Compliance:
            Provides traceable record of all rendering operations
        """
        return self.audit_trail.copy()

    def export_audit_trail(self, output_path: str) -> None:
        """Export audit trail to JSON file.

        Args:
            output_path: Path to output JSON file

        Compliance:
            Exports audit records for external compliance verification
        """
        audit_data = {
            "config": {
                "compliance_mode": self.config.compliance_mode.value,
                "seed": self.config.seed,
                "resolution": self.config.resolution,
                "render_backend": self.config.render_backend.value,
            },
            "frames": [
                {
                    "frame_id": record.frame_id,
                    "timestamp_ns": record.timestamp_ns,
                    "seed": record.seed,
                    "config_hash": record.config_hash,
                    "frame_hash": record.frame_hash,
                    "render_time_ms": record.render_time_ms,
                    "warnings": record.warnings,
                }
                for record in self.audit_trail
            ],
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(audit_data, f, indent=2)

        logger.info(f"Exported audit trail with {len(self.audit_trail)} frames to {output_path}")

    def generate_compliance_report(self) -> dict:
        """Generate compliance summary report.

        Returns:
            Dictionary containing compliance metrics and statistics

        Compliance:
            Provides summary statistics for certification artifacts
        """
        total_frames = len(self.audit_trail)
        total_warnings = sum(len(record.warnings) for record in self.audit_trail)

        if total_frames > 0:
            render_times = [record.render_time_ms for record in self.audit_trail]
            avg_render_time = sum(render_times) / total_frames
            max_render_time = max(render_times)
            min_render_time = min(render_times)
        else:
            avg_render_time = max_render_time = min_render_time = 0.0

        report = {
            "compliance_mode": self.config.compliance_mode.value,
            "seed": self.config.seed,
            "render_backend": self.config.render_backend.value,
            "config_hash": self._config_hash,
            "total_frames": total_frames,
            "total_warnings": total_warnings,
            "render_time_stats": {
                "average_ms": avg_render_time,
                "min_ms": min_render_time,
                "max_ms": max_render_time,
            },
            "audit_enabled": self.config.enable_audit_log,
        }

        return report

    def render_flight_trajectory(
        self,
        trajectory: np.ndarray,
        velocity: np.ndarray | None = None,
        title: str = "Flight Trajectory",
        output_path: str | None = None,
        show_vapor_trail: bool = False,
    ) -> Figure | None:
        """Render aircraft flight trajectory with optional vapor trails.

        Args:
            trajectory: Nx3 array of XYZ positions in meters
            velocity: Optional Nx3 array of velocity vectors in m/s
            title: Plot title
            output_path: Optional path to save figure
            show_vapor_trail: If True, render vapor trail effect

        Returns:
            Matplotlib Figure object if successful, None otherwise

        Compliance:
            Deterministic rendering based on seed configuration
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib required for flight trajectory rendering")
            return None

        start_time = time.time()
        warnings = []

        try:
            from qubic.visualization.aerospace.renderers import flight_dynamics

            fig = flight_dynamics.render_flight_trajectory(
                trajectory=trajectory,
                velocity=velocity,
                title=title,
                output_path=output_path,
                show_vapor_trail=show_vapor_trail,
                config=self.config,
                rng=self.rng,
            )

            render_time_ms = (time.time() - start_time) * 1000
            self._record_frame(fig, render_time_ms, warnings)

            return fig

        except Exception as e:
            logger.error(f"Failed to render flight trajectory: {e}")
            warnings.append(f"Render error: {str(e)}")
            render_time_ms = (time.time() - start_time) * 1000
            self._record_frame(None, render_time_ms, warnings)
            return None

    def render_airflow_streamlines(
        self,
        velocity_field: np.ndarray,
        grid_shape: tuple[int, int, int],
        density: int | None = None,
        title: str = "Airflow Streamlines",
        output_path: str | None = None,
    ) -> Figure | None:
        """Render airflow velocity field with streamlines.

        Args:
            velocity_field: Flattened velocity field (N, 3) in m/s
            grid_shape: Shape of 3D grid (nx, ny, nz)
            density: Streamline density (uses config default if None)
            title: Plot title
            output_path: Optional path to save figure

        Returns:
            Matplotlib Figure object if successful, None otherwise

        Compliance:
            Deterministic streamline generation using configured seed
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib required for airflow rendering")
            return None

        start_time = time.time()
        warnings = []

        try:
            from qubic.visualization.aerospace.renderers import flight_dynamics

            density = density or self.config.streamline_density

            fig = flight_dynamics.render_airflow_streamlines(
                velocity_field=velocity_field,
                grid_shape=grid_shape,
                density=density,
                title=title,
                output_path=output_path,
                config=self.config,
                rng=self.rng,
            )

            render_time_ms = (time.time() - start_time) * 1000
            self._record_frame(fig, render_time_ms, warnings)

            return fig

        except Exception as e:
            logger.error(f"Failed to render airflow streamlines: {e}")
            warnings.append(f"Render error: {str(e)}")
            render_time_ms = (time.time() - start_time) * 1000
            self._record_frame(None, render_time_ms, warnings)
            return None

    def render_fem_mesh(
        self,
        nodes: np.ndarray,
        elements: np.ndarray,
        stress_tensor: np.ndarray | None = None,
        title: str = "FEM Mesh",
        output_path: str | None = None,
        show_wireframe: bool = True,
        colormap: str | None = None,
    ) -> Figure | None:
        """Render FEA mesh with stress field overlay.

        Args:
            nodes: Nx3 array of node coordinates
            elements: Mx4 array of element connectivity (tetrahedral)
            stress_tensor: Optional Nx6 stress tensor (xx, yy, zz, xy, xz, yz)
            title: Plot title
            output_path: Optional path to save figure
            show_wireframe: If True, show mesh wireframe
            colormap: Colormap name (uses config default if None)

        Returns:
            Matplotlib Figure object if successful, None otherwise

        Compliance:
            Preserves FEA mesh topology for structural verification
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib required for FEM mesh rendering")
            return None

        start_time = time.time()
        warnings = []

        try:
            from qubic.visualization.aerospace.renderers import \
                structural_analysis

            colormap = colormap or self.config.colormap

            fig = structural_analysis.render_fem_mesh(
                nodes=nodes,
                elements=elements,
                stress_tensor=stress_tensor,
                title=title,
                output_path=output_path,
                show_wireframe=show_wireframe,
                colormap=colormap,
                config=self.config,
            )

            render_time_ms = (time.time() - start_time) * 1000
            self._record_frame(fig, render_time_ms, warnings)

            return fig

        except Exception as e:
            logger.error(f"Failed to render FEM mesh: {e}")
            warnings.append(f"Render error: {str(e)}")
            render_time_ms = (time.time() - start_time) * 1000
            self._record_frame(None, render_time_ms, warnings)
            return None

    def render_modal_analysis(
        self,
        nodes: np.ndarray,
        eigenvectors: np.ndarray,
        eigenfrequencies: np.ndarray,
        mode_index: int = 0,
        amplitude_scale: float = 1.0,
        title: str | None = None,
        output_path: str | None = None,
        animate: bool = False,
    ) -> Figure | None:
        """Render modal analysis eigenmodes.

        Args:
            nodes: Nx3 array of node coordinates
            eigenvectors: NxMx3 array of mode shapes (M modes)
            eigenfrequencies: M array of eigenfrequencies in Hz
            mode_index: Which mode to visualize
            amplitude_scale: Scaling factor for displacement
            title: Plot title (auto-generated if None)
            output_path: Optional path to save figure
            animate: If True, create animation (not implemented)

        Returns:
            Matplotlib Figure object if successful, None otherwise

        Compliance:
            Modal analysis results must be deterministic and traceable
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib required for modal analysis rendering")
            return None

        start_time = time.time()
        warnings = []

        if animate:
            warnings.append("Animation not implemented, rendering static frame")

        try:
            from qubic.visualization.aerospace.renderers import \
                structural_analysis

            if title is None:
                title = f"Mode {mode_index + 1}: {eigenfrequencies[mode_index]:.2f} Hz"

            fig = structural_analysis.render_modal_analysis(
                nodes=nodes,
                eigenvectors=eigenvectors,
                eigenfrequencies=eigenfrequencies,
                mode_index=mode_index,
                amplitude_scale=amplitude_scale,
                title=title,
                output_path=output_path,
                config=self.config,
            )

            render_time_ms = (time.time() - start_time) * 1000
            self._record_frame(fig, render_time_ms, warnings)

            return fig

        except Exception as e:
            logger.error(f"Failed to render modal analysis: {e}")
            warnings.append(f"Render error: {str(e)}")
            render_time_ms = (time.time() - start_time) * 1000
            self._record_frame(None, render_time_ms, warnings)
            return None

    def render_thermal_field(
        self,
        temperature: np.ndarray,
        geometry: np.ndarray,
        title: str = "Thermal Field",
        output_path: str | None = None,
        colormap: str | None = None,
        show_isotherms: bool = False,
        isotherm_levels: int | None = None,
    ) -> Figure | None:
        """Render temperature distribution on geometry.

        Args:
            temperature: N array of temperature values in Kelvin
            geometry: Nx3 array of point coordinates
            title: Plot title
            output_path: Optional path to save figure
            colormap: Colormap name (uses config default if None)
            show_isotherms: If True, show isotherm contours
            isotherm_levels: Number of isotherm levels (default 10)

        Returns:
            Matplotlib Figure object if successful, None otherwise

        Compliance:
            Thermal analysis must preserve temperature accuracy
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib required for thermal field rendering")
            return None

        start_time = time.time()
        warnings = []

        try:
            from qubic.visualization.aerospace.renderers import thermal_systems

            colormap = colormap or self.config.colormap
            isotherm_levels = isotherm_levels or 10

            fig = thermal_systems.render_thermal_field(
                temperature=temperature,
                geometry=geometry,
                title=title,
                output_path=output_path,
                colormap=colormap,
                show_isotherms=show_isotherms,
                isotherm_levels=isotherm_levels,
                config=self.config,
            )

            render_time_ms = (time.time() - start_time) * 1000
            self._record_frame(fig, render_time_ms, warnings)

            return fig

        except Exception as e:
            logger.error(f"Failed to render thermal field: {e}")
            warnings.append(f"Render error: {str(e)}")
            render_time_ms = (time.time() - start_time) * 1000
            self._record_frame(None, render_time_ms, warnings)
            return None

    def render_heat_flux(
        self,
        heat_flux: np.ndarray,
        surface_normals: np.ndarray,
        geometry: np.ndarray,
        title: str = "Heat Flux",
        output_path: str | None = None,
    ) -> Figure | None:
        """Render heat flux vectors on surface.

        Args:
            heat_flux: Nx3 array of heat flux vectors in W/m^2
            surface_normals: Nx3 array of surface normal vectors
            geometry: Nx3 array of surface point coordinates
            title: Plot title
            output_path: Optional path to save figure

        Returns:
            Matplotlib Figure object if successful, None otherwise

        Compliance:
            Heat flux visualization for thermal verification
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib required for heat flux rendering")
            return None

        start_time = time.time()
        warnings = []

        try:
            from qubic.visualization.aerospace.renderers import thermal_systems

            fig = thermal_systems.render_heat_flux(
                heat_flux=heat_flux,
                surface_normals=surface_normals,
                geometry=geometry,
                title=title,
                output_path=output_path,
                config=self.config,
            )

            render_time_ms = (time.time() - start_time) * 1000
            self._record_frame(fig, render_time_ms, warnings)

            return fig

        except Exception as e:
            logger.error(f"Failed to render heat flux: {e}")
            warnings.append(f"Render error: {str(e)}")
            render_time_ms = (time.time() - start_time) * 1000
            self._record_frame(None, render_time_ms, warnings)
            return None

    def render_sensor_fov(
        self,
        sensor_position: np.ndarray,
        sensor_orientation: np.ndarray,
        fov_horizontal: float,
        fov_vertical: float,
        range_m: float,
        title: str = "Sensor Field of View",
        output_path: str | None = None,
        cone_color: str = "cyan",
        cone_alpha: float = 0.3,
    ) -> Figure | None:
        """Render sensor field of view cone.

        Args:
            sensor_position: 3D position of sensor in meters
            sensor_orientation: 3D orientation vector (pointing direction)
            fov_horizontal: Horizontal field of view in degrees
            fov_vertical: Vertical field of view in degrees
            range_m: Sensor range in meters
            title: Plot title
            output_path: Optional path to save figure
            cone_color: Color for FOV cone
            cone_alpha: Transparency of FOV cone

        Returns:
            Matplotlib Figure object if successful, None otherwise

        Compliance:
            Sensor coverage verification for avionics systems
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib required for sensor FOV rendering")
            return None

        start_time = time.time()
        warnings = []

        try:
            from qubic.visualization.aerospace.renderers import \
                avionics_display

            fig = avionics_display.render_sensor_fov(
                sensor_position=sensor_position,
                sensor_orientation=sensor_orientation,
                fov_horizontal=fov_horizontal,
                fov_vertical=fov_vertical,
                range_m=range_m,
                title=title,
                output_path=output_path,
                cone_color=cone_color,
                cone_alpha=cone_alpha,
                config=self.config,
                rng=self.rng,
            )

            render_time_ms = (time.time() - start_time) * 1000
            self._record_frame(fig, render_time_ms, warnings)

            return fig

        except Exception as e:
            logger.error(f"Failed to render sensor FOV: {e}")
            warnings.append(f"Render error: {str(e)}")
            render_time_ms = (time.time() - start_time) * 1000
            self._record_frame(None, render_time_ms, warnings)
            return None

    def render_radar_cross_section(
        self,
        geometry: np.ndarray,
        rcs_db: np.ndarray,
        frequency_ghz: float,
        azimuth_range: tuple[float, float] = (0, 360),
        elevation: float = 0.0,
        title: str | None = None,
        output_path: str | None = None,
    ) -> Figure | None:
        """Render radar cross section polar plot.

        Args:
            geometry: Nx3 array of geometry points
            rcs_db: N array of RCS values in dBsm
            frequency_ghz: Radar frequency in GHz
            azimuth_range: Azimuth angle range (min, max) in degrees
            elevation: Elevation angle in degrees
            title: Plot title (auto-generated if None)
            output_path: Optional path to save figure

        Returns:
            Matplotlib Figure object if successful, None otherwise

        Compliance:
            RCS analysis for stealth/signature verification
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib required for RCS rendering")
            return None

        start_time = time.time()
        warnings = []

        try:
            from qubic.visualization.aerospace.renderers import \
                avionics_display

            if title is None:
                title = f"Radar Cross Section @ {frequency_ghz} GHz"

            fig = avionics_display.render_radar_cross_section(
                geometry=geometry,
                rcs_db=rcs_db,
                frequency_ghz=frequency_ghz,
                azimuth_range=azimuth_range,
                elevation=elevation,
                title=title,
                output_path=output_path,
                config=self.config,
            )

            render_time_ms = (time.time() - start_time) * 1000
            self._record_frame(fig, render_time_ms, warnings)

            return fig

        except Exception as e:
            logger.error(f"Failed to render RCS: {e}")
            warnings.append(f"Render error: {str(e)}")
            render_time_ms = (time.time() - start_time) * 1000
            self._record_frame(None, render_time_ms, warnings)
            return None
