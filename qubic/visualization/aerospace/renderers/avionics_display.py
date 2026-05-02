"""Avionics display rendering module.

Provides visualization for:
- Sensor field of view cones
- Radar cross section polar plots
"""

from __future__ import annotations

import logging

import numpy as np

try:
    import matplotlib  # noqa: F401
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Figure = None  # type: ignore

logger = logging.getLogger(__name__)


def render_sensor_fov(
    sensor_position: np.ndarray,
    sensor_orientation: np.ndarray,
    fov_horizontal: float,
    fov_vertical: float,
    range_m: float,
    title: str,
    output_path: str | None,
    cone_color: str,
    cone_alpha: float,
    config,
    rng: np.random.Generator,
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
        config: AerospaceVizConfig instance
        rng: Numpy random number generator

    Returns:
        Matplotlib Figure object if successful, None otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib not available")
        return None

    plt.style.use(config.style)
    fig = plt.figure(figsize=config.figsize, dpi=config.dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Normalize orientation vector
    orientation = sensor_orientation / (np.linalg.norm(sensor_orientation) + 1e-10)

    # Convert FOV angles to radians
    fov_h_rad = np.deg2rad(fov_horizontal)
    fov_v_rad = np.deg2rad(fov_vertical)

    # Generate cone mesh
    n_theta = 20
    n_phi = 20

    theta = np.linspace(-fov_h_rad / 2, fov_h_rad / 2, n_theta)
    phi = np.linspace(-fov_v_rad / 2, fov_v_rad / 2, n_phi)

    # Create basis vectors perpendicular to orientation
    # Find an arbitrary perpendicular vector
    if abs(orientation[0]) < abs(orientation[1]):
        perp1 = np.array([0, -orientation[2], orientation[1]])
    else:
        perp1 = np.array([-orientation[2], 0, orientation[0]])

    perp1 = perp1 / (np.linalg.norm(perp1) + 1e-10)
    perp2 = np.cross(orientation, perp1)
    perp2 = perp2 / (np.linalg.norm(perp2) + 1e-10)

    # Generate cone surface points
    cone_points = []

    for t in theta:
        for p in phi:
            # Compute direction in local coordinate system
            local_dir = orientation + np.tan(t) * perp1 + np.tan(p) * perp2
            local_dir = local_dir / (np.linalg.norm(local_dir) + 1e-10)

            # Scale to range
            point = sensor_position + range_m * local_dir
            cone_points.append(point)

    cone_points = np.array(cone_points)

    # Plot cone surface as scatter
    ax.scatter(
        cone_points[:, 0],
        cone_points[:, 1],
        cone_points[:, 2],
        c=cone_color,
        s=5,
        alpha=cone_alpha,
        label="FOV Cone",
    )

    # Draw sensor position
    ax.scatter(
        sensor_position[0],
        sensor_position[1],
        sensor_position[2],
        c="red",
        s=100,
        marker="o",
        label="Sensor",
    )

    # Draw center axis
    center_point = sensor_position + range_m * orientation
    ax.plot(
        [sensor_position[0], center_point[0]],
        [sensor_position[1], center_point[1]],
        [sensor_position[2], center_point[2]],
        color="yellow",
        linewidth=2,
        label="Boresight",
    )

    # Draw FOV boundary lines
    corners = [
        orientation + np.tan(fov_h_rad / 2) * perp1 + np.tan(fov_v_rad / 2) * perp2,
        orientation + np.tan(fov_h_rad / 2) * perp1 - np.tan(fov_v_rad / 2) * perp2,
        orientation - np.tan(fov_h_rad / 2) * perp1 + np.tan(fov_v_rad / 2) * perp2,
        orientation - np.tan(fov_h_rad / 2) * perp1 - np.tan(fov_v_rad / 2) * perp2,
    ]

    for corner_dir in corners:
        corner_dir = corner_dir / (np.linalg.norm(corner_dir) + 1e-10)
        corner_point = sensor_position + range_m * corner_dir
        ax.plot(
            [sensor_position[0], corner_point[0]],
            [sensor_position[1], corner_point[1]],
            [sensor_position[2], corner_point[2]],
            color="white",
            linewidth=1,
            alpha=0.5,
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set equal aspect ratio
    max_range = range_m * 1.2
    mid = sensor_position
    ax.set_xlim([mid[0] - max_range, mid[0] + max_range])
    ax.set_ylim([mid[1] - max_range, mid[1] + max_range])
    ax.set_zlim([mid[2] - max_range, mid[2] + max_range])

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved sensor FOV to {output_path}")

    return fig


def render_radar_cross_section(
    geometry: np.ndarray,
    rcs_db: np.ndarray,
    frequency_ghz: float,
    azimuth_range: tuple[float, float],
    elevation: float,
    title: str,
    output_path: str | None,
    config,
) -> Figure | None:
    """Render radar cross section polar plot.

    Args:
        geometry: Nx3 array of geometry points
        rcs_db: N array of RCS values in dBsm
        frequency_ghz: Radar frequency in GHz
        azimuth_range: Azimuth angle range (min, max) in degrees
        elevation: Elevation angle in degrees
        title: Plot title
        output_path: Optional path to save figure
        config: AerospaceVizConfig instance

    Returns:
        Matplotlib Figure object if successful, None otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib not available")
        return None

    plt.style.use(config.style)
    fig = plt.figure(figsize=config.figsize, dpi=config.dpi)

    # Create polar plot
    ax = fig.add_subplot(111, projection="polar")

    # Convert geometry to azimuth angles (assuming centered at origin)
    azimuth_angles = np.arctan2(geometry[:, 1], geometry[:, 0])

    # Filter by azimuth range
    az_min, az_max = np.deg2rad(azimuth_range[0]), np.deg2rad(azimuth_range[1])
    mask = (azimuth_angles >= az_min) & (azimuth_angles <= az_max)

    filtered_azimuth = azimuth_angles[mask]
    filtered_rcs = rcs_db[mask]

    # Sort by azimuth for plotting
    sorted_indices = np.argsort(filtered_azimuth)
    filtered_azimuth = filtered_azimuth[sorted_indices]
    filtered_rcs = filtered_rcs[sorted_indices]

    # Convert RCS from dBsm to linear scale for radius
    # Use offset to handle negative dB values
    rcs_offset = max(0, -filtered_rcs.min()) + 1
    rcs_linear = filtered_rcs + rcs_offset

    # Plot RCS
    ax.plot(
        filtered_azimuth,
        rcs_linear,
        color="cyan",
        linewidth=2,
        label=f"{frequency_ghz} GHz",
    )
    ax.fill(filtered_azimuth, rcs_linear, color="cyan", alpha=0.3)

    # Formatting
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(True, alpha=0.3)

    # Add radial labels
    ax.set_ylabel("RCS (dBsm + offset)", labelpad=30)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved RCS plot to {output_path}")

    return fig
