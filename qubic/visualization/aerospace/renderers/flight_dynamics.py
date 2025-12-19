"""Flight dynamics rendering module.

Provides visualization for:
- Aircraft trajectories with vapor trails
- Airflow velocity fields with streamlines
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


def render_flight_trajectory(
    trajectory: np.ndarray,
    velocity: np.ndarray | None,
    title: str,
    output_path: str | None,
    show_vapor_trail: bool,
    config,
    rng: np.random.Generator,
) -> Figure | None:
    """Render flight trajectory with optional vapor trails.

    Args:
        trajectory: Nx3 array of XYZ positions in meters
        velocity: Optional Nx3 array of velocity vectors in m/s
        title: Plot title
        output_path: Optional path to save figure
        show_vapor_trail: If True, render vapor trail effect
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

    # Plot trajectory line
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        color="cyan",
        linewidth=2,
        label="Trajectory",
    )

    # Add start and end markers
    ax.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        trajectory[0, 2],
        color="green",
        s=100,
        marker="o",
        label="Start",
    )
    ax.scatter(
        trajectory[-1, 0],
        trajectory[-1, 1],
        trajectory[-1, 2],
        color="red",
        s=100,
        marker="^",
        label="End",
    )

    # Show velocity vectors if provided
    if velocity is not None and config.show_velocity_vectors:
        # Subsample for clarity
        step = max(1, len(trajectory) // 20)
        indices = range(0, len(trajectory), step)

        for i in indices:
            vel_scale = 50.0  # Adjust for visibility
            ax.quiver(
                trajectory[i, 0],
                trajectory[i, 1],
                trajectory[i, 2],
                velocity[i, 0],
                velocity[i, 1],
                velocity[i, 2],
                length=vel_scale,
                color="yellow",
                alpha=0.6,
                arrow_length_ratio=0.3,
            )

    # Vapor trail effect
    if show_vapor_trail:
        # Create particle effect along trajectory
        len(trajectory) * 5
        particle_positions = []

        for i in range(len(trajectory) - 1):
            # Interpolate between trajectory points
            alpha = rng.random(5)
            for a in alpha:
                pos = trajectory[i] * (1 - a) + trajectory[i + 1] * a
                # Add random offset perpendicular to motion
                offset = rng.normal(0, 2.0, 3)
                particle_positions.append(pos + offset)

        if particle_positions:
            particles = np.array(particle_positions)
            ax.scatter(
                particles[:, 0],
                particles[:, 1],
                particles[:, 2],
                c="white",
                s=1,
                alpha=0.3,
                label="Vapor Trail",
            )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved flight trajectory to {output_path}")

    return fig


def render_airflow_streamlines(
    velocity_field: np.ndarray,
    grid_shape: tuple[int, int, int],
    density: int,
    title: str,
    output_path: str | None,
    config,
    rng: np.random.Generator,
) -> Figure | None:
    """Render airflow velocity field with streamlines.

    Args:
        velocity_field: Flattened velocity field (N, 3) in m/s
        grid_shape: Shape of 3D grid (nx, ny, nz)
        density: Streamline density
        title: Plot title
        output_path: Optional path to save figure
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

    # Reshape velocity field
    nx, ny, nz = grid_shape
    try:
        vx = velocity_field[:, 0].reshape(grid_shape)
        vy = velocity_field[:, 1].reshape(grid_shape)
        vz = velocity_field[:, 2].reshape(grid_shape)
    except ValueError as e:
        logger.error(f"Failed to reshape velocity field: {e}")
        return None

    # Create coordinate grids
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    np.linspace(0, nz - 1, nz)

    # Compute velocity magnitude for coloring
    vel_mag = np.sqrt(vx**2 + vy**2 + vz**2)

    # Generate streamline seed points deterministically
    n_seeds = density
    seed_indices = rng.integers(0, nx * ny * nz, size=n_seeds)

    for idx in seed_indices[: min(n_seeds, 50)]:  # Limit for performance
        # Convert flat index to 3D coordinates
        iz = idx // (nx * ny)
        iy = (idx % (nx * ny)) // nx
        ix = idx % nx

        # Simple streamline integration (Euler method)
        points = [(ix, iy, iz)]
        current = np.array([ix, iy, iz], dtype=float)

        for _ in range(50):  # Max steps
            i, j, k = int(current[0]), int(current[1]), int(current[2])

            # Check bounds
            if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
                break

            # Get velocity at current position
            vel = np.array([vx[i, j, k], vy[i, j, k], vz[i, j, k]])
            vel_norm = np.linalg.norm(vel)

            if vel_norm < 1e-6:
                break

            # Step along velocity direction
            step_size = 0.5
            current = current + (vel / vel_norm) * step_size

            points.append(tuple(current))

        if len(points) > 2:
            points_array = np.array(points)
            ax.plot(
                points_array[:, 0],
                points_array[:, 1],
                points_array[:, 2],
                color="cyan",
                alpha=0.6,
                linewidth=1,
            )

    # Add velocity magnitude contour on a slice
    if config.show_pressure_gradients:
        slice_z = nz // 2
        im = ax.contourf(
            x,
            y,
            vel_mag[:, :, slice_z].T,
            zdir="z",
            offset=slice_z,
            levels=10,
            cmap=config.colormap,
            alpha=0.5,
        )
        if config.show_colorbar:
            cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.1)
            cbar.set_label("Velocity Magnitude (m/s)")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved airflow streamlines to {output_path}")

    return fig
