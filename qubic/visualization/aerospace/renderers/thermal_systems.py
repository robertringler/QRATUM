"""Thermal systems rendering module.

Provides visualization for:
- Temperature field distribution
- Heat flux vectors
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


def render_thermal_field(
    temperature: np.ndarray,
    geometry: np.ndarray,
    title: str,
    output_path: str | None,
    colormap: str,
    show_isotherms: bool,
    isotherm_levels: int,
    config,
) -> Figure | None:
    """Render temperature distribution on geometry.

    Args:
        temperature: N array of temperature values in Kelvin
        geometry: Nx3 array of point coordinates
        title: Plot title
        output_path: Optional path to save figure
        colormap: Colormap name
        show_isotherms: If True, show isotherm contours
        isotherm_levels: Number of isotherm levels
        config: AerospaceVizConfig instance

    Returns:
        Matplotlib Figure object if successful, None otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib not available")
        return None

    plt.style.use(config.style)
    fig = plt.figure(figsize=config.figsize, dpi=config.dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Plot temperature field
    scatter = ax.scatter(
        geometry[:, 0],
        geometry[:, 1],
        geometry[:, 2],
        c=temperature,
        cmap=colormap,
        s=30,
        alpha=config.alpha,
    )

    if config.show_colorbar:
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label("Temperature (K)")

    # Show isotherms if requested
    if show_isotherms:
        # Create isotherm levels
        temp_min, temp_max = temperature.min(), temperature.max()
        levels = np.linspace(temp_min, temp_max, isotherm_levels)

        # For simplicity, project to XY plane and draw contours
        try:
            from scipy.interpolate import griddata

            # Create regular grid
            xi = np.linspace(geometry[:, 0].min(), geometry[:, 0].max(), 50)
            yi = np.linspace(geometry[:, 1].min(), geometry[:, 1].max(), 50)
            xi_grid, yi_grid = np.meshgrid(xi, yi)

            # Interpolate temperature to grid
            zi = griddata(
                (geometry[:, 0], geometry[:, 1]),
                temperature,
                (xi_grid, yi_grid),
                method="linear",
            )

            # Draw contours at z=geometry.min()
            z_offset = geometry[:, 2].min()
            ax.contour(
                xi_grid,
                yi_grid,
                zi,
                levels=levels,
                zdir="z",
                offset=z_offset,
                colors="white",
                alpha=0.5,
                linewidths=1,
            )
        except ImportError:
            logger.warning("scipy not available, skipping isotherm contours")
        except Exception as e:
            logger.warning(f"Failed to render isotherms: {e}")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved thermal field to {output_path}")

    return fig


def render_heat_flux(
    heat_flux: np.ndarray,
    surface_normals: np.ndarray,
    geometry: np.ndarray,
    title: str,
    output_path: str | None,
    config,
) -> Figure | None:
    """Render heat flux vectors on surface.

    Args:
        heat_flux: Nx3 array of heat flux vectors in W/m^2
        surface_normals: Nx3 array of surface normal vectors
        geometry: Nx3 array of surface point coordinates
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
    ax = fig.add_subplot(111, projection="3d")

    # Compute heat flux magnitude for coloring
    flux_magnitude = np.linalg.norm(heat_flux, axis=1)

    # Plot surface points colored by heat flux magnitude
    scatter = ax.scatter(
        geometry[:, 0],
        geometry[:, 1],
        geometry[:, 2],
        c=flux_magnitude,
        cmap=config.colormap,
        s=30,
        alpha=config.alpha,
    )

    if config.show_colorbar:
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label("Heat Flux Magnitude (W/m²)")

    # Draw heat flux vectors (subsampled for clarity)
    step = max(1, len(geometry) // 30)
    indices = range(0, len(geometry), step)

    # Scale vectors for visibility
    flux_scale = 0.1 / (flux_magnitude.max() + 1e-10)

    for i in indices:
        ax.quiver(
            geometry[i, 0],
            geometry[i, 1],
            geometry[i, 2],
            heat_flux[i, 0] * flux_scale,
            heat_flux[i, 1] * flux_scale,
            heat_flux[i, 2] * flux_scale,
            color="red",
            alpha=0.7,
            arrow_length_ratio=0.3,
        )

    # Draw surface normals (subsampled)
    for i in indices:
        ax.quiver(
            geometry[i, 0],
            geometry[i, 1],
            geometry[i, 2],
            surface_normals[i, 0] * 0.1,
            surface_normals[i, 1] * 0.1,
            surface_normals[i, 2] * 0.1,
            color="cyan",
            alpha=0.5,
            arrow_length_ratio=0.3,
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved heat flux to {output_path}")

    return fig
