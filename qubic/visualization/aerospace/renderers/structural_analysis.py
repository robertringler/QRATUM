"""Structural analysis rendering module.

Provides visualization for:
- FEA mesh with stress field overlay
- Modal analysis eigenmodes
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


def render_fem_mesh(
    nodes: np.ndarray,
    elements: np.ndarray,
    stress_tensor: np.ndarray | None,
    title: str,
    output_path: str | None,
    show_wireframe: bool,
    colormap: str,
    config,
) -> Figure | None:
    """Render FEA mesh with stress field overlay.

    Args:
        nodes: Nx3 array of node coordinates
        elements: Mx4 array of element connectivity (tetrahedral)
        stress_tensor: Optional Nx6 stress tensor (xx, yy, zz, xy, xz, yz)
        title: Plot title
        output_path: Optional path to save figure
        show_wireframe: If True, show mesh wireframe
        colormap: Colormap name
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

    # Compute von Mises stress if stress tensor provided
    if stress_tensor is not None:
        # von Mises: sqrt(0.5*((s1-s2)^2 + (s2-s3)^2 + (s3-s1)^2))
        # Simplified for principal stresses from tensor
        sxx = stress_tensor[:, 0]
        syy = stress_tensor[:, 1]
        szz = stress_tensor[:, 2]
        sxy = stress_tensor[:, 3]
        sxz = stress_tensor[:, 4]
        syz = stress_tensor[:, 5]

        von_mises = np.sqrt(
            0.5
            * (
                (sxx - syy) ** 2
                + (syy - szz) ** 2
                + (szz - sxx) ** 2
                + 6 * (sxy**2 + sxz**2 + syz**2)
            )
        )

        # Plot nodes colored by stress
        scatter = ax.scatter(
            nodes[:, 0],
            nodes[:, 1],
            nodes[:, 2],
            c=von_mises,
            cmap=colormap,
            s=20,
            alpha=config.alpha,
        )

        if config.show_colorbar:
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
            cbar.set_label("Von Mises Stress (Pa)")
    else:
        # Plot nodes without stress coloring
        ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c="cyan", s=10, alpha=config.alpha)

    # Draw element edges (wireframe)
    if show_wireframe:
        # For each tetrahedral element, draw edges
        edges_drawn = set()
        for elem in elements:
            # Tetrahedral element has 6 edges
            edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

            for i, j in edge_pairs:
                n1, n2 = sorted([elem[i], elem[j]])
                edge_key = (n1, n2)

                if edge_key not in edges_drawn:
                    ax.plot(
                        [nodes[n1, 0], nodes[n2, 0]],
                        [nodes[n1, 1], nodes[n2, 1]],
                        [nodes[n1, 2], nodes[n2, 2]],
                        color="gray",
                        alpha=0.3,
                        linewidth=0.5,
                    )
                    edges_drawn.add(edge_key)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved FEM mesh to {output_path}")

    return fig


def render_modal_analysis(
    nodes: np.ndarray,
    eigenvectors: np.ndarray,
    eigenfrequencies: np.ndarray,
    mode_index: int,
    amplitude_scale: float,
    title: str,
    output_path: str | None,
    config,
) -> Figure | None:
    """Render modal analysis eigenmodes.

    Args:
        nodes: Nx3 array of node coordinates
        eigenvectors: NxMx3 array of mode shapes (M modes)
        eigenfrequencies: M array of eigenfrequencies in Hz
        mode_index: Which mode to visualize
        amplitude_scale: Scaling factor for displacement
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

    # Extract mode shape for selected mode
    mode_shape = eigenvectors[:, mode_index, :]

    # Compute deformed positions
    deformed_nodes = nodes + amplitude_scale * mode_shape

    # Compute displacement magnitude for coloring
    displacement = np.linalg.norm(mode_shape, axis=1)

    # Plot original mesh (ghosted)
    ax.scatter(
        nodes[:, 0],
        nodes[:, 1],
        nodes[:, 2],
        c="gray",
        s=10,
        alpha=0.3,
        label="Original",
    )

    # Plot deformed mesh colored by displacement
    scatter = ax.scatter(
        deformed_nodes[:, 0],
        deformed_nodes[:, 1],
        deformed_nodes[:, 2],
        c=displacement,
        cmap=config.colormap,
        s=30,
        alpha=config.alpha,
        label="Deformed",
    )

    # Draw displacement vectors (subsampled)
    step = max(1, len(nodes) // 50)
    indices = range(0, len(nodes), step)

    for i in indices:
        ax.quiver(
            nodes[i, 0],
            nodes[i, 1],
            nodes[i, 2],
            mode_shape[i, 0] * amplitude_scale,
            mode_shape[i, 1] * amplitude_scale,
            mode_shape[i, 2] * amplitude_scale,
            color="yellow",
            alpha=0.6,
            arrow_length_ratio=0.3,
        )

    if config.show_colorbar:
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label("Displacement Magnitude")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved modal analysis to {output_path}")

    return fig
