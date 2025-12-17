"""Analog Spacetime Metric Module.

Generates effective metric tensors from refractive-index maps for analog gravity simulations.
Based on analogy between wave propagation in media and curved spacetime.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.complex128]


def refractive_index_to_metric(
    refractive_index: NDArray[np.float64],
    flow_velocity: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Convert refractive index profile to effective metric tensor.

    For acoustic/optical analog systems, effective metric:
        g_μν = (ρ/c) diag(-1, n²(r), n²(r), n²(r))

    where n(r) is refractive index and ρ is density.

    Args:
        refractive_index: Spatial refractive index profile n(r)
        flow_velocity: Optional flow velocity field v(r)

    Returns:
        Effective metric tensor (simplified 2x2 for 1D+time)

    Example:
        >>> n = np.ones(64)
        >>> n[32:] = 1.5  # Index jump
        >>> g = refractive_index_to_metric(n)

    Reference:
        Unruh (1981), "Experimental Black-Hole Evaporation?"
    """

    n_points = len(refractive_index)

    # Simplified 2x2 metric (time, space)
    metric = np.zeros((n_points, 2, 2))

    for i in range(n_points):
        n = refractive_index[i]

        # Metric components
        g_tt = -1.0  # Time-time component
        g_xx = n**2  # Spatial component

        # Include flow if provided
        if flow_velocity is not None:
            v = flow_velocity[i]
            g_tx = -n * v  # Cross term from flow
            metric[i] = [[g_tt, g_tx], [g_tx, g_xx]]
        else:
            metric[i] = [[g_tt, 0], [0, g_xx]]

    return metric


def geodesic_propagation(
    initial_position: float,
    initial_momentum: float,
    metric: NDArray[np.float64],
    n_steps: int,
    dt: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Propagate particle/wave along geodesic in effective curved spacetime.

    Geodesic equation:
        d²x^μ/dτ² = -Γ^μ_νλ (dx^ν/dτ)(dx^λ/dτ)

    Simplified for null geodesics in analog metric.

    Args:
        initial_position: Starting position x₀
        initial_momentum: Initial momentum p₀
        metric: Effective metric tensor field
        n_steps: Number of integration steps
        dt: Time step

    Returns:
        Tuple of (positions, momenta) trajectories

    Example:
        >>> g = refractive_index_to_metric(np.ones(64))
        >>> x, p = geodesic_propagation(0.5, 1.0, g, 100, 0.01)
    """

    n_points = len(metric)

    positions = np.zeros(n_steps)
    momenta = np.zeros(n_steps)

    positions[0] = initial_position
    momenta[0] = initial_momentum

    for step in range(1, n_steps):
        x = positions[step - 1]
        p = momenta[step - 1]

        # Discrete position index
        idx = int(x * n_points) % n_points

        # Extract metric at current position
        g = metric[idx]

        # Simplified geodesic evolution (Hamilton's equations)
        # dx/dt = ∂H/∂p
        # dp/dt = -∂H/∂x

        # Velocity from metric
        g_inv = np.linalg.inv(g)
        v = g_inv[1, 1] * p  # Simplified

        # Update position
        positions[step] = x + v * dt

        # Update momentum (force from metric gradient)
        if idx < n_points - 1:
            g_next = metric[idx + 1]
            metric_gradient = (g_next[1, 1] - g[1, 1]) / (1.0 / n_points)
            force = -0.5 * metric_gradient * p**2
        else:
            force = 0

        momenta[step] = p + force * dt

        # Keep in bounds
        positions[step] = positions[step] % 1.0

    return positions, momenta


def check_normalization_preservation(
    wavefunction: Array,
    metric: NDArray[np.float64],
    tolerance: float = 1e-6,
) -> tuple[bool, float]:
    """Verify that geodesic propagation preserves wavefunction normalization.

    For valid metric, ∫|ψ|² √g dx = const.

    Args:
        wavefunction: Quantum wavefunction ψ(x)
        metric: Spacetime metric tensor
        tolerance: Maximum allowed deviation

    Returns:
        Tuple of (is_normalized, deviation):
            - is_normalized: True if norm preserved
            - deviation: Deviation from unity

    Example:
        >>> psi = np.ones(64, dtype=complex) / 8
        >>> g = refractive_index_to_metric(np.ones(64))
        >>> is_ok, dev = check_normalization_preservation(psi, g)
    """

    n_points = len(wavefunction)

    # Compute measure √|g| for each point
    sqrt_g = np.zeros(n_points)
    for i in range(n_points):
        g = metric[i]
        det_g = np.linalg.det(g)
        sqrt_g[i] = np.sqrt(np.abs(det_g))

    # Integrate |ψ|² √g dx
    integrand = np.abs(wavefunction) ** 2 * sqrt_g
    norm = np.sum(integrand) / n_points  # Normalized by grid spacing

    deviation = abs(norm - 1.0)
    is_normalized = deviation < tolerance

    return is_normalized, float(deviation)


def effective_potential_from_metric(
    metric: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Extract effective potential from metric tensor.

    In analog systems, metric curvature creates effective potential:
        V_eff(x) ∝ ∂_x g_μν

    Args:
        metric: Metric tensor field

    Returns:
        Effective potential energy V_eff(x)

    Example:
        >>> g = refractive_index_to_metric(np.linspace(1, 2, 64))
        >>> V = effective_potential_from_metric(g)
    """

    n_points = len(metric)
    potential = np.zeros(n_points)

    for i in range(n_points):
        if i < n_points - 1:
            # Spatial gradient of spatial metric component
            g_xx_curr = metric[i, 1, 1]
            g_xx_next = metric[i + 1, 1, 1]
            grad_g = (g_xx_next - g_xx_curr) * n_points

            # Effective potential
            potential[i] = 0.5 * grad_g
        else:
            potential[i] = potential[i - 1]

    return potential


def acoustic_black_hole_metric(
    n_points: int,
    horizon_position: float = 0.5,
    flow_strength: float = 1.0,
) -> NDArray[np.float64]:
    """Generate analog acoustic black hole metric.

    Models acoustic analog of black hole using supersonic flow.

    Args:
        n_points: Number of spatial grid points
        horizon_position: Location of sonic horizon (0 to 1)
        flow_strength: Strength of flow velocity

    Returns:
        Acoustic black hole metric tensor

    Example:
        >>> g = acoustic_black_hole_metric(128, 0.5, 1.5)

    Reference:
        Unruh (1981), Barcelo et al. (2005)
    """

    x = np.linspace(0, 1, n_points)

    # Flow velocity increasing toward horizon
    v = flow_strength * np.tanh(10 * (x - horizon_position))

    # Refractive index (inverse of sound speed)
    n = np.ones(n_points) / (1.0 + 0.5 * v**2)

    # Generate metric with flow
    metric = refractive_index_to_metric(n, v)

    return metric
