"""Fractional Dynamics and Anomalous Diffusion Module.

Implements fractional Laplacian and fractional-order differential operators
for anomalous transport and fractal systems.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.complex128]


def fractional_laplacian_fft(
    wavefunction: Array,
    alpha: float,
    dx: float,
) -> Array:
    """Compute fractional Laplacian (-∇²)^α via FFT spectral method.
    
    Fractional Laplacian in Fourier space:
        ℱ[(-∇²)^α ψ] = |k|^(2α) ℱ[ψ]
    
    Args:
        wavefunction: Spatial wavefunction ψ(x)
        alpha: Fractional exponent (0.5 = half-derivative, 1.0 = Laplacian)
        dx: Spatial grid spacing
        
    Returns:
        (-∇²)^α ψ in position space
        
    Example:
        >>> psi = np.sin(2*np.pi*np.arange(64)/64) + 0j
        >>> laplacian_frac = fractional_laplacian_fft(psi, 0.5, 0.1)
        
    Reference:
        Laskin (2000), "Fractional quantum mechanics"
    """
    n = len(wavefunction)
    
    # Fourier transform
    psi_k = np.fft.fft(wavefunction)
    
    # Wavevectors
    k = 2 * np.pi * np.fft.fftfreq(n, dx)
    
    # Apply fractional Laplacian in Fourier space
    psi_k_frac = (np.abs(k) ** (2 * alpha)) * psi_k
    
    # Inverse transform
    result = np.fft.ifft(psi_k_frac)
    
    return result


def fractional_schrodinger_step(
    wavefunction: Array,
    alpha: float,
    potential: NDArray[np.float64],
    dx: float,
    dt: float,
) -> Array:
    """Evolve wavefunction using fractional Schrödinger equation.
    
    Fractional Schrödinger equation:
        iℏ ∂ψ/∂t = (-∇²)^α ψ + V ψ
    
    Args:
        wavefunction: Current wavefunction ψ
        alpha: Fractional order
        potential: Potential energy V(x)
        dx: Spatial grid spacing
        dt: Time step
        
    Returns:
        Updated wavefunction after time dt
        
    Example:
        >>> psi = np.ones(64, dtype=complex) / 8
        >>> V = np.zeros(64)
        >>> psi_new = fractional_schrodinger_step(psi, 1.0, V, 0.1, 0.01)
    """
    # Kinetic energy term (fractional Laplacian)
    kinetic = fractional_laplacian_fft(wavefunction, alpha, dx)
    
    # Potential energy term
    potential_term = potential * wavefunction
    
    # Hamiltonian action
    H_psi = kinetic + potential_term
    
    # Time evolution (first-order)
    psi_new = wavefunction - 1j * dt * H_psi
    
    # Normalize
    psi_new /= np.linalg.norm(psi_new)
    
    return psi_new


def anomalous_diffusion_propagator(
    x: NDArray[np.float64],
    t: float,
    alpha: float,
    diffusion_coeff: float = 1.0,
) -> NDArray[np.float64]:
    """Compute propagator for fractional diffusion equation.
    
    Fractional diffusion:
        ∂ρ/∂t = D_α (-∇²)^α ρ
    
    Solution (Green's function):
        G(x,t) ∝ t^(-d/2α) exp(-|x|^β / (D_α t)^(1/α))
    
    Args:
        x: Spatial positions
        t: Time
        alpha: Fractional exponent
        diffusion_coeff: Diffusion coefficient D_α
        
    Returns:
        Propagator G(x,t)
        
    Example:
        >>> x = np.linspace(-10, 10, 200)
        >>> G = anomalous_diffusion_propagator(x, 1.0, 0.75, 1.0)
        
    Reference:
        Metzler & Klafter (2000), "The random walk's guide to anomalous diffusion"
    """
    # Simplified form for 1D
    if t < 1e-10:
        # Delta function at t=0
        G = np.zeros_like(x)
        G[len(G) // 2] = 1.0 / (x[1] - x[0]) if len(x) > 1 else 1.0
        return G
    
    # Anomalous diffusion exponent
    beta = 2 * alpha
    
    # Scaling
    width = (diffusion_coeff * t) ** (1.0 / alpha)
    
    # Propagator (Gaussian-like with power-law tails)
    G = np.exp(-(np.abs(x) ** beta) / (beta * width**beta))
    
    # Normalize
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    G /= np.sum(G) * dx
    
    return G


def measure_diffusion_exponent(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
) -> tuple[float, float]:
    """Extract anomalous diffusion exponent from mean-square displacement.
    
    Mean-square displacement:
        ⟨x²(t)⟩ ∝ t^β
    
    where β = 1 (normal), β < 1 (subdiffusion), β > 1 (superdiffusion).
    
    Args:
        positions: Trajectory positions x(t)
        times: Time points
        
    Returns:
        Tuple of (exponent_beta, diffusion_coefficient):
            - exponent_beta: Fitted exponent β
            - diffusion_coefficient: Generalized diffusion constant
            
    Example:
        >>> t = np.linspace(0, 10, 100)[1:]  # Exclude t=0
        >>> x = np.sqrt(t) + 0.1*np.random.randn(len(t))  # β ≈ 0.5
        >>> beta, D = measure_diffusion_exponent(x, t)
        >>> assert abs(beta - 0.5) < 0.2
    """
    # Mean-square displacement
    msd = positions**2
    
    # Exclude t=0 to avoid log(0)
    valid = times > 0
    log_t = np.log(times[valid])
    log_msd = np.log(msd[valid] + 1e-10)  # Add small offset to avoid log(0)
    
    # Fit log(MSD) = log(D) + β log(t)
    if len(log_t) > 1:
        coeffs = np.polyfit(log_t, log_msd, 1)
        exponent_beta = coeffs[0]
        log_D = coeffs[1]
        D = np.exp(log_D)
    else:
        exponent_beta = 1.0
        D = 1.0
    
    return float(exponent_beta), float(D)


def levy_flight_step(
    current_position: float,
    alpha: float,
    scale: float = 1.0,
) -> float:
    """Generate single step of Lévy flight for anomalous transport.
    
    Lévy distribution with power-law tails:
        P(Δx) ∝ |Δx|^(-(1+α))
    
    Args:
        current_position: Current position x
        alpha: Lévy exponent (0 < α ≤ 2)
        scale: Scale parameter
        
    Returns:
        New position after Lévy flight
        
    Example:
        >>> x = 0.0
        >>> x_new = levy_flight_step(x, 1.5, 1.0)
    """
    # Use Chambers-Mallows-Stuck algorithm
    rng = np.random.RandomState()
    
    if alpha == 2.0:
        # Gaussian limit
        step = rng.normal(0, scale)
    else:
        # Simplified Lévy: use Pareto-like distribution
        u = rng.uniform(-np.pi / 2, np.pi / 2)
        w = rng.exponential(1.0)
        
        # Lévy step
        if alpha != 1.0:
            step = scale * (
                np.sin(alpha * u)
                / (np.cos(u) ** (1.0 / alpha))
                * (np.cos((1 - alpha) * u) / w) ** ((1 - alpha) / alpha)
            )
        else:
            step = scale * np.tan(u)
    
    return float(current_position + step)


def fractal_dimension_capacity(
    positions: NDArray[np.float64],
    box_sizes: NDArray[np.float64],
) -> float:
    """Compute capacity (box-counting) fractal dimension.
    
    Fractal dimension:
        D = -lim_{ε→0} log(N(ε)) / log(ε)
    
    where N(ε) is number of boxes of size ε needed to cover the set.
    
    Args:
        positions: Set of points in 1D
        box_sizes: Range of box sizes to test
        
    Returns:
        Fractal dimension D
        
    Example:
        >>> x = np.random.randn(1000)  # Random 1D points
        >>> sizes = np.logspace(-2, 0, 10)
        >>> D = fractal_dimension_capacity(x, sizes)
    """
    counts = []
    
    for epsilon in box_sizes:
        if epsilon > 0:
            # Count boxes
            x_min, x_max = np.min(positions), np.max(positions)
            n_boxes = int(np.ceil((x_max - x_min) / epsilon))
            counts.append(n_boxes)
        else:
            counts.append(1)
    
    # Fit log(N) vs log(1/ε)
    log_epsilon = np.log(box_sizes)
    log_counts = np.log(np.array(counts) + 1)
    
    # Slope gives -D
    if len(log_epsilon) > 1:
        coeffs = np.polyfit(log_epsilon, log_counts, 1)
        D = -coeffs[0]
    else:
        D = 1.0
    
    return float(D)
