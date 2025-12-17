# quasim/info/geometry.py
# Geometric + statistical layer: OU flow, FR metric, Wasserstein, free energy

import numpy as np


def ou_step(mu, sigma, dt, omega=1.0, D=1.0):
    """Ornstein–Uhlenbeck step on (mu, sigma)."""

    mu_new = mu + dt * (-(omega**2) * mu)
    var = sigma**2
    var_new = var + dt * (2 * D - 2 * omega**2 * var)
    sigma_new = np.sqrt(max(var_new, 1e-12))
    return mu_new, sigma_new


def gaussian_w2(mu1, s1, mu2, s2):
    """1D W2^2 between Gaussians."""

    return (mu1 - mu2) ** 2 + (s1 + s2 - 2 * np.sqrt(max(s1, 0) * max(s2, 0)))


def fisher_rao_metric(mu, sigma):
    """FR metric for (mu, log sigma) parametrization."""

    eta = np.log(sigma)
    return np.array([[np.exp(-2 * eta), 0.0], [0.0, 2.0]])


def fr_speed(mu_t, sigma_t, mu_tp1, sigma_tp1, dt):
    """Arc-speed between two Gaussian params under FR metric."""

    g = fisher_rao_metric(mu_t, sigma_t)
    dtheta = np.array([(mu_tp1 - mu_t), (np.log(sigma_tp1) - np.log(sigma_t))])
    return float(np.sqrt(dtheta @ g @ dtheta) / max(dt, 1e-12))


def free_energy(mu, sigma, omega=1.0):
    """Simple Gaussian free energy = entropy + potential energy."""

    H = 0.5 * np.log(2 * np.pi * np.e * sigma**2)
    EV = 0.5 * omega**2 * (mu**2 + sigma**2)
    return H + EV
