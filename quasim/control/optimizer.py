# quasim/control/optimizer.py
# Coupled simulation + control over geometric scale a(t)

import numpy as np

from quasim.info.geometry import fr_speed, free_energy, gaussian_w2, ou_step
from quasim.quantum.lindblad import (
    build_H,
    bures_distance,
    bures_fidelity,
    mixed_step,
    sld_qfi_numeric,
)


def simulate(
    a,
    T=2.0,
    N=400,
    Omega=2.0,
    lam=0.6,
    gamma=0.3,
    omega=1.0,
    mu0=1.0,
    sigma0=1.0,
    rho0=None,
    alpha=1.0,
    beta=0.1,
    gamma_loss=0.1,
    delta=0.01,
):
    """
    Run the coupled classical–quantum flow for a fixed geometric schedule a(t).
    Returns: total objective, logs dict
    """
    dt = T / N
    if rho0 is None:
        rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)

    mu, sigma = mu0, sigma0
    rho = rho0.copy()

    JF = JI = JQ = JG = 0.0

    w2_list = []
    fr_list = []
    bures_list = []
    Fq_list = []
    F_list = []
    FE_list = []

    eps_mu = 1e-3

    for k in range(N):
        Hk = build_H(Omega, lam, mu)
        rho_next = mixed_step(rho, Hk, gamma, dt)

        mu_next, sigma_next = ou_step(mu, sigma, dt, omega=omega, D=1.0)
        sigma_next = max(1e-6, a[k] * sigma_next)  # geometric rescale

        w2 = gaussian_w2(mu, sigma, mu_next, sigma_next)
        fr = fr_speed(mu, sigma, mu_next, sigma_next, dt)
        bd = bures_distance(rho, rho_next)
        FE = free_energy(mu, sigma, omega=omega)

        # QFI wrt μ by perturbing Hamiltonian coupling
        H_minus = build_H(Omega, lam, mu - eps_mu)
        H_plus = build_H(Omega, lam, mu + eps_mu)
        rho_minus = mixed_step(rho, H_minus, gamma, dt)
        rho_plus = mixed_step(rho, H_plus, gamma, dt)
        Fq = sld_qfi_numeric(rho_minus, rho, rho_plus, eps_mu)

        JF += alpha * FE * dt
        JI += beta * 0.5 * (fr**2) * dt
        JQ += gamma_loss * (1.0 - Fq) * dt
        JG += delta * (a[k] - 1.0) ** 2 * dt

        w2_list.append(w2)
        fr_list.append(fr)
        bures_list.append(bd)
        Fq_list.append(Fq)
        F_list.append(bures_fidelity(rho, rho_next))
        FE_list.append(FE)

        mu, sigma, rho = mu_next, sigma_next, rho_next

    J = JF + JI + JQ + JG
    logs = {
        "W2": np.array(w2_list),
        "FR_speed": np.array(fr_list),
        "Bures_dist": np.array(bures_list),
        "QFI_mu": np.array(Fq_list),
        "Fidelity_step": np.array(F_list),
        "FreeEnergy": np.array(FE_list),
    }
    return J, logs


def optimize_a(
    steps=50,
    N=400,
    lr=5e-2,
    seed=0,
    **sim_kwargs,
):
    """
    Finite-difference control over the geometric scale schedule a(t).
    """
    np.random.default_rng(seed)
    a = np.ones(N, dtype=float)
    hist = []

    for it in range(steps):
        J0, _ = simulate(a, N=N, **sim_kwargs)
        grad = np.zeros_like(a)

        probe_idx = np.arange(0, N, max(1, N // 50))
        eps = 1e-2
        for k in probe_idx:
            a[k] += eps
            Jp, _ = simulate(a, N=N, **sim_kwargs)
            grad[k] = (Jp - J0) / eps
            a[k] -= eps

        grad = np.interp(np.arange(N), probe_idx, grad[probe_idx])

        a -= lr * grad
        a = np.clip(a, 1e-2, 5.0)

        J, logs = simulate(a, N=N, **sim_kwargs)
        hist.append((J, a.copy()))

        if it > 5 and abs(hist[-1][0] - hist[-6][0]) < 1e-4:
            break

    return a, hist, logs
