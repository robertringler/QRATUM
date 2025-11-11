# quasim/quantum/lindblad.py
# Qubit layer: Pauli, unitary, dephasing Lindblad, Bures, QFI

import numpy as np

I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)


def dagger(M):
    return M.conj().T


def expm_2x2(H):
    vals, vecs = np.linalg.eig(H)
    return vecs @ np.diag(np.exp(vals)) @ np.linalg.inv(vecs)


def build_H(Omega, lam, mu_t):
    # H = 0.5 Ω σz + λ μ σx
    return 0.5 * Omega * sz + lam * mu_t * sx


def unitary_step(rho, H, dt):
    U = expm_2x2(-1j * H * dt)
    return U @ rho @ dagger(U)


def lindblad_dephase_step(rho, gamma, dt):
    # L[ρ] = γ(σz ρ σz - ρ)
    return rho + dt * (gamma * (sz @ rho @ sz - rho))


def mixed_step(rho, H, gamma, dt, split=0.5):
    # Strang splitting
    rho = unitary_step(rho, H, split * dt)
    rho = lindblad_dephase_step(rho, gamma, dt)
    rho = unitary_step(rho, H, split * dt)
    return rho


def bures_fidelity(rho, sigma):
    w, V = np.linalg.eigh(rho)
    sqrt_rho = V @ np.diag(np.sqrt(np.clip(w, 0, None))) @ dagger(V)
    M = sqrt_rho @ sigma @ sqrt_rho
    M = 0.5 * (M + dagger(M))
    w2, _ = np.linalg.eigh(M)
    return (np.sum(np.sqrt(np.clip(w2, 0, None)))) ** 2


def bures_distance(rho, sigma):
    F = np.clip(bures_fidelity(rho, sigma), 0.0, 1.0)
    return np.arccos(np.sqrt(F))


def sld_qfi_numeric(rho_mu_minus, rho_mu, rho_mu_plus, dmu):
    # ∂ρ = (ρL + Lρ)/2
    I2 = np.eye(2, dtype=complex)
    drho = (rho_mu_plus - rho_mu_minus) / (2 * dmu)
    A = 0.5 * (np.kron(I2, rho_mu.T) + np.kron(rho_mu, I2))
    b = drho.reshape(-1, order="F")
    L_vec, *_ = np.linalg.lstsq(A + 1e-10 * np.eye(4), b, rcond=None)
    L = L_vec.reshape(2, 2, order="F")
    Fq = np.real(np.trace(rho_mu @ (L @ L)))
    return max(Fq, 0.0)
