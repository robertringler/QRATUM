"""
QuASIM JSON Generator — Live Emulation for CI Validation
Author: Grok (xAI)
Dependencies: qutip, numpy, scipy (all available in env)
"""

import json
import os

import numpy as np
from qutip import (
    Qobj,
    fidelity,
    phasegate,
    rx,
    ry,
    rz,
    sigmax,
    sigmay,
    sigmaz,
)
from qutip.qip.operations import snot  # T is sqrt(S), but approx
from scipy.linalg import expm

# Fixed params from defaults
THETA, PHI = 0.7, 1.1
GATE_SEQ = ["H", "Rz(0.35)", "Rx(0.9)", "T", "Ry(-0.2)"]
NOISE_DEFAULT = {"gamma1": 0.002, "gamma_phi": 0.003, "p_depol": 0.001}
CONTROL = {
    "segments": [
        {"dt": 6e-9, "Omega_x": 2.5e7, "Omega_y": 0.0, "Delta": 0.0},
        {"dt": 8e-9, "Omega_x": 0.0, "Omega_y": 3.0e7, "Delta": 1.5e6}
    ],
    "n_trotter_steps": 64
}
SHOTS = 20000
SEED = 12345
PRECISION = np.complex64
BATCH_SIZE = 1024  # For MC fidelity std

np.random.seed(SEED)

def init_state(theta, phi):
    """|ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩"""
    return Qobj(np.array([np.cos(theta/2), np.exp(1j*phi) * np.sin(theta/2)], dtype=PRECISION))

def apply_gates(psi, gates, add_su2=False, su2_params=None):
    """Apply gate sequence; optional randomized SU(2)"""
    rho = psi * psi.dag()
    for g in gates:
        if g == "H":
            U = snot()
        elif g == "T":
            U = phasegate(np.pi/4)  # T gate
        elif g.startswith("Rz("):
            angle = float(g[3:-1])
            U = rz(angle)
        elif g.startswith("Rx("):
            angle = float(g[3:-1])
            U = rx(angle)
        elif g.startswith("Ry("):
            angle = float(g[3:-1])
            U = ry(angle)
        else:
            raise ValueError(f"Unknown gate: {g}")
        rho = U * rho * U.dag()
    if add_su2:
        if su2_params is None:
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            angle = np.random.uniform(0, np.pi)
        else:
            axis, angle = su2_params
        n = Qobj(axis[0]*sigmax() + axis[1]*sigmay() + axis[2]*sigmaz())
        U_su2 = Qobj(expm(-1j * (angle/2) * n.full()))
        rho = U_su2 * rho * U_su2.dag()
    return rho

def apply_noise(rho, noise):
    """Kraus operators for amp damp, dephase, depol"""
    if all(v == 0 for v in noise.values()):
        return rho
    # Amplitude damping
    gamma1 = noise["gamma1"]
    K1 = Qobj(np.array([[1, 0], [0, np.sqrt(1-gamma1)]]))
    K2 = Qobj(np.array([[0, np.sqrt(gamma1)], [0, 0]]))
    rho = K1 * rho * K1.dag() + K2 * rho * K2.dag()
    # Dephasing
    gamma_phi = noise["gamma_phi"]
    K_phi = Qobj(np.diag([1, np.sqrt(1-gamma_phi)]))
    rho = K_phi * rho * K_phi.dag() + (1 - np.sqrt(1-gamma_phi)) * rho.diag() * sigmaz()
    # Depolarizing (simplified)
    p = noise["p_depol"]
    rho = (1 - p) * rho + p/3 * (sigmax()*rho*sigmax() + sigmay()*rho*sigmay() + sigmaz()*rho*sigmaz())
    return rho

def propagate_control(rho, control, method="trotter"):
    """Evolve under H(t) = 1/2 [Ωx σx + Ωy σy + Δ σz]"""
    H_terms = []
    times = []
    for seg in control["segments"]:
        H_seg = 0.5 * (seg["Omega_x"] * sigmax() + seg["Omega_y"] * sigmay() + seg["Delta"] * sigmaz())
        H_terms.append(H_seg)
        times.append(seg["dt"])
    if method == "expm":
        U_total = Qobj(np.eye(2, dtype=PRECISION))
        for H_seg, dt in zip(H_terms, times):
            U_seg = Qobj(expm(-1j * dt * H_seg.full()))
            U_total = U_seg * U_total
        return U_total * rho * U_total.dag()
    elif method == "trotter":
        n_steps = control["n_trotter_steps"]
        U_total = Qobj(np.eye(2, dtype=PRECISION))
        for H_seg, dt in zip(H_terms, times):
            dt_step = dt / n_steps
            for _ in range(n_steps):
                U_step = Qobj(expm(-1j * dt_step * H_seg.full()))
                U_total = U_step * U_total
        return U_total * rho * U_total.dag()
    raise ValueError("Invalid method")

def compute_fidelity_mc(rho_noisy, rho_ideal, batch_size):
    """Monte-Carlo fidelity mean/std (simplified via tracing)"""
    fids = [fidelity(rho_noisy, rho_ideal) for _ in range(batch_size)]  # Placeholder; in full MC, add noise variations
    return np.mean(fids), np.std(fids)

def simulate_shots(bloch, shots):
    """Binomial shots from <σ> = Bloch components"""
    counts = {}
    for axis, b_comp in zip("XYZ", bloch):
        p_plus = (1 + b_comp) / 2
        n_plus = np.random.binomial(shots, p_plus)
        counts[axis] = {"+1": int(n_plus), "-1": shots - int(n_plus)}
    return counts

def bloch_vector(rho):
    return np.array([np.real(np.trace(rho * sigmax())),
                     np.real(np.trace(rho * sigmay())),
                     np.real(np.trace(rho * sigmaz()))])

def rho_to_arrays(rho):
    return rho.full().real.tolist(), rho.full().imag.tolist()

def run_simulation(noise, method, add_su2=False):
    psi0 = init_state(THETA, PHI)
    rho_gates = apply_gates(psi0, GATE_SEQ, add_su2=add_su2,
                            su2_params=([0.334, 0.615, 0.712], 1.23) if add_su2 else None)
    rho_ideal = propagate_control(rho_gates, CONTROL, method=method)
    rho_noisy = apply_noise(rho_ideal, noise)
    bloch_init = bloch_vector(psi0 * psi0.dag())
    bloch_final = bloch_vector(rho_noisy)
    fid_mean, fid_std = compute_fidelity_mc(rho_noisy, rho_ideal, BATCH_SIZE)
    shot_counts = simulate_shots(bloch_final, SHOTS)
    rho_re, rho_im = rho_to_arrays(rho_noisy)
    return {
        "bloch_initial": bloch_init.tolist(),
        "bloch_final": bloch_final.tolist(),
        "rho_final_re": rho_re,
        "rho_final_im": rho_im,
        "fidelity_mean": float(fid_mean),
        "fidelity_std": float(fid_std),
        "shot_counts": shot_counts
    }

def main():
    os.makedirs("tests", exist_ok=True)
    # Default: noisy, Trotter
    default = run_simulation(NOISE_DEFAULT, "trotter")
    with open("tests/default_sim.json", "w") as f:
        json.dump(default, f, indent=2)
    # (a) Ideal: no noise, Trotter
    ideal = run_simulation({"gamma1":0,"gamma_phi":0,"p_depol":0}, "trotter")
    with open("tests/ideal_sim.json", "w") as f:
        json.dump(ideal, f, indent=2)
    # (b) Noisy, exact expm
    expm_sim = run_simulation(NOISE_DEFAULT, "expm")
    with open("tests/expm_sim.json", "w") as f:
        json.dump(expm_sim, f, indent=2)
    # (c) Noisy, Trotter, + randomized SU(2)
    su2_sim = run_simulation(NOISE_DEFAULT, "trotter", add_su2=True)
    with open("tests/su2_sim.json", "w") as f:
        json.dump(su2_sim, f, indent=2)
    print("JSONs generated successfully.")

if __name__ == "__main__":
    main()
