"""Example VQE workflow using the QuASIM simulation stack."""
from __future__ import annotations

from quantum.python.quasim_sim import simulate


def heisenberg_hamiltonian(n_qubits: int) -> list[np.ndarray]:
    gates = []
    for _ in range(n_qubits):
        gates.append([0 + 0j, 1 + 0j, 1 + 0j, 0 + 0j])
    return gates


def run_vqe(num_qubits: int = 4):
    circuit = heisenberg_hamiltonian(num_qubits)
    amplitudes = simulate(circuit)
    energy = float(sum(abs(value) ** 2 for value in amplitudes))
    return energy


if __name__ == "__main__":
    print(f"Estimated energy: {run_vqe():.3f}")
