#!/usr/bin/env python3
"""
Basic QCMG Field Simulation Example

Demonstrates initialization, evolution, and analysis of the
Quantacosmorphysigenetic field system.
"""

import json

import matplotlib.pyplot as plt

from quasim.sim import QCMGParameters, QuantacosmorphysigeneticField


def main():
    """Run basic QCMG simulation."""

    # Configure simulation
    print("Initializing QCMG field simulation...")
    params = QCMGParameters(
        grid_size=64,
        dt=0.01,
        coupling_strength=0.1,
        interaction_strength=0.05,
        thermal_noise=0.001,
        random_seed=42,
    )

    # Create field
    field = QuantacosmorphysigeneticField(params)
    field.initialize(mode="gaussian")

    print(f"Grid size: {params.grid_size}")
    print(f"Time step: {params.dt}")
    print(f"Coupling strength: {params.coupling_strength}")

    # Initial state
    initial_state = field.get_state()
    print("\nInitial state:")
    print(f"  Coherence: {initial_state.coherence:.4f}")
    print(f"  Entropy: {initial_state.entropy:.4f}")
    print(f"  Energy: {initial_state.energy:.4f}")

    # Evolve system
    print("\nEvolving for 200 time steps...")
    num_steps = 200

    for i in range(num_steps):
        state = field.evolve()

        if (i + 1) % 50 == 0:
            print(
                f"  Step {i + 1:3d}: C={state.coherence:.4f}, "
                f"S={state.entropy:.4f}, E={state.energy:.4f}"
            )

    # Final state
    final_state = field.get_state()
    print("\nFinal state:")
    print(f"  Coherence: {final_state.coherence:.4f}")
    print(f"  Entropy: {final_state.entropy:.4f}")
    print(f"  Energy: {final_state.energy:.4f}")

    # Analyze evolution
    history = field.get_history()
    times = [s.time for s in history]
    coherences = [s.coherence for s in history]
    entropies = [s.entropy for s in history]
    energies = [s.energy for s in history]

    print("\nEvolution statistics:")
    print(f"  Coherence: {min(coherences):.4f} to {max(coherences):.4f}")
    print(f"  Entropy: {min(entropies):.4f} to {max(entropies):.4f}")
    print(f"  Energy: {min(energies):.4f} to {max(energies):.4f}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Coherence evolution
    axes[0, 0].plot(times, coherences, "b-", linewidth=2)
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Coherence C(t)")
    axes[0, 0].set_title("Quantum Coherence Dynamics")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])

    # Entropy evolution
    axes[0, 1].plot(times, entropies, "r-", linewidth=2)
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Entropy S(t)")
    axes[0, 1].set_title("Entropy Production")
    axes[0, 1].grid(True, alpha=0.3)

    # Energy evolution
    axes[1, 0].plot(times, energies, "g-", linewidth=2)
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Energy E(t)")
    axes[1, 0].set_title("Field Energy")
    axes[1, 0].grid(True, alpha=0.3)

    # Phase space: Coherence vs Entropy
    axes[1, 1].plot(coherences, entropies, "k-", linewidth=1, alpha=0.5)
    axes[1, 1].scatter(
        [coherences[0]], [entropies[0]], c="green", s=100, marker="o", label="Start", zorder=5
    )
    axes[1, 1].scatter(
        [coherences[-1]], [entropies[-1]], c="red", s=100, marker="s", label="End", zorder=5
    )
    axes[1, 1].set_xlabel("Coherence C")
    axes[1, 1].set_ylabel("Entropy S")
    axes[1, 1].set_title("Phase Space: C-S Dynamics")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("qcmg_evolution.png", dpi=150)
    print("\nPlot saved to qcmg_evolution.png")

    # Export final state
    export_data = field.export_state()

    with open("qcmg_results.json", "w") as f:
        json.dump(export_data, f, indent=2)

    print("Results exported to qcmg_results.json")

    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
