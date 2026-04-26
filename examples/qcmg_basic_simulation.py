#!/usr/bin/env python3
"""Basic QCMG field simulation example with visualization.

This script demonstrates:
- Setting up a QCMG simulation
- Running field evolution
- Computing observables
- Visualizing results
- Exporting data
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quasim.sim import QCMGParameters, QuantacosmomorphysigeneticField


def run_basic_simulation():
    """Run a basic QCMG field simulation."""

    print("=" * 60)
    print("QCMG Field Simulation Example")
    print("=" * 60)

    # Configure simulation parameters
    params = QCMGParameters(
        grid_size=64,
        dt=0.01,
        coupling_strength=1.0,
        dissipation_rate=0.01,
        random_seed=42,  # For reproducibility
    )

    print("\nParameters:")
    print(f"  Grid size: {params.grid_size}")
    print(f"  Time step: {params.dt}")
    print(f"  Coupling: {params.coupling_strength}")
    print(f"  Dissipation: {params.dissipation_rate}")
    print(f"  Random seed: {params.random_seed}")

    # Initialize field with Gaussian wave packet
    print("\nInitializing field with Gaussian mode...")
    field = QuantacosmomorphysigeneticField(params)
    field.initialize(mode="gaussian")

    # Print initial state
    initial_state = field.history[-1]
    print("\nInitial state:")
    print(f"  Coherence: {initial_state.coherence:.6f}")
    print(f"  Entropy: {initial_state.entropy:.6f}")
    print(f"  Energy: {initial_state.energy:.6f}")

    # Evolve field
    n_steps = 200
    print(f"\nEvolving for {n_steps} steps...")

    for i in range(n_steps):
        state = field.evolve(steps=1)

        # Print progress every 50 steps
        if (i + 1) % 50 == 0:
            print(
                f"  Step {i + 1:3d}: C={state.coherence:.4f}, "
                f"S={state.entropy:.4f}, E={state.energy:.4f}"
            )

    # Print final state
    final_state = field.history[-1]
    print("\nFinal state:")
    print(f"  Time: {final_state.time:.4f}")
    print(f"  Coherence: {final_state.coherence:.6f}")
    print(f"  Entropy: {final_state.entropy:.6f}")
    print(f"  Energy: {final_state.energy:.6f}")

    return field


def visualize_results(field: QuantacosmomorphysigeneticField):
    """Create visualization of simulation results."""

    print("\nCreating visualizations...")

    # Extract data from history
    times = [state.time for state in field.history]
    coherences = [state.coherence for state in field.history]
    entropies = [state.entropy for state in field.history]
    energies = [state.energy for state in field.history]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("QCMG Field Simulation Results", fontsize=16, fontweight="bold")

    # Plot 1: Field amplitudes at final time
    ax1 = axes[0, 0]
    x = np.linspace(0, 2 * np.pi, field.params.grid_size, endpoint=False)
    final_state = field.history[-1]

    ax1.plot(x, np.abs(final_state.phi_m), "b-", label="|Φ_m|", linewidth=2)
    ax1.plot(x, np.abs(final_state.phi_i), "r--", label="|Φ_i|", linewidth=2)
    ax1.set_xlabel("Position x", fontsize=12)
    ax1.set_ylabel("Field Amplitude", fontsize=12)
    ax1.set_title("Final Field Amplitudes", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Coherence evolution
    ax2 = axes[0, 1]
    ax2.plot(times, coherences, "g-", linewidth=2)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.set_ylabel("Coherence C(t)", fontsize=12)
    ax2.set_title("Coherence Dynamics", fontsize=13, fontweight="bold")
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.5, color="k", linestyle=":", alpha=0.5, label="C=0.5")

    # Plot 3: Entropy evolution
    ax3 = axes[1, 0]
    ax3.plot(times, entropies, "m-", linewidth=2)
    ax3.set_xlabel("Time", fontsize=12)
    ax3.set_ylabel("Entropy S(t)", fontsize=12)
    ax3.set_title("Entropy Production", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Energy evolution
    ax4 = axes[1, 1]
    ax4.plot(times, energies, "c-", linewidth=2)
    ax4.set_xlabel("Time", fontsize=12)
    ax4.set_ylabel("Energy E(t)", fontsize=12)
    ax4.set_title("Energy Dissipation", fontsize=13, fontweight="bold")
    ax4.grid(True, alpha=0.3)

    # Add exponential decay fit for energy
    if len(times) > 10:
        # Fit E(t) = E0 * exp(-gamma * t)
        E0 = energies[0]
        gamma = field.params.dissipation_rate * 2  # Theoretical decay rate
        E_fit = E0 * np.exp(-gamma * np.array(times))
        ax4.plot(
            times,
            E_fit,
            "k--",
            linewidth=1.5,
            alpha=0.7,
            label="Theory: $E_0 \\, \\exp(-2\\gamma t)$",
        )
        ax4.legend(fontsize=10)

    plt.tight_layout()

    # Save figure
    output_dir = Path("qcmg_output")
    output_dir.mkdir(exist_ok=True)

    figure_path = output_dir / "qcmg_simulation.png"
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    print(f"  Saved figure to: {figure_path}")

    plt.show()


def export_data(field: QuantacosmomorphysigeneticField):
    """Export simulation data to JSON."""

    output_dir = Path("qcmg_output")
    output_dir.mkdir(exist_ok=True)

    json_path = output_dir / "qcmg_results.json"
    field.save_to_json(str(json_path), include_history=True)
    print(f"  Saved data to: {json_path}")


def main():
    """Main execution function."""

    # Run simulation
    field = run_basic_simulation()

    # Visualize results
    try:
        visualize_results(field)
    except ImportError:
        print("\nWarning: matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"\nWarning: visualization failed: {e}")

    # Export data
    export_data(field)

    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
