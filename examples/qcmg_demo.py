#!/usr/bin/env python3
"""
Demonstration of the QCMG Field Evolution Module.

This example shows how to:
1. Initialize a QCMG field with different modes
2. Evolve the field over time
3. Monitor coherence, entropy, and energy
4. Export state for analysis
"""

from quasim.qcmg import QCMGParameters, QuantacosmorphysigeneticField


def main():
    """Run QCMG field evolution demonstration."""
    print("=" * 70)
    print("Quantacosmorphysigenetic Field Evolution Demo")
    print("=" * 70)

    # Configure field parameters
    params = QCMGParameters(
        grid_size=64,
        spatial_extent=10.0,
        dt=0.01,
        coupling_strength=0.1,
        random_seed=42,
    )

    # Initialize field with Gaussian mode
    field = QuantacosmorphysigeneticField(params)
    field.initialize(mode="gaussian")

    print(f"\nInitialized field with {params.grid_size} grid points")
    print(f"Spatial extent: {params.spatial_extent}")
    print(f"Time step: {params.dt}")

    # Get initial state
    state = field.get_state()
    print("\nInitial state:")
    print(f"  Time: {state.time:.3f}")
    print(f"  Coherence: {state.coherence:.4f}")
    print(f"  Entropy: {state.entropy:.4f}")
    print(f"  Energy: {state.energy:.4f}")

    # Evolve the field
    print("\nEvolving field...")
    print("-" * 70)
    print(f"{'Step':<8} {'Time':<12} {'Coherence':<12} {'Entropy':<12} {'Energy':<12}")
    print("-" * 70)

    for i in range(100):
        state = field.evolve()

        # Print every 10 steps
        if i % 10 == 0:
            print(
                f"{i:<8} {state.time:<12.3f} {state.coherence:<12.4f} "
                f"{state.entropy:<12.4f} {state.energy:<12.4f}"
            )

    print("-" * 70)

    # Export final state
    export = field.export_state()
    print("\nFinal state export:")
    print(f"  QCMG version: {export['qcmg_version']}")
    print(f"  History length: {export['metadata']['history_length']}")
    print(f"  State bounded: {export['metadata']['bounded']}")

    # Demonstrate different initialization modes
    print("\n" + "=" * 70)
    print("Comparing initialization modes")
    print("=" * 70)

    modes = ["gaussian", "soliton", "random"]
    for mode in modes:
        field_test = QuantacosmorphysigeneticField(params)
        field_test.initialize(mode=mode)
        state_test = field_test.get_state()

        print(f"\n{mode.capitalize()} mode:")
        print(f"  Coherence: {state_test.coherence:.4f}")
        print(f"  Entropy: {state_test.entropy:.4f}")
        print(f"  Energy: {state_test.energy:.4f}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
