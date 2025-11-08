#!/usr/bin/env python3
"""Example usage of QuASIM distributed quantum simulation.

This script demonstrates:
1. Multi-qubit entangled state creation
2. Tensor-network simulation with GPU backends
3. Distributed cluster execution
4. Noise modeling and fidelity analysis
"""

from __future__ import annotations

import numpy as np

from quasim.qc.quasim_dist import (init_cluster, initialize_zero_state,
                                   profile, shard_state)
from quasim.qc.quasim_multi import (MultiQubitSimulator, create_bell_plus,
                                    create_ghz_state_exact)
from quasim.qc.quasim_tn import TensorNetworkEngine


def example_bell_state():
    """Example: Create and verify Bell pair."""
    print("\n" + "=" * 60)
    print("Example 1: Bell State Creation")
    print("=" * 60)

    sim = MultiQubitSimulator(num_qubits=2, seed=42)
    sim.initialize_state()
    print("Initialized 2-qubit system")

    # Create Bell pair using gates
    sim.apply_gate("H", [0])
    sim.apply_gate("CNOT", [0, 1])
    print("Applied H⊗I and CNOT")

    # Verify fidelity
    target = create_bell_plus()
    fidelity = sim.compute_fidelity(target)
    print(f"Fidelity with |Φ+⟩: {fidelity:.6f}")

    # Compute entanglement
    entropy = sim.entanglement_entropy([0])
    print(f"Entanglement entropy: {entropy:.6f} bits")

    # Run with results
    results = sim.run()
    print(f"Final state dimension: {len(results['state_vector_real'])}")


def example_ghz_state():
    """Example: Create and verify GHZ state."""
    print("\n" + "=" * 60)
    print("Example 2: GHZ State for 4 Qubits")
    print("=" * 60)

    sim = MultiQubitSimulator(num_qubits=4, seed=42)
    sim.initialize_state()
    print("Initialized 4-qubit system")

    # Create GHZ state
    sim.create_ghz_state()
    print("Created GHZ state: (|0000⟩ + |1111⟩)/√2")

    # Verify fidelity
    target = create_ghz_state_exact(4)
    fidelity = sim.compute_fidelity(target)
    print(f"Fidelity with exact GHZ: {fidelity:.6f}")

    # Tomography
    results = sim.tomography()
    print(f"Purity: {results['purity']:.6f}")

    # Check probability distribution
    probs = np.abs(sim.state) ** 2
    print(f"P(|0000⟩) = {probs[0]:.4f}")
    print(f"P(|1111⟩) = {probs[-1]:.4f}")


def example_noise_simulation():
    """Example: Simulate noise effects on entangled state."""
    print("\n" + "=" * 60)
    print("Example 3: Noise Effects on Bell State")
    print("=" * 60)

    sim = MultiQubitSimulator(num_qubits=2, seed=42)
    sim.initialize_state()
    sim.create_bell_pair((0, 1))

    target = create_bell_plus()
    fidelity_before = sim.compute_fidelity(target)
    print(f"Fidelity before noise: {fidelity_before:.6f}")

    # Apply realistic noise
    noise = {
        "gamma1": 0.01,  # Amplitude damping
        "gamma_phi": 0.02,  # Phase damping
        "p_depol": 0.005,  # Depolarizing
    }
    sim.apply_noise(noise)

    fidelity_after = sim.compute_fidelity(target)
    print(f"Fidelity after noise:  {fidelity_after:.6f}")
    print(f"Fidelity loss: {(fidelity_before - fidelity_after):.6f}")

    # Compute tomography
    results = sim.tomography()
    print(f"Purity after noise: {results['purity']:.6f}")


def example_tensor_network():
    """Example: Tensor-network simulation."""
    print("\n" + "=" * 60)
    print("Example 4: Tensor-Network Simulation")
    print("=" * 60)

    # Initialize TN engine
    tn = TensorNetworkEngine(num_qubits=6, bond_dim=64, backend="numpy", seed=42)
    tn.initialize_state()
    print(f"Initialized {tn.num_qubits}-qubit TN with bond_dim={tn.bond_dim}")

    # Apply gate sequence
    tn.apply_gate("H", [0])
    tn.apply_gate("CNOT", [0, 1])
    tn.apply_gate("H", [2])
    tn.apply_gate("CNOT", [2, 3])
    print("Applied gate sequence")

    # Get state and verify normalization
    state = tn.get_state_vector()
    norm = np.linalg.norm(state)
    print(f"State norm: {norm:.6f}")

    # Profile
    prof = tn.profile()
    print(f"Backend: {prof['backend']}")
    print(f"Device: {prof['device']}")
    print(f"Execution time: {prof['execution_time_s']:.4f}s")


def example_distributed():
    """Example: Distributed execution setup."""
    print("\n" + "=" * 60)
    print("Example 5: Distributed Context Initialization")
    print("=" * 60)

    # Initialize cluster (will use fallback if JAX/PyTorch not available)
    ctx = init_cluster(backend="jax", mesh_shape=(1, 1), seed=12345)
    print("Initialized cluster:")
    print(f"  Backend: {ctx.backend}")
    print(f"  Rank: {ctx.global_rank}/{ctx.world_size}")
    print(f"  Device: {ctx.device}")
    print(f"  Seed: {ctx.seed}")

    # Create and shard state
    num_qubits = 8
    state = initialize_zero_state(num_qubits=num_qubits)
    sharded = shard_state(ctx, state)
    print(f"\nSharded {num_qubits}-qubit state:")
    print(f"  Global shape: {sharded.global_shape}")
    print(f"  Shard spec: {sharded.shard_spec}")

    # Profile
    prof = profile(ctx)
    print("\nProfile:")
    print(f"  World size: {prof['world_size']}")
    print(f"  Wall time: {prof['wall_time_s']:.4f}s")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("QuASIM Distributed Quantum Simulation Examples")
    print("=" * 60)

    try:
        example_bell_state()
        example_ghz_state()
        example_noise_simulation()
        example_tensor_network()
        example_distributed()
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("All examples completed successfully! ✓")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
