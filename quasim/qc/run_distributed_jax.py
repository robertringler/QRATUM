#!/usr/bin/env python3
"""Distributed JAX execution script for QuASIM.

This script is called by launch_jax.sbatch.
It initializes the distributed context and runs a quantum simulation.
"""

from quasim.qc.quasim_dist import init_cluster, initialize_zero_state, profile, shard_state
from quasim.qc.quasim_multi import MultiQubitSimulator


def main():
    """Main distributed execution."""

    # Initialize distributed context
    print("Initializing JAX cluster with 2x4 mesh...")
    ctx = init_cluster(backend="jax", mesh_shape=(2, 4), seed=12345)
    print(f"Context: rank {ctx.global_rank}/{ctx.world_size}, device {ctx.device}")

    # Create initial state
    num_qubits = 28
    print(f"Initializing {num_qubits}-qubit state...")
    psi0 = initialize_zero_state(num_qubits=num_qubits, dtype="complex64")

    # Shard across devices
    print("Sharding state...")
    S = shard_state(ctx, psi0)
    print(f"Global shape: {S.global_shape}, shard spec: {S.shard_spec}")

    # Apply gates (simplified example using MultiQubitSimulator)
    print("Applying gates...")
    sim = MultiQubitSimulator(num_qubits=min(num_qubits, 10), seed=12345)
    sim.initialize_state()
    sim.apply_gate("H", [0])
    sim.apply_gate("CNOT", [0, 1])
    print("Applied H and CNOT gates to first 2 qubits")

    # Profile execution
    print("Profiling...")
    prof = profile(ctx)
    print("Profile:")
    print(f"  Backend: {prof['backend']}")
    print(f"  Device: {prof['device']}")
    print(f"  World size: {prof['world_size']}")
    print(f"  Wall time: {prof['wall_time_s']:.4f}s")

    print("Simulation complete!")


if __name__ == "__main__":
    main()
