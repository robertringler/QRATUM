#!/usr/bin/env python3
"""Distributed PyTorch execution script for QuASIM.

This script is called by launch_torch.sbatch via torchrun.
It initializes the distributed context and runs a quantum simulation.
"""

import os

import torch.distributed as dist

from quasim.qc.quasim_dist import (init_cluster, initialize_zero_state,
                                   profile, shard_state)


def main():
    """Main distributed execution."""

    # Initialize distributed context
    rank = int(os.environ.get("RANK", 0))
    print(f"Rank {rank}: Initializing PyTorch cluster...")

    ctx = init_cluster(backend="torch", mesh_shape=(32, 1), seed=12345)

    if ctx.global_rank == 0:
        print(f"Cluster initialized: {ctx.world_size} ranks")
        print(f"Mesh shape: {ctx.mesh_shape}")
        print(f"Device: {ctx.device}")

    # Create and shard state
    num_qubits = 30
    if ctx.global_rank == 0:
        print(f"Creating {num_qubits}-qubit state...")

    psi0 = initialize_zero_state(num_qubits=num_qubits, dtype="complex64")
    S = shard_state(ctx, psi0)

    if ctx.global_rank == 0:
        print(f"State sharded: global shape {S.global_shape}")

    # Synchronize all ranks
    if dist.is_initialized():
        dist.barrier()

    # Profile
    if ctx.global_rank == 0:
        prof = profile(ctx)
        print(f"Profile: {prof}")
        print("Simulation complete!")


if __name__ == "__main__":
    main()
