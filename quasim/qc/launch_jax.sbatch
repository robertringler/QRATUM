#!/bin/bash
#SBATCH --job-name=quasim_jax
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --output=quasim_jax_%j.out
#SBATCH --error=quasim_jax_%j.err

# QuASIM JAX Distributed Execution
# Target: 8 GPUs on single node with NVLink/NVSwitch

# Load modules
module load cuda/12.x
module load nccl/2.x

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/path/to/cuda"
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

# JAX configuration
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1

# Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/sybernix"

# Run example
echo "Starting QuASIM JAX distributed simulation"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: 8"

python3 << EOF
from quasim.qc.quasim_dist import init_cluster, shard_state, profile, initialize_zero_state
from quasim.qc.quasim_multi import MultiQubitSimulator

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

# Apply gates (simplified example)
print("Applying gates...")
sim = MultiQubitSimulator(num_qubits=num_qubits, seed=12345)
sim.initialize_state()
sim.apply_gate("H", [0])
sim.apply_gate("CNOT", [0, 1])

# Profile execution
print("Profiling...")
prof = profile(ctx)
print(f"Profile: {prof}")

print("Simulation complete!")
EOF

echo "Job completed at $(date)"
