#!/bin/bash
#SBATCH --job-name=quasim_torch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --output=quasim_torch_%j.out
#SBATCH --error=quasim_torch_%j.err

# QuASIM PyTorch Distributed Execution
# Target: 4 nodes × 8 GPUs = 32 ranks with InfiniBand interconnect

# Load modules
module load cuda/12.x
module load nccl/2.x
module load openmpi/4.x

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_SOCKET_IFNAME=ib0

# PyTorch configuration
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1

# Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/sybernix"

# Master node info
MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
MASTER_PORT=29500

echo "Starting QuASIM PyTorch distributed simulation"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Total ranks: $((SLURM_NNODES * 8))"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# Launch with torchrun on each node
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    << EOF
import os
import torch
import torch.distributed as dist
from quasim.qc.quasim_dist import init_cluster, shard_state, profile, initialize_zero_state

# Initialize distributed context
print(f"Rank {os.environ.get('RANK', 0)}: Initializing PyTorch cluster...")
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

EOF

echo "Job completed at $(date)"
