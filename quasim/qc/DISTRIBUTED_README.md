# QuASIM Distributed Quantum Simulation

## Overview

This module provides scalable, fault-tolerant distributed quantum simulation across multi-GPU and multi-node clusters. It supports:

- **Multi-qubit simulation** (2-32 qubits) with entangled states (Bell, GHZ, W)
- **Tensor-network backends** with JAX/PyTorch GPU acceleration
- **Distributed execution** across NVLink/NVSwitch clusters
- **Deterministic, reproducible** results for certification and auditing
- **Fault tolerance** with checkpoint/restore
- **Performance profiling** and telemetry

## Architecture

### Components

1. **`quasim_multi.py`** - Multi-qubit simulator with entanglement
   - State vector representation for N qubits (dimension 2^N)
   - Single/multi-qubit gates (H, X, Y, Z, CNOT, CZ, SWAP, rotations)
   - Entangled state constructors (Bell, GHZ, W)
   - Noise channels (amplitude/phase damping, depolarizing)
   - Quantum tomography and entanglement entropy
   - Time-dependent Hamiltonian evolution

2. **`quasim_tn.py`** - Tensor-network engine with GPU acceleration
   - Full tensor representation for N≤12 qubits
   - Matrix Product State (MPS) compression for N>12 with bond dimension
   - JAX/PyTorch/NumPy backend support
   - Batched trajectory execution for Monte Carlo
   - JIT compilation and memory optimization
   - Performance profiling

3. **`quasim_dist.py`** - Distributed cluster kernel
   - Multi-GPU parallelism via JAX pjit/pmap or PyTorch DDP/FSDP
   - Multi-node execution with MPI/NCCL
   - State sharding and distributed gate application
   - Checkpoint save/restore for fault tolerance
   - Performance metrics and telemetry

## Installation

### Requirements

```bash
# Core dependencies
pip install numpy scipy

# Optional: JAX with GPU support
pip install "jax[cuda12]"  # or cuda11

# Optional: PyTorch with GPU support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For distributed execution
pip install mpi4py
```

### Cluster Setup

#### NVLink/NVSwitch Configuration

For optimal performance on NVIDIA GPU clusters:

```bash
# Enable NVLink
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NCCL configuration
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0           # Enable InfiniBand
export NCCL_NET_GDR_LEVEL=5        # Enable GPUDirect RDMA
export NCCL_SOCKET_IFNAME=ib0      # InfiniBand interface
```

#### InfiniBand Tuning

For multi-node clusters with InfiniBand:

```bash
# IPoIB configuration
sudo ip link set dev ib0 mtu 65520

# UCX configuration (if using)
export UCX_NET_DEVICES=mlx5_0:1
export UCX_TLS=rc,cuda_copy,cuda_ipc
export UCX_MEMTYPE_CACHE=n
```

## Usage

### Basic Multi-Qubit Simulation

```python
from quasim.qc.quasim_multi import MultiQubitSimulator

# Initialize simulator
sim = MultiQubitSimulator(num_qubits=3, seed=42)
sim.initialize_state()

# Create GHZ state: (|000⟩ + |111⟩)/√2
sim.create_ghz_state()

# Apply noise
noise = {
    "gamma1": 0.002,      # Amplitude damping
    "gamma_phi": 0.003,   # Phase damping
    "p_depol": 0.001      # Depolarizing
}
sim.apply_noise(noise)

# Compute entanglement
entropy = sim.entanglement_entropy(subsystem=[0])
print(f"Entanglement entropy: {entropy:.3f} bits")

# Tomography
results = sim.tomography()
print(f"Purity: {results['purity']:.4f}")
```

### Tensor-Network Simulation

```python
from quasim.qc.quasim_tn import TensorNetworkEngine

# Initialize with JAX backend
tn = TensorNetworkEngine(
    num_qubits=12,
    bond_dim=64,
    backend="jax",
    seed=42
)

tn.initialize_state()

# Apply gates
tn.apply_gate("H", [0])
tn.apply_gate("CNOT", [0, 1])

# Batch execution
results = tn.batch_run(trajectories=1024)

# Profile performance
profile = tn.profile()
print(f"Backend: {profile['backend']}")
print(f"Device: {profile['device']}")
print(f"Execution time: {profile['execution_time_s']:.2f}s")
```

### Distributed Execution

#### JAX (Single Node, 8 GPUs)

```python
from quasim.qc.quasim_dist import (
    init_cluster, shard_state, profile, initialize_zero_state
)

# Initialize cluster with 2×4 mesh (data × model parallelism)
ctx = init_cluster(backend="jax", mesh_shape=(2, 4), seed=12345)
print(f"Initialized rank {ctx.global_rank}/{ctx.world_size}")

# Create 28-qubit state
psi0 = initialize_zero_state(num_qubits=28, dtype="complex64")

# Shard across devices
S = shard_state(ctx, psi0)
print(f"State sharded: {S.global_shape}")

# Profile
prof = profile(ctx)
print(f"Device: {prof['device']}")
print(f"HBM usage: {prof['hbm_usage_mb']:.1f} MB")
```

#### PyTorch (Multi-Node via torchrun)

```bash
# 4 nodes × 8 GPUs = 32 ranks
torchrun \
    --nnodes=4 \
    --nproc_per_node=8 \
    --rdzv_id=12345 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=node0:29500 \
    my_simulation.py
```

```python
# my_simulation.py
from quasim.qc.quasim_dist import init_cluster, shard_state

ctx = init_cluster(backend="torch", mesh_shape=(32, 1), seed=12345)

if ctx.global_rank == 0:
    print(f"Running on {ctx.world_size} GPUs")

# Your simulation code...
```

### Checkpointing

```python
from quasim.qc.quasim_dist import save_checkpoint, load_checkpoint

# Save checkpoint
save_checkpoint(ctx, sharded_state, path="checkpoints/step_1000")

# Later, restore
restored_state = load_checkpoint(ctx, path="checkpoints/step_1000")
```

## SLURM Job Scripts

### JAX Launch Script

```bash
sbatch quasim/qc/launch_jax.sbatch
```

See `launch_jax.sbatch` for single-node 8-GPU configuration.

### PyTorch Launch Script

```bash
sbatch quasim/qc/launch_torch.sbatch
```

See `launch_torch.sbatch` for multi-node 32-GPU configuration.

## Testing

Run the test suite:

```bash
# All tests
pytest quasim/qc/test_quasim_dist.py -v

# Specific test class
pytest quasim/qc/test_quasim_dist.py::TestMultiQubitSimulator -v

# With coverage
pytest quasim/qc/test_quasim_dist.py --cov=quasim.qc --cov-report=html
```

### Test Categories

1. **Correctness** - Verify fidelity vs exact states
2. **Determinism** - Identical seeds yield identical results
3. **Scalability** - GHZ/Bell states scale to 8+ qubits
4. **Entanglement** - Entropy measures are correct
5. **Checkpointing** - Save/restore preserves state

## Performance Targets

### Single-Node (8× H100 with NVSwitch)

- **32-qubit MPS** (bond_dim=128), 512 trajectories
- Target: ≥0.8× ideal strong-scaling
- Communication/computation overlap: ≥60%

### Multi-Node (4 nodes × 8 GPUs)

- **30-qubit state vector** distributed across 32 GPUs
- Checkpoint overhead: ≤5% at 10-min cadence
- Deterministic across runs with same seed

## Troubleshooting

### NCCL Issues

If you see NCCL timeouts:

```bash
export NCCL_TIMEOUT=3600           # Increase timeout
export NCCL_DEBUG=INFO             # Enable logging
export NCCL_P2P_DISABLE=1          # Disable P2P if needed
```

### Memory Issues

For large simulations:

```python
# Use MPS compression
tn = TensorNetworkEngine(num_qubits=20, bond_dim=32)  # Lower bond_dim

# Or reduce batch size
results = tn.batch_run(trajectories=256)  # Fewer trajectories
```

### JAX Device Issues

```python
import jax
print(jax.devices())  # Check available devices

# Force CPU if GPU unavailable
import os
os.environ["JAX_PLATFORMS"] = "cpu"
```

## Compliance and Certification

This implementation follows aerospace-grade determinism principles:

- **DO-178C** style: Traceable, testable, reproducible
- **ECSS-Q-ST-80C** process assurance
- Deterministic RNG with recorded seeds
- Bitwise-stable reductions for reproducibility
- Immutable audit trails in checkpoints

## References

1. Nielsen & Chuang, "Quantum Computation and Quantum Information"
2. Preskill, "Quantum Computation" lecture notes (Caltech)
3. JAX documentation: <https://jax.readthedocs.io/>
4. PyTorch distributed: <https://pytorch.org/tutorials/beginner/dist_overview.html>
5. NCCL tuning: <https://docs.nvidia.com/deeplearning/nccl/user-guide/>

## License

Part of the Sybernix/QuASIM project. See main repository for license details.
