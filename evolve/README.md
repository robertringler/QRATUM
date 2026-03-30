# Phase III Kernel Evolution

This directory contains the autonomous kernel evolution system for QuASIM.

## Components

- `introspection.py`: Runtime introspection agent for collecting kernel metrics
- `rl_controller.py`: Reinforcement learning controller for kernel optimization
- `init_population.py`: Initial population generator
- `policies/`: Serialized RL policies and kernel genomes
- `logs/`: Introspection logs and telemetry data

## Usage

Initialize population:

```bash
python -m evolve.init_population --population-size 20 --seed 42
```

## Evolution Strategy

The system uses an evolutionary algorithm with:

- Population-based optimization
- Elite selection (top 50%)
- Mutation-based exploration
- Fitness evaluation based on latency and energy

## Kernel Genome

Each kernel genome encodes:

- `tile_size`: Computation tile size (64-1024)
- `warp_count`: Number of warps (8-64)
- `unroll_factor`: Loop unroll factor (1-16)
- `async_depth`: Async pipeline depth (1-8)
- `precision`: Numerical precision (fp8, fp16, bf16, fp32)
