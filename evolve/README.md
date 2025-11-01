# Phase III Autonomous Kernel Evolution

This directory contains the core implementation of Phase III autonomous evolution capabilities for QuASIM.

## Components

### Core Evolution System

- **`init_population.py`**: Initial kernel genome generation and population management
- **`introspection_agent.py`**: Runtime performance monitoring and metrics collection
- **`rl_controller.py`**: Reinforcement learning-based kernel optimization
- **`precision_controller.py`**: Hierarchical hybrid precision management
- **`evolution_dashboard.py`**: Progress tracking and visualization
- **`phase3_orchestrator.py`**: Main orchestration and integration

## Quick Start

```bash
# Initialize population
python init_population.py

# Run full evolution cycle
python phase3_orchestrator.py
```

## Genome Structure

Each kernel genome encodes:
- `tile_size`: 8-128 (tile dimension)
- `warp_count`: 1-32 (parallel warps)
- `unroll_factor`: 1-16 (loop unrolling)
- `async_depth`: 1-8 (async queue depth)
- `precision`: fp32, fp16, bf16, fp8, int8

## Evolution Process

1. **Initialize**: Generate diverse population with random parameters
2. **Evaluate**: Run each genome through optimization pipeline
3. **Select**: Keep top 50% performers
4. **Mutate**: Create variants of top performers
5. **Repeat**: Iterate for multiple generations

## RL Optimization

The RL controller:
- Observes current kernel configuration and performance
- Selects parameter adjustments
- Receives reward based on latency/divergence/cache performance
- Updates policy via gradient descent

## Precision Management

Three-zone hierarchy:
- **Outer**: FP32 (2 layers) - stability
- **Inner**: FP8 (6 layers) - performance  
- **Boundary**: BF16 (2 layers) - balance

Automatic fallback to higher precision when error budget exceeded.

## Performance Targets

- Speedup: ≥ 3×
- Energy reduction: ≥ 40%
- Numerical error: < 1e-6

## File Structure

```
evolve/
├── genomes/              # Saved kernel genomes
│   ├── gen0_kernel_000.json
│   └── population_index.json
├── policies/             # RL policy checkpoints
│   ├── policy.json
│   └── optimization_history.json
├── init_population.py    # Population generation
├── introspection_agent.py # Performance monitoring
├── rl_controller.py      # RL optimization
├── precision_controller.py # Precision management
├── evolution_dashboard.py # Progress tracking
└── phase3_orchestrator.py # Main orchestrator
```

## Integration

The orchestrator integrates with:
- `schedules/` - Differentiable scheduling
- `quantum_search/` - Ising optimization
- `memgraph/` - Memory layout optimization
- `profiles/` - Causal profiling
- `federated/` - Intelligence sharing

## Monitoring

Track evolution via dashboard:

```python
from evolve.evolution_dashboard import EvolutionDashboard

dashboard = EvolutionDashboard()
dashboard.load_dashboard()
print(dashboard.generate_report())
```

## Checkpointing

All state is automatically saved:
- Genomes → `genomes/`
- RL policies → `policies/`
- Metrics → `profiles/evolution/`

Resume evolution by loading saved population.

## Advanced Usage

### Custom Fitness Function

```python
from evolve.rl_controller import RLController

controller = RLController()

# Override reward computation
def custom_reward(metrics):
    return 100.0 / metrics.latency_ms - 0.5 * metrics.energy_j

# Use in optimization
genome = controller.optimize_kernel(genome, metrics)
```

### Population Analysis

```python
from evolve.init_population import load_population

population = load_population(Path("genomes"))
avg_tile_size = sum(g.tile_size for g in population) / len(population)
best_genome = max(population, key=lambda g: g.fitness)
```

## See Also

- [Phase III Documentation](../docs/phase3_autonomous_evolution.md)
- [Testing Guide](../tests/software/test_phase3.py)
- [QuASIM Main README](../README.md)
