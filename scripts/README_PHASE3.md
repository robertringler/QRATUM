# Phase III Scripts

This directory contains scripts for running and demonstrating Phase III autonomous kernel evolution.

## Scripts

### `demo_phase3.py`
Comprehensive demonstration of all Phase III components individually.

**Usage:**
```bash
python scripts/demo_phase3.py
```

**Features:**
- Shows each Phase III component in isolation
- Generates sample data for each module
- No arguments required - runs all demos
- ~30 seconds to complete

**Output:**
- Demonstrates 9 core Phase III capabilities
- Generates artifacts in all Phase III directories
- Provides detailed output for each component

### `run_phase3_cycle.py`
Complete integrated evolution cycle combining all Phase III components.

**Usage:**
```bash
# Run with defaults (10 generations, 20 population)
python scripts/run_phase3_cycle.py

# Custom parameters
python scripts/run_phase3_cycle.py --generations 20 --population 30 --seed 42
```

**Arguments:**
- `--generations N`: Number of evolution generations (default: 10)
- `--population N`: Population size (default: 20)
- `--deployment ID`: Deployment ID for federated learning (default: "demo_deployment")
- `--seed N`: Random seed for reproducibility (default: 42)

**Features:**
- Integrates all 10 Phase III objectives
- Simulates realistic kernel execution
- Tracks fitness improvement over generations
- Monitors thermal conditions and applies throttling
- Manages precision with automatic fallback
- Aggregates federated telemetry
- Generates complete artifact set

**Output:**
- Evolution summary with fitness progression
- Energy efficiency metrics
- Thermal statistics
- All Phase III artifacts saved to disk

### Quick Comparison

| Aspect | demo_phase3.py | run_phase3_cycle.py |
|--------|---------------|---------------------|
| Purpose | Education/Demo | Production-like |
| Duration | ~30 seconds | ~1-2 minutes |
| Components | Isolated | Integrated |
| Artifacts | Sample data | Full evolution run |
| Use Case | Learning Phase III | Testing complete workflow |

## Examples

### Quick Demo
See all Phase III features:
```bash
python scripts/demo_phase3.py
```

### Short Evolution Run
Fast evolution cycle for testing:
```bash
python scripts/run_phase3_cycle.py --generations 5 --population 10
```

### Full Evolution Run
Production-scale evolution:
```bash
python scripts/run_phase3_cycle.py --generations 50 --population 50 --seed 123
```

## Output Artifacts

Both scripts generate artifacts in Phase III directories:

```
evolve/
├── logs/              # Introspection metrics (JSON)
├── policies/          # RL policies (JSON)
├── precision_maps/    # Precision configurations (JSON)
└── energy_dashboard.json

schedules/
└── *.json            # Optimized schedules

quantum_search/
└── history.json      # Optimization trajectory

memgraph/
└── *.json           # Memory layouts

profiles/causal/
└── influence_map.json

federated/aggregated/
└── aggregated_performance.json

certs/
└── *_certificate.json
```

## Dependencies

Both scripts require:
- Python 3.11+
- Standard library only (no external packages beyond pytest for testing)

All dependencies are included in the base QuASIM installation.

## Testing

Run Phase III tests:
```bash
PYTHONPATH=.:runtime/python:quantum python -m pytest tests/software/test_phase3.py -v
```

All 11 Phase III tests should pass.

## See Also

- `PHASE3_OVERVIEW.md` - Complete Phase III documentation
- `tests/software/test_phase3.py` - Unit tests for all modules
- Individual module documentation in each Phase III directory
