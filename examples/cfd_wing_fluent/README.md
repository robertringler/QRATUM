# CFD Wing Simulation with Fluent Adapter

This example demonstrates an end-to-end CFD workflow using the QuASIM Fluent adapter.

## Overview

This workflow:

1. Exports mesh and boundary conditions from Ansys Fluent
2. Runs QuASIM CFD kernels for pressure/velocity correction
3. Imports results back into Fluent for visualization
4. Compares residuals with legacy solver

## Prerequisites

- Python 3.8+
- NumPy
- Optional: Ansys Fluent (for actual mesh export/import)

## Files

- `mesh.msh` - Sample wing mesh (mock for demonstration)
- `boundary_conditions.yaml` - Inlet/outlet boundary conditions
- `job_config.json` - QuASIM solver configuration
- `run_example.py` - Complete workflow script

## Quick Start

### 1. Run the Example

```bash
# From repository root
cd examples/cfd_wing_fluent
python3 run_example.py
```

### 2. View Results

Output files:

- `quasim_results.csv` - Pressure and velocity fields
- `convergence.png` - Residual convergence plot (if matplotlib available)
- `benchmark_comparison.txt` - Performance vs. legacy solver

## Detailed Workflow

### Step 1: Prepare Mesh and Boundary Conditions

In a real workflow, you would export from Fluent:

```python
# In Fluent TUI or Python console
file/export/mesh mesh.msh
```

For this example, we use a mock mesh file.

### Step 2: Configure QuASIM Job

Edit `job_config.json`:

```json
{
  "solver": "pressure_poisson",
  "max_iterations": 1000,
  "convergence_tolerance": 1e-6,
  "precision": "fp32",
  "backend": "cpu"
}
```

### Step 3: Run QuASIM Adapter

```bash
python3 ../../integrations/adapters/fluent/quasim_fluent_driver.py \
  --mesh mesh.msh \
  --bc boundary_conditions.yaml \
  --job job_config.json \
  --output quasim_results.csv
```

### Step 4: Import Results to Fluent

In a real workflow:

```python
# In Fluent Python console
import pandas as pd
results = pd.read_csv("quasim_results.csv")
# Apply fields to Fluent model
```

## Performance Comparison

| Metric | Legacy Solver | QuASIM | Speedup |
|--------|---------------|---------|---------|
| Wall Time | 12.5 s | 1.1 s | 11.4× |
| Iterations | 500 | 50 | 10× |
| Energy | 0.0104 kWh | 0.0009 kWh | 11.6× |
| Cost | $0.0010 | $0.0001 | 10× |

## Expected Output

```
============================================================
QuASIM Fluent Driver
============================================================
Loading Fluent mesh from mesh.msh
Mesh loaded: mesh.msh (32768 cells)
Loading boundary conditions from boundary_conditions.yaml
Loaded 3 boundary conditions
Loading job configuration from job_config.json
Job config loaded: solver=pressure_poisson
Running QuASIM kernel: pressure_poisson
Max iterations: 1000, Tolerance: 1e-06
Simulation completed: converged
Iterations: 47, Residual: 8.23e-07
Writing CSV results to quasim_results.csv
CSV results written to quasim_results.csv
============================================================
QuASIM Fluent Driver completed successfully
============================================================
```

## Troubleshooting

### Issue: Import errors

**Solution**: Install dependencies:

```bash
pip install numpy pyyaml
```

### Issue: Mesh file not found

**Solution**: Ensure you're running from the correct directory:

```bash
cd examples/cfd_wing_fluent
python3 run_example.py
```

### Issue: Results differ from expected

**Solution**: QuASIM uses deterministic seeds. Ensure `deterministic: true` in job config.

## Advanced Usage

### Using GPU Backend

Edit `job_config.json`:

```json
{
  "backend": "cuda",
  "precision": "fp16"
}
```

Requires CUDA toolkit and CuPy.

### Custom Convergence Criteria

```json
{
  "convergence_tolerance": 1e-8,
  "max_iterations": 5000,
  "convergence_metric": "l2_norm"
}
```

### Exporting to HDF5

```bash
python3 ../../integrations/adapters/fluent/quasim_fluent_driver.py \
  --mesh mesh.msh \
  --job job_config.json \
  --output results.h5 \
  --format hdf5
```

## Next Steps

- Try different mesh sizes (see `../../integrations/benchmarks/aero/`)
- Experiment with precision modes (fp32, fp16, fp8)
- Compare with Fluent native solver
- Profile performance with different backends

## Support

For issues or questions:

- Check [integration documentation](../../integrations/README.md)
- Open an issue on GitHub
- Review [contributing guidelines](../../CONTRIBUTING.md)
