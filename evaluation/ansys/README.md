# BM_001 Production Benchmark Executor

## Overview

The BM_001 executor implements a production-ready benchmark framework for comparing Ansys Mechanical baseline performance against QuASIM GPU-accelerated solver for large-strain rubber block compression simulations.

## Features

- **Production Executors**: QuasimCudaSolver (GPU/CPU) and PyMapdlExecutor (Ansys)
- **Deterministic Execution**: SHA-256 hash verification for reproducibility
- **Multi-Format Reports**: CSV, JSON, HTML, and PDF
- **Statistical Validation**: Bootstrap confidence intervals, outlier detection, acceptance criteria
- **Hardware Monitoring**: GPU/CPU utilization metrics
- **CI/CD Integration**: Automated workflow with security scanning

## Execution Commands

### Default Run
Run 5 Ansys + 5 QuASIM executions with default settings:

```bash
python3 evaluation/ansys/bm_001_executor.py --output reports/BM_001
```

### Custom Run
Run with custom parameters:

```bash
python3 evaluation/ansys/bm_001_executor.py \
    --runs 10 \
    --cooldown 120 \
    --device gpu \
    --seed 42 \
    --output reports/BM_001/custom
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--runs` | 5 | Number of runs per solver |
| `--cooldown` | 60 | Cooldown period between runs (seconds) |
| `--device` | `gpu` | Compute device (`cpu` or `gpu`) |
| `--seed` | 42 | Random seed for deterministic execution |
| `--output` | `reports/BM_001` | Output directory for reports |
| `--yaml` | `benchmarks/ansys/benchmark_definitions.yaml` | Benchmark definitions file |

## Report Formats

After execution, the following reports are generated in the output directory:

### 1. CSV Summary (`summary.csv`)
Quick summary table with key metrics:
- Benchmark ID, Status, Speedup, Confidence Intervals
- Displacement, Stress, and Energy errors
- Coefficient of Variation

### 2. JSON Metadata (`results.json`)
Complete metadata including:
- All execution results (Ansys and QuASIM)
- Statistical metrics
- Reproducibility information (SHA-256 hashes)
- Hardware metrics
- Timestamp information

### 3. HTML Report (`report.html`)
Styled web report with:
- Performance and accuracy metrics
- Execution results tables
- Reproducibility status
- Visual formatting for easy review

### 4. PDF Executive Summary (`executive_summary.pdf`)
Professional executive summary with:
- Key results and metrics
- Performance comparison table
- Compliance with acceptance criteria

## Acceptance Criteria

The benchmark must meet the following criteria to pass:

### Performance
- **Speedup**: ≥ 3.0x vs Ansys baseline
- **Coefficient of Variation**: < 2% (reproducibility)

### Accuracy
- **Displacement Error**: < 2%
- **Stress Error**: < 5%
- **Energy Conservation Error**: < 1e-6

### Reproducibility
- All QuASIM runs with the same seed must produce identical SHA-256 hashes

## Statistical Validation

The executor performs rigorous statistical analysis:

### Bootstrap Confidence Intervals
- 1000 bootstrap samples per metric
- 95% confidence intervals for speedup

### Outlier Detection
- Modified Z-score method
- Threshold: |Z| > 3.5

### Acceptance Testing
- Automated verification of all acceptance criteria
- Detailed failure messages for debugging

## Hardware Metrics

The following hardware utilization metrics are collected:

- Device type (CPU/GPU)
- GPU memory (allocated, reserved, peak)
- Execution duration
- CPU core count (for Ansys baseline)

## Deterministic Execution

All executions are fully deterministic and reproducible:

1. **Random Seed Control**: Fixed seed (default: 42) for RNG
2. **SHA-256 Hashing**: State hash computed from displacement fields
3. **Verification**: Automatic check that all runs produce identical hashes

## CI/CD Integration

The `.github/workflows/bm_001.yml` workflow automates:

1. **Environment Setup**: Python 3.12, NumPy 2.3.5, dependencies
2. **Linting**: ruff code quality checks
3. **Execution**: 5 Ansys + 5 QuASIM runs
4. **Validation**: Reproducibility and acceptance criteria checks
5. **Security**: CodeQL security scanning
6. **Artifacts**: Reports and hash logs (30-90 day retention)

### Workflow Triggers

- Push to main/develop branches
- Pull requests to main/develop
- Manual workflow dispatch with custom parameters

### Workflow Outputs

- Execution reports (CSV, JSON, HTML, PDF)
- Hash logs for audit trail
- GitHub Actions summary with key metrics

## Troubleshooting

### Common Issues

#### 1. GPU Not Available
**Symptom**: Warning "GPU requested but not available, falling back to CPU"

**Solution**: 
- Ensure CUDA drivers are installed
- Install PyTorch: `pip install torch`
- For CI: Use CPU mode explicitly with `--device cpu`

#### 2. PyYAML Import Error
**Symptom**: "PyYAML required. Install with: pip install pyyaml"

**Solution**: `pip install pyyaml>=6.0`

#### 3. Benchmark Definition Not Found
**Symptom**: "BM_001 not found in YAML"

**Solution**: Verify `benchmarks/ansys/benchmark_definitions.yaml` exists and contains BM_001 definition

#### 4. Acceptance Criteria Failure
**Symptom**: "Acceptance check failed: Speedup X.XXx < 3.0x"

**Solution**: This is expected with stub solvers. Production C++/CUDA backend will achieve >3x speedup on real hardware.

#### 5. Non-Deterministic Behavior
**Symptom**: "Non-deterministic: N unique hashes detected"

**Solution**: 
- Verify same `--seed` value across all runs
- Check for external randomness sources
- Ensure consistent NumPy version

## Integration with Production Backend

### Current Status (Stub Mode)
The current implementation uses production-ready stubs:
- `QuasimCudaSolver`: Simulates GPU-accelerated tensor solver
- `PyMapdlExecutor`: Simulates Ansys MAPDL execution

### Production Integration (TODO)
To integrate with actual C++/CUDA backend:

1. **QuASIM Solver**:
   ```python
   # In QuasimCudaSolver.solve()
   # Replace stub with actual cuQuantum tensor network call
   from quasim.backends.cuda import CUDATensorSolver
   solver = CUDATensorSolver(device=self.device, seed=self.random_seed)
   result = solver.solve(mesh_data, material_params, boundary_conditions)
   ```

2. **PyMAPDL Executor**:
   ```python
   # In PyMapdlExecutor.execute()
   from ansys.mapdl.core import launch_mapdl
   self._mapdl_session = launch_mapdl(nproc=4, override=True)
   # Execute Ansys commands via PyMAPDL API
   ```

## Development Notes

### Adding New Benchmarks
To extend the framework to BM_002-BM_005:

1. Copy `bm_001_executor.py` to `bm_00X_executor.py`
2. Update benchmark ID in the code
3. Modify mesh/material parameters as needed
4. Update CI/CD workflow to include new benchmark
5. Add acceptance criteria to YAML

### Code Quality Requirements
- **Linting**: Pass `ruff check` with no errors
- **Formatting**: Use `ruff format` (PEP 8, line length 100)
- **Type Hints**: Required for all public functions
- **Security**: Pass CodeQL scan with zero alerts

## References

- Benchmark Definitions: `benchmarks/ansys/benchmark_definitions.yaml`
- SDK Adapter: `sdk/ansys/quasim_ansys_adapter.py`
- Performance Runner: `evaluation/ansys/performance_runner.py`
- CI/CD Workflow: `.github/workflows/bm_001.yml`

## Contact

For questions or issues:
- Open GitHub issue with label `benchmark` or `ansys-integration`
- See `CONTRIBUTING.md` for contribution guidelines
- Check `SECURITY.md` for security vulnerability reporting
