# BM_001 Tier-0 Execution Guide

## Overview

This guide describes how to execute the BM_001 (Large-Strain Rubber Block Compression) benchmark from the Ansys Tier-0 Embedment Package.

## Benchmark Description

**BM_001: Large-Strain Rubber Block Compression**

- **Physics Domain**: Nonlinear hyperelastic mechanics
- **Material Model**: Mooney-Rivlin (2-parameter)
- **Strain Regime**: Large strain (70% engineering compression)
- **Element Count**: ~5,000 SOLID186 elements
- **Target Speedup**: 3-5x vs Ansys baseline

## Execution

### Prerequisites

```bash
# Python 3.10+ with required packages
pip install numpy pyyaml reportlab

# Verify installation
python3 -c "import numpy, yaml, reportlab; print('Dependencies OK')"
```

### Running the Benchmark

**Option 1: Using the dedicated executor (recommended)**

```bash
python3 evaluation/ansys/bm_001_executor.py \
    --output reports/BM_001 \
    --runs 5 \
    --seed 42 \
    --device gpu \
    --cooldown 60
```

**Option 2: Using the legacy runner**

```bash
python3 run_bm001_tier0.py \
    --output results/bm001_tier0 \
    --runs 5 \
    --seed 42 \
    --device gpu \
    --cooldown 60
```

**Parameters:**
- `--output`: Output directory for results
- `--runs`: Number of independent runs per solver (default: 5)
- `--seed`: Random seed for deterministic execution (default: 42)
- `--device`: Compute device for QuASIM (cpu/gpu, default: gpu)
- `--cooldown`: Cooldown period between runs in seconds (default: 60)

### Execution Workflow

The script executes the following steps:

1. **Load Benchmark Definition**: Reads BM_001 spec from `benchmark_definitions.yaml`
2. **Ansys Baseline Solves**: Execute 5 independent Ansys CPU baseline runs
3. **QuASIM Solves**: Execute 5 independent QuASIM GPU-accelerated runs
4. **Statistical Analysis**: Compute median, confidence intervals, outlier detection
5. **Report Generation**: Generate CSV, JSON, HTML, and PDF reports
6. **Reproducibility Verification**: Verify deterministic hashing across runs

## Results

### Output Structure

```
results/bm001_tier0/
├── ansys/
│   └── BM_001/
│       ├── run_1.json
│       ├── run_2.json
│       ├── run_3.json
│       ├── run_4.json
│       └── run_5.json
├── quasim/
│   └── BM_001/
│       ├── run_1.json
│       ├── run_2.json
│       ├── run_3.json
│       ├── run_4.json
│       └── run_5.json
└── reports/
    ├── results.csv      # Machine-readable summary
    ├── results.json     # Detailed metrics with full run data
    ├── report.html      # Human-readable web report
    └── report.pdf       # Printable PDF report
```

### Performance Metrics

Expected results (with mock solver):

| Metric | Value |
|--------|-------|
| **Ansys Median Time** | ~180s |
| **QuASIM Median Time** | ~45s |
| **Speedup** | 4.0x |
| **Displacement Error** | < 2% |
| **Stress Error** | < 5% |
| **Reproducibility** | ✓ Verified |

### Acceptance Criteria

BM_001 passes Tier-0 validation if:

- ✓ Displacement error < 2% (vs Ansys reference)
- ✓ Stress error < 5% (vs Ansys reference)
- ✓ Speedup ≥ 3.0x (vs Ansys CPU baseline)
- ✓ Zero convergence failures
- ✓ Deterministic reproducibility (identical hashes with same seed)

## Deterministic Reproducibility

QuASIM guarantees bit-exact reproducibility with the following constraints:

1. **Fixed Random Seed**: Same seed produces identical results
2. **Deterministic Operations**: No race conditions or non-deterministic algorithms
3. **Hash Verification**: SHA-256 hash of displacement field verifies reproducibility

Example verification:

```python
# All 5 QuASIM runs should produce identical hash:
43d602ecb9602d78ea187e33426a9df2c27639f429c7767b4289f25f90de68f1
```

## Production Deployment

### Replacing Mock Solver

The current implementation uses a mock solver for demonstration. To integrate real C++/CUDA backend:

1. **Mesh Generation**: Replace mock mesh with actual Ansys CDB import
2. **Solver Integration**: Call `quasim_ansys_adapter.solve()` with real GPU kernels
3. **Result Extraction**: Extract actual displacement/stress fields from GPU memory
4. **Hash Computation**: Compute SHA-256 of actual nodal displacements

### Expected Timeline

- **Mock Solver** (current): ~30s total execution time
- **Production Solver**: ~10 minutes (5 runs × 180s Ansys + 5 runs × 45s QuASIM)
- **Cooldown**: 5 minutes (60s × 9 cooldown periods)
- **Total**: ~15 minutes for full BM_001 Tier-0 validation

## Troubleshooting

### GPU Not Available

If GPU is not available, QuASIM automatically falls back to CPU:

```
WARNING - GPU requested but not available, falling back to CPU
```

CPU fallback still provides competitive performance vs Ansys baseline.

### Convergence Failures

If solver fails to converge:

1. Check mesh quality (negative Jacobian?)
2. Reduce load step size (increase substeps)
3. Enable line search
4. Review material parameters

### Non-Deterministic Behavior

If reproducibility verification fails:

1. Verify random seed is set correctly
2. Check for race conditions in parallel code
3. Ensure fixed iteration order (sorted element IDs)
4. Disable any non-deterministic optimizations

## References

- **Benchmark Spec**: `benchmarks/ansys/benchmark_definitions.yaml`
- **Adapter Implementation**: `sdk/ansys/quasim_ansys_adapter.py`
- **Performance Runner**: `evaluation/ansys/performance_runner.py`
- **Integration Spec**: `docs/ansys/ANSYS_INTEGRATION_SPEC.md`

## Support

For issues or questions:
- GitHub Issues: https://github.com/robertringler/Qubic/issues
- Email: support@quasim.io
