# QuASIM Benchmark Suite

Comprehensive benchmarking infrastructure for the QuASIM platform, providing automated performance measurement, analysis, and reporting across all computational kernels.

## Overview

The QuASIM benchmark suite provides:

- **Automatic Kernel Discovery**: Scans repository for computational kernels
- **Multi-Backend Support**: CUDA, ROCm, CPU, JAX, PyTorch
- **Multi-Precision Testing**: FP32, FP16, FP8, FP4 (where supported)
- **Statistical Analysis**: Latency percentiles (p50, p90, p99), throughput, memory
- **Regression Detection**: Compare against baseline to identify performance regressions
- **Rich Reporting**: Machine-readable JSON + human-readable Markdown
- **CI/CD Integration**: GitHub Actions workflow for automated benchmarking

## Quick Start

### Local Execution

Run the full benchmark suite:

```bash
python3 tools/bench_all.py --iters 30 --warmup 3 --precision fp32 --backends auto
```

Or use the Makefile target:

```bash
make bench
```

### Common Use Cases

**Quick test with minimal iterations:**
```bash
python3 tools/bench_all.py --iters 5 --warmup 1
```

**Multi-precision benchmarking:**
```bash
python3 tools/bench_all.py --precision fp32,fp16,fp8
```

**Specific backend:**
```bash
python3 tools/bench_all.py --backends cuda
```

**Custom output directory:**
```bash
python3 tools/bench_all.py --output-dir my_results
```

**Compare against a baseline:**
```bash
python3 tools/bench_all.py --compare-to main
```

## Command-Line Options

```
--iters N              Number of timed iterations per kernel (default: 30)
--warmup N             Number of warmup iterations (default: 3)
--precision LIST       Comma-separated precisions: fp32,fp16,fp8,fp4 (default: fp32)
--backends LIST        Comma-separated backends: cuda,rocm,cpu,jax,auto (default: auto)
--output-dir PATH      Output directory for reports (default: reports)
--seed N               Random seed for reproducibility (default: 1337)
--compare-to BRANCH    Git branch to compare against for regression detection
```

## Output Structure

The benchmark suite generates the following files:

```
reports/
├── env.json                    # System environment information
├── kernel_manifest.json        # Discovered kernels catalog
├── summary.json                # Aggregated results (machine-readable)
├── summary.md                  # Human-readable summary with tables
├── regressions.md              # Regression analysis vs. baseline
└── kernels/                    # Per-kernel detailed results
    ├── kernel1_backend_precision.json
    ├── kernel2_backend_precision.json
    └── ...
```

## Report Contents

### Environment (`env.json`)

- System information (OS, architecture, hostname)
- Python version
- Git commit, branch, and dirty status
- GPU information (if available)
- CUDA/ROCm versions
- Library versions (numpy, torch, jax, cupy)

### Kernel Manifest (`kernel_manifest.json`)

- List of all discovered kernels
- Path, backend, module path
- Dependencies
- Test commands

### Per-Kernel Results (`kernels/*.json`)

Each kernel result includes:

- **Timing metrics**: p50, p90, p99, mean, std, min, max latency
- **Throughput**: Operations per second
- **Memory**: Peak, allocated, reserved (MB)
- **Energy**: Consumption in Joules (if GPU metrics available)
- **Accuracy**: RMSE, MAE, ULPs, numerical error
- **Configuration**: Input parameters
- **Status**: Success/failure, error messages

### Summary (`summary.md`)

Human-readable report with:

- **Environment**: Platform, Python, Git, GPU information
- **Overall Leaderboard**: Ranked performance table
- **Backend Comparison**: Performance by backend
- **Key Findings**: Top performers, throughput leaders, fastest backend
- **Recommendations**: Optimization suggestions

### Regressions (`regressions.md`)

- Comparison against baseline (if available)
- Latency regression alerts (>10% slower)
- Accuracy drift detection (>1e-3 RMSE change)
- Detailed regression analysis

## Kernel Configuration

Per-kernel configuration files can be placed at `kernels/<name>/bench.yaml`:

```yaml
# Example: integrations/kernels/cfd/bench.yaml
name: cfd_pressure_poisson
description: "Tensorized multigrid V-cycle solver"

configs:
  - name: small
    grid_size: [32, 32, 32]
    max_iterations: 10
    
  - name: large
    grid_size: [128, 128, 128]
    max_iterations: 30

precisions:
  - fp32
  - fp16

backends:
  - cpu
  - cuda
  - jax

tolerances:
  rmse: 1e-3
  mae: 1e-4

skip_if:
  - condition: backend == "cuda" and not has_cupy
    reason: "CuPy not available"

baseline:
  cuda:
    fp32:
      p50_ms: 0.5
      throughput: 2000
```

## CI/CD Integration

The benchmark suite is integrated with GitHub Actions via `.github/workflows/bench.yml`:

### Automatic Execution

Benchmarks run automatically on:
- Push to `main`, `develop`, or `copilot/**` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

### Matrix Strategy

Tests multiple configurations:
- Backend: cpu, cuda, rocm (based on availability)
- Precision: fp32, fp16

### Artifacts

- Results uploaded as workflow artifacts (30-day retention)
- Summary uploaded separately (90-day retention)

### PR Comments

On pull requests, the workflow automatically:
- Posts benchmark results as a PR comment
- Updates existing comment on subsequent runs
- Includes links to full artifacts

### Main Branch Publishing

On push to `main`:
- Commits summary to `benchmarks/latest_summary.md`
- Archives results to `benchmarks/history/`
- Creates/updates GitHub issue with checklist

## Architecture

### `bench_all.py`

Main orchestrator script that:
1. Discovers kernels in repository
2. Captures system environment
3. Runs benchmarks with warmup and timed iterations
4. Collects metrics (timing, memory, energy)
5. Generates comprehensive reports
6. Detects regressions vs. baseline

### `metrics.py`

Utilities library providing:
- Data classes for metrics (Timing, Memory, Energy, Accuracy)
- Timer context manager for precise measurements
- GPU memory tracking (PyTorch, CuPy)
- GPU energy estimation (nvidia-smi)
- Statistical computations (percentiles)
- Markdown table formatting
- System information collection
- Regression detection

## Extending the Suite

### Adding New Kernels

Kernels are auto-discovered if placed in:
- `integrations/kernels/`
- `autonomous_systems_platform/services/backend/quasim/kernels/`
- `kernels/`
- `src/kernels/`
- `quasim/kernels/`

Ensure kernels follow conventions:
- Expose a `solve()`, `run()`, or `<name>_kernel()` function
- Accept configuration parameters
- Support deterministic execution with seed parameter

### Adding New Metrics

Extend the metrics data classes in `metrics.py`:

```python
@dataclass
class CustomMetrics:
    my_metric: float = 0.0
    
    def to_dict(self) -> dict[str, float]:
        return {"my_metric": self.my_metric}
```

Add to `BenchmarkResult`:

```python
custom: CustomMetrics = field(default_factory=CustomMetrics)
```

### Adding New Backends

Update backend detection in `KernelDiscovery._detect_backend()`:

```python
elif "my_backend" in content:
    return "my_backend"
```

Add backend filtering in `BenchmarkOrchestrator.run()`.

## Best Practices

1. **Reproducibility**: Always use the same seed for comparable results
2. **Warmup**: Use at least 3 warmup iterations to prime caches
3. **Iterations**: Use 30+ iterations for stable statistics
4. **Isolation**: Run benchmarks on idle systems for accurate measurements
5. **Baselines**: Establish baselines early and track trends over time
6. **CI**: Run benchmarks on every PR to catch regressions early

## Troubleshooting

### Kernel Not Discovered

- Verify kernel is in a supported directory
- Check that file is not named `__init__.py`
- Ensure file has `.py` extension

### Import Errors

- Install required dependencies: `pip install numpy`
- For GPU support: Install PyTorch, CuPy, or JAX
- Check that kernel modules are on Python path

### Memory Issues

- Reduce problem sizes in kernel configurations
- Run fewer iterations: `--iters 10`
- Run single precision: `--precision fp32`
- Run specific backend: `--backends cpu`

### Timing Variance

- Increase iterations: `--iters 100`
- Ensure system is idle
- Disable power management on GPUs
- Pin CPU affinity if possible

## Performance Guidelines

Expected benchmark performance:

- **CFD Kernels**: 0.3-5ms per iteration (grid-dependent)
- **Autonomous Systems**: 0.1-1ms per iteration
- **Quantum Circuits**: 1-100ms (qubit-count dependent)
- **Digital Twins**: 10-1000ms (complexity-dependent)

## Support

For issues or questions:
- Check existing kernel configurations in `kernels/*/bench.yaml`
- Review GitHub Actions logs for CI failures
- Consult QuASIM documentation
- Open an issue with benchmark results attached

## License

Part of the QuASIM project. See root LICENSE file.
