# QuASIM Benchmark Suite

This directory contains microbenchmark harnesses for all compute kernels in the GB10 QuASIM reference platform.

## Directory Structure

```
benchmarks/
├── README.md              # This file
├── quasim_bench.py        # Legacy benchmark driver (retained for compatibility)
├── harnesses/             # Individual kernel benchmark harnesses
│   ├── bench_k001_tensor_contraction.py
│   ├── bench_k002_tensor_generation.py
│   └── bench_k003_columnar_sum.py
└── results/               # Cached benchmark results (JSON format)
    └── <commit-sha>.json
```

## Running Benchmarks

### Quick Start

Run all benchmarks with default parameters:
```bash
make bench
```

Or directly:
```bash
PYTHONPATH=runtime/python:quantum python3 benchmarks/quasim_bench.py
```

### Individual Kernel Benchmarks

Run specific kernel benchmarks:
```bash
# K001: Tensor Contraction
PYTHONPATH=runtime/python:quantum python3 -m pytest benchmarks/harnesses/bench_k001_tensor_contraction.py -v

# K002: Tensor Generation
PYTHONPATH=runtime/python:quantum python3 -m pytest benchmarks/harnesses/bench_k002_tensor_generation.py -v

# K003: Columnar Sum
PYTHONPATH=runtime/python:quantum python3 -m pytest benchmarks/harnesses/bench_k003_columnar_sum.py -v
```

### CPU vs GPU

Currently, all kernels run on CPU. When GPU implementations are added:
- Use `CUDA_VISIBLE_DEVICES=0` to select GPU
- Use `CUDA_VISIBLE_DEVICES=-1` or omit to run on CPU only
- Benchmarks will automatically detect and report device type

## Benchmark Output Format

Each benchmark reports:
- **Median**: 50th percentile latency (most stable metric)
- **P95**: 95th percentile latency (tail latency)
- **Min/Max**: Range of observed latencies
- **FLOP/s**: Estimated floating-point operations per second (if computable)
- **GB/s**: Memory bandwidth utilization (if computable)
- **Achieved Occupancy**: GPU occupancy percentage (GPU kernels only)

### Example Output

```
QuASIM Tensor Benchmark — batches=32 rank=4 dim=2048
=======================================================
runs:        5
min (s):     0.002451
median (s):  0.002563
mean (s):    0.002589
max (s):     0.002801
elements/s:  25,333,333
```

## Reproducibility

All benchmarks:
- Fix RNG seeds (`random.seed(42)`, `np.random.seed(42)`, etc.)
- Warm up JIT compilers before timing
- Report confidence intervals or standard deviation
- Run multiple iterations (default: 5-10)

## Comparing Before/After

To compare optimization results:

1. Run baseline benchmarks and save:
   ```bash
   python3 benchmarks/harnesses/bench_k001_tensor_contraction.py --save baseline.json
   ```

2. Apply optimizations

3. Run optimized benchmarks:
   ```bash
   python3 benchmarks/harnesses/bench_k001_tensor_contraction.py --save optimized.json
   ```

4. Compare results:
   ```bash
   python3 scripts/compare_benchmarks.py baseline.json optimized.json
   ```

Expected output:
```
Kernel: K001 (Tensor Contraction)
Speedup: 2.3×
Latency: 2.563ms → 1.114ms (↓56.5%)
Throughput: 25.3M elem/s → 58.2M elem/s (↑130%)
```

## Golden Output Tests

Numerical correctness tests are located in `tests/data/` and `tests/software/`.
Before/after optimization, verify:
```bash
PYTHONPATH=runtime/python:quantum python3 -m pytest tests/software/ -v
```

All optimizations must maintain:
- Bit-for-bit identical results (when specified), OR
- Relative error < 1e-6 for floating-point operations
- Deterministic output (with fixed seeds)

## Adding New Benchmarks

When adding a new kernel benchmark:

1. Create `benchmarks/harnesses/bench_kXXX_<name>.py`
2. Use pytest-benchmark or torch.utils.benchmark framework
3. Include:
   - Multiple problem sizes (small, medium, large)
   - Dtype variations (fp8, fp16, fp32, fp64, complex)
   - RNG seed initialization
   - Warmup iterations
   - Statistical reporting (median, p95)
4. Add entry to `kernels/MANIFEST.md`
5. Update this README with usage instructions

## Benchmark Frameworks

- **Python/NumPy**: `pytest-benchmark` or `timeit`
- **PyTorch**: `torch.utils.benchmark.Timer`
- **JAX**: `jax.profiler` + `timeit` + `jax.block_until_ready()`
- **CUDA/HIP**: Google Benchmark or `pytest-benchmark` with ctypes
- **Triton**: `triton.testing.perf_report`

## Performance Targets

Optimization acceptance criteria (see kernels/MANIFEST.md):
- ≥1.3× speedup, OR
- ≥30% energy/latency reduction
- No numerical correctness regression
- Maintain portability across backends

## CI Integration

Benchmarks run in CI on CPU with reduced iterations.
Full GPU benchmarks should be run locally or on dedicated benchmark infrastructure.

See `.github/workflows/ci.yml` for CI configuration.

---

**Last Updated**: 2025-11-01  
**Benchmark Suite Version**: 1.0
