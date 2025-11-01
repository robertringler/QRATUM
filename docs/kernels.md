# QuASIM Compute Kernels Documentation

This document provides comprehensive documentation for all compute kernels in the GB10 QuASIM reference platform, including their implementation details, optimization status, and performance characteristics.

## Overview

The QuASIM platform implements quantum simulation and classical compute workloads across multiple backends. This document tracks the optimization journey of each kernel from baseline implementation to optimized versions.

## Quick Reference

| Kernel ID | Name | Backend(s) | Status | Speedup | Test Status | Last Profile |
|-----------|------|------------|--------|---------|-------------|--------------|
| K001 | Tensor Contraction | Python | ⚠️ Baseline | 1.0× | ✅ Pass | 2025-11-01 |
| K002 | Tensor Generation | Python | ⚠️ Baseline | 1.0× | ✅ Pass | 2025-11-01 |
| K003 | Columnar Sum | Python | ⚠️ Baseline | 1.0× | ✅ Pass | 2025-11-01 |
| K004 | VQE Circuit Sim | Python | ⚠️ Baseline | 1.0× | ✅ Pass | 2025-11-01 |
| K005 | Profiler Sampling | Python | ✅ Utility | N/A | ✅ Pass | 2025-11-01 |

**Status Legend:**
- ⚠️ Baseline: Initial implementation, not yet optimized
- 🔧 In Progress: Optimization work underway
- ✅ Optimized: Performance targets met
- 🔄 Maintenance: Periodic review for new opportunities

---

## Kernel Details

### K001: Tensor Contraction Simulator

**Location:** `runtime/python/quasim/runtime.py`  
**Entry Point:** `_RuntimeHandle.simulate()`  
**Backend:** Pure Python

#### Purpose
Simulates tensor contraction operations for quantum circuit simulation. This is the primary compute kernel in the QuASIM runtime, responsible for processing batches of complex-valued tensors.

#### Expected Shapes/Dtypes
- **Input:** Iterable of iterables of `complex` (Python complex type)
- **Typical Shapes:** 
  - Small: 8 batches × 256 elements
  - Medium: 32 batches × 2048 elements
  - Large: 64 batches × 4096 elements
- **Data Types:** complex64, complex128 (Python complex maps to complex128)

#### Mathematical Summary
For each tensor batch `T[i]`, compute:
```
result[i] = Σ(T[i][j]) for all j
```
Sequential reduction over tensor elements with complex arithmetic.

#### Current Implementation
Pure Python nested loops:
```python
for tensor in tensors:
    total = 0 + 0j
    for value in tensor:
        total += complex(value)
    aggregates.append(total)
```

#### Performance Baseline (CPU)
- **Small (8×256):** 0.124ms median, 16.5 M elem/s
- **Medium (32×2048):** 3.807ms median, 17.2 M elem/s
- **Large (64×4096):** 15.378ms median, 17.0 M elem/s

#### Tiling Strategy
N/A (baseline implementation has no tiling)

#### Tunables
- `simulation_precision`: "fp8", "fp16", "fp32" (currently ignored in Python)
- `max_workspace_mb`: Maximum memory allocation (currently unused)

#### Optimization Opportunities
1. **Vectorization:** Use NumPy for SIMD operations
2. **Parallelization:** Process batches in parallel (multiprocessing/threading)
3. **JAX JIT:** Compile with `@jax.jit` for XLA optimization
4. **Numerical Stability:** Implement Kahan/Neumaier summation
5. **GPU Offload:** CuPy or JAX GPU backend

#### Fallback Path
Current pure Python implementation serves as fallback when NumPy/JAX unavailable.

---

### K002: Tensor Generation

**Location:** `benchmarks/quasim_bench.py`  
**Entry Point:** `_generate_tensor()`  
**Backend:** Pure Python

#### Purpose
Generates deterministic synthetic tensor payloads for benchmarking. Ensures reproducible workloads across benchmark runs.

#### Expected Shapes/Dtypes
- **Output:** 1D list of `complex` values
- **Typical Dimensions:** 256, 512, 1024, 2048, 4096
- **Data Types:** complex64, complex128

#### Mathematical Summary
For dimension `d`, rank `r`:
```
scale = r + 1
step = 1.0 / (d - 1)
tensor[i] = complex(i * step * scale, -i * step * scale) for i in [0, d)
```

#### Current Implementation
Python list comprehension:
```python
[complex(idx * step * scale, -idx * step * scale) for idx in range(dimension)]
```

#### Performance Baseline (CPU)
- **Small (8×256):** 0.309ms median, 6.6 M elem/s
- **Medium (32×2048):** 10.035ms median, 6.5 M elem/s
- **Large (64×4096):** 40.179ms median, 6.5 M elem/s

#### Optimization Opportunities
1. **Pre-allocation:** Use `numpy.empty()` + vectorized fill
2. **Caching:** Pre-compute and cache common tensor sizes
3. **Vectorization:** Replace list comprehension with NumPy operations

---

### K003: Columnar Sum

**Location:** `sdk/rapids/dataframe.py`  
**Entry Point:** `columnar_sum()`  
**Backend:** Pure Python

#### Purpose
RAPIDS-inspired columnar aggregation utility. Computes column-wise sums over dataframe-like structures represented as dictionaries of lists.

#### Expected Shapes/Dtypes
- **Input:** `dict[str, list[float]]` (column name → values)
- **Typical Shapes:**
  - Small: 10 columns × 1,000 rows
  - Medium: 50 columns × 10,000 rows
  - Large: 100 columns × 50,000 rows
- **Data Types:** float32, float64

#### Mathematical Summary
For each column `C[col]`:
```
result[col] = Σ(C[col][i]) for all i
```
Independent reduction per column (embarrassingly parallel).

#### Current Implementation
Dictionary comprehension with built-in `sum()`:
```python
{name: float(sum(column)) for name, column in table.items()}
```

#### Performance Baseline (CPU)
- **Small (10×1K):** 0.078ms median, 128.5 M elem/s
- **Medium (50×10K):** 3.893ms median, 128.4 M elem/s
- **Large (100×50K):** 40.737ms median, 122.7 M elem/s

#### Optimization Opportunities
1. **NumPy:** Use `numpy.sum(axis=0)` for vectorization
2. **Parallel:** Process columns in parallel (ThreadPoolExecutor)
3. **GPU:** CuPy for GPU-accelerated sums
4. **SIMD:** Numba JIT with explicit SIMD pragmas

---

### K004: VQE Circuit Simulation

**Location:** `quantum/examples/vqe.py`  
**Entry Point:** `run_vqe()`  
**Backend:** Pure Python (delegates to K001)

#### Purpose
Variational Quantum Eigensolver workflow demonstrating quantum chemistry applications. Constructs Heisenberg Hamiltonian and estimates ground state energy.

#### Expected Shapes/Dtypes
- **Input:** `n_qubits` (typically 2-8)
- **Gate Representation:** 4-element complex arrays per qubit
- **Data Types:** complex64

#### Mathematical Summary
1. Build Hamiltonian gates (Pauli operators)
2. Simulate circuit via tensor contraction (calls K001)
3. Compute energy expectation: `E = Σ|amplitude|²`

#### Performance Baseline
Dominated by K001 performance; optimization of K001 directly benefits K004.

#### Optimization Opportunities
1. **Gate Fusion:** Combine consecutive gates before simulation
2. **Circuit Optimization:** Eliminate redundant gates
3. **Depends on K001:** Primary benefit from K001 optimization

---

### K005: Runtime Telemetry Collector

**Location:** `sdk/profiler/gb10_profiler.py`  
**Entry Point:** `collect_samples()`  
**Backend:** Pure Python

#### Purpose
Diagnostic utility for collecting runtime performance samples. Used for profiling infrastructure, not in critical path.

#### Status
✅ **UTILITY** - Low priority for optimization. Sufficient performance for diagnostic purposes.

---

## Optimization Playbook

### General Principles

1. **Measure First:** Always profile before optimizing
2. **Numerical Correctness:** Verify <1e-6 relative error (or bit-exact when specified)
3. **Portability:** Maintain CPU fallback paths
4. **Documentation:** Update this doc with every optimization

### Vectorization Checklist
- [ ] Replace Python loops with NumPy operations
- [ ] Use `numpy.einsum()` for tensor contractions
- [ ] Enable SIMD flags (`-march=native` for compiled extensions)

### Parallelization Checklist
- [ ] Identify embarrassingly parallel operations (e.g., K003)
- [ ] Use `multiprocessing.Pool` or `concurrent.futures`
- [ ] Consider GIL overhead (pure Python) vs. GIL-free (NumPy/native)

### GPU Acceleration Checklist
- [ ] Assess data transfer overhead (PCIe bandwidth)
- [ ] Use CuPy for drop-in NumPy replacement
- [ ] JAX for auto-differentiation + GPU
- [ ] Triton for custom GPU kernels

### Numerical Stability Checklist
- [ ] Implement Kahan summation for reductions
- [ ] Use higher-precision accumulators (FP32→FP64, FP16→FP32)
- [ ] Provide deterministic mode (fixed reduction order)

---

## Testing Strategy

### Correctness Tests
- **Location:** `tests/software/test_quasim.py`
- **Framework:** pytest
- **Coverage:** All kernels must have golden output tests

### Property-Based Tests
- **Framework:** Hypothesis
- **Coverage:** Random tensor shapes, dtypes, edge cases
- **Status:** Planned for validation phase

### Performance Regression Tests
- **Threshold:** >5% regression fails CI (soft limit on CPU, hard limit on GPU)
- **Baseline:** Stored in `benchmarks/results/<commit>.json`

---

## Profiling Guide

### Python Profiling
```bash
# cProfile for function-level analysis
./scripts/profile_py.sh k001

# Line-level profiling (requires line_profiler)
./scripts/profile_py.sh --mode line k001

# Memory profiling (requires memory_profiler)
./scripts/profile_py.sh --mode memory k001
```

### CUDA Profiling (when available)
```bash
# Nsight Systems (timeline)
./scripts/profile_nvidia.sh --mode nsys kernel_name

# Nsight Compute (detailed metrics)
./scripts/profile_nvidia.sh --mode ncu kernel_name
```

### ROCm Profiling (when available)
```bash
# rocprof for kernel metrics
./scripts/profile_amd.sh --mode rocprof kernel_name
```

---

## CI/CD Integration

### Automated Checks (see `.github/workflows/ci.yml`)
1. **Lint:** ruff, mypy, pylint
2. **Unit Tests:** pytest on CPU
3. **Benchmarks:** Reduced iteration count for CI
4. **Docs:** Validate kernel manifest and TODO tags

### Manual Verification (before merging)
- [ ] Run full benchmark suite locally
- [ ] Profile with production workload sizes
- [ ] Verify GPU performance (if GPU kernels added)
- [ ] Update kernel table in this document

---

## Future Backend Support

### CUDA/HIP Kernels
When GPU kernels are added:
- Thread block configuration (`BLOCK_SIZE`, `GRID_SIZE`)
- Shared memory tiling strategies
- Warp-level primitives (`__shfl_sync`)
- Tensor Core utilization (if applicable)

### JAX/XLA Kernels
When JAX is integrated:
- `@jax.jit` boundaries
- `donate_argnums` for buffer reuse
- XLA fusion analysis
- TPU-specific optimizations

### Triton Kernels
When Triton is added:
- Autotuning configs (`BLOCK_M`, `BLOCK_N`, `num_warps`, `num_stages`)
- Pointer arithmetic patterns
- Load/store vectorization

---

## Performance Targets

### Acceptance Criteria
An optimization PR is accepted if:
- ≥1.3× speedup **OR** ≥30% energy/latency reduction
- All tests pass (no numerical regressions)
- Documentation updated (this file + MANIFEST.md)
- Profiling data attached to PR

### Stretch Goals
- ≥2× speedup: Excellent
- ≥5× speedup: Outstanding
- ≥10× speedup: Exceptional (publish results!)

---

## Contact & Contributions

For questions about kernel optimizations:
1. Check `kernels/MANIFEST.md` for detailed kernel specs
2. Review `benchmarks/README.md` for benchmark guidelines
3. See existing PRs tagged with `kernel:<name>` for examples

**Last Updated:** 2025-11-01  
**Document Version:** 1.0  
**Maintainer:** QuASIM Kernel Team
