# QuASIM Kernel Manifest

This document catalogs all compute kernels in the GB10 QuASIM reference platform, including their locations, backends, characteristics, and optimization status.

## Manifest Format

Each kernel entry includes:
- **ID**: Unique identifier (K001, K002, etc.)
- **Name**: Descriptive kernel name
- **File Path**: Location in repository
- **Backend**: Execution backend (Python, CUDA, HIP, JAX, PyTorch, Triton)
- **Function/Entry**: Primary function or entry point
- **Tensor Shapes**: Input/output tensor dimensions
- **Data Types**: Supported dtypes (fp8, fp16, fp32, fp64, complex)
- **Hot Call Sites**: Primary invocation locations
- **Estimated FLOPS**: Theoretical floating-point operations
- **Status**: Current optimization state

---

## Python Backend Kernels

### K001: Tensor Contraction Simulator
- **ID**: K001
- **Name**: Tensor Contraction Simulator
- **File Path**: `runtime/python/quasim/runtime.py`
- **Backend**: Python (pure)
- **Function/Entry**: `_RuntimeHandle.simulate()`
- **Tensor Shapes**: Variable batches × variable dimension (default: 32 × 2048)
- **Data Types**: complex64, complex128
- **Hot Call Sites**:
  - `quantum/python/quasim_sim.py:11` (simulate function)
  - `benchmarks/quasim_bench.py:36` (run_benchmark)
  - `tests/software/test_quasim.py:14` (test_simulation_roundtrip)
- **Estimated FLOPS**: O(batches × dimension) additions per invocation
- **Current Performance**: ~5.0ms median (32 batches, 2048 dim)
- **Status**: ⚠️ BASELINE - Pure Python loop, no vectorization
- **Optimization Opportunities**:
  - Vectorize with NumPy/CuPy
  - Parallelize batch processing
  - Use JAX JIT compilation
  - Memory-efficient accumulation (Kahan summation)

### K002: Tensor Generation
- **ID**: K002
- **Name**: Tensor Generation (Benchmark Workload)
- **File Path**: `benchmarks/quasim_bench.py`
- **Backend**: Python (pure)
- **Function/Entry**: `_generate_tensor()`
- **Tensor Shapes**: 1D arrays of configurable dimension (default: 2048)
- **Data Types**: complex64, complex128
- **Hot Call Sites**:
  - `benchmarks/quasim_bench.py:25` (_generate_workload)
- **Estimated FLOPS**: O(dimension) complex multiplications + additions
- **Status**: ⚠️ BASELINE - List comprehension, sequential
- **Optimization Opportunities**:
  - Pre-allocate NumPy arrays
  - Vectorize arithmetic operations
  - Cache pre-computed tensors

### K003: Columnar Aggregation
- **ID**: K003
- **Name**: Columnar Sum (RAPIDS-inspired)
- **File Path**: `sdk/rapids/dataframe.py`
- **Backend**: Python (pure)
- **Function/Entry**: `columnar_sum()`
- **Tensor Shapes**: Dictionary of columns, each with variable length
- **Data Types**: float32, float64
- **Hot Call Sites**: None identified (utility function)
- **Estimated FLOPS**: O(columns × rows) additions
- **Status**: ⚠️ BASELINE - Pure Python dict comprehension
- **Optimization Opportunities**:
  - Use NumPy sum operations
  - Parallel column processing
  - SIMD acceleration via Numba
  - GPU offload with CuPy/JAX

### K004: VQE Circuit Simulation
- **ID**: K004
- **Name**: VQE Hamiltonian Simulator
- **File Path**: `quantum/examples/vqe.py`
- **Backend**: Python (pure)
- **Function/Entry**: `run_vqe()`
- **Tensor Shapes**: n_qubits gates × 4 elements (default: 4 qubits)
- **Data Types**: complex64
- **Hot Call Sites**:
  - `quantum/examples/vqe.py:22` (main execution)
- **Estimated FLOPS**: O(n_qubits) gate operations + reductions
- **Status**: ⚠️ BASELINE - Delegates to K001
- **Optimization Opportunities**:
  - Optimize underlying simulate() call (K001)
  - Fuse gate operations
  - Exploit quantum circuit structure

### K005: Profiler Sampling
- **ID**: K005
- **Name**: Runtime Telemetry Collector
- **File Path**: `sdk/profiler/gb10_profiler.py`
- **Backend**: Python (pure)
- **Function/Entry**: `collect_samples()`
- **Tensor Shapes**: 1D array of sample_count (default: 16)
- **Data Types**: float64
- **Hot Call Sites**:
  - `sdk/profiler/gb10_profiler.py:17` (main CLI)
- **Estimated FLOPS**: Minimal (RNG dominated)
- **Status**: ✅ UTILITY - Not performance critical
- **Optimization Opportunities**: Low priority

---

## Future Backend Kernels

### Planned CUDA/HIP Kernels
When CUDA or HIP implementations are added, they will be cataloged here with:
- PTX/SASS assembly analysis
- Register usage and occupancy
- Memory access patterns
- Roofline model positioning

### Planned JAX/XLA Kernels
When JAX primitives are implemented:
- XLA HLO IR analysis
- Fusion boundaries
- Device synchronization points
- Donation buffer opportunities

### Planned Triton Kernels
When Triton implementations are added:
- Block/warp tile configurations
- Autotuning parameter space
- Shared memory usage
- Tensor Core utilization

---

## Hotspot Priority Ranking

Based on cumulative execution time in typical workloads:

1. **K001 (Tensor Contraction)** - HIGHEST PRIORITY
   - Primary compute bottleneck
   - Called in tight loops
   - ~80% of total runtime in benchmarks

2. **K002 (Tensor Generation)** - HIGH PRIORITY
   - Called for every batch
   - ~15% of total runtime

3. **K003 (Columnar Sum)** - MEDIUM PRIORITY
   - Utility function, usage pattern unclear
   - Optimization provides general benefit

4. **K004 (VQE Circuit)** - MEDIUM PRIORITY
   - Depends on K001 optimization
   - Application-specific workload

5. **K005 (Profiler)** - LOW PRIORITY
   - Diagnostic tool, not in critical path

---

## Optimization Tracking

| Kernel | Baseline (Medium) | Optimized (Medium) | Speedup | Backend Variants | Last Profiled |
|--------|-------------------|---------------------|---------|------------------|---------------|
| K001   | 3.807ms, 17.2 M/s | 3.249ms, 20.2 M/s  | 1.17× (1.82× large) | Python + NumPy | 2025-11-01 |
| K002   | 10.035ms, 6.5 M/s | 1.346ms, 48.7 M/s  | **7.46×** ✅ | Python + NumPy | 2025-11-01 |
| K003   | 3.893ms, 128 M/s  | (baseline retained) | 1.0× | Python (optimal) | 2025-11-01 |
| K004   | (depends on K001) | (depends on K001)  | 1.17×+ | Python | 2025-11-01 |
| K005   | N/A               | N/A                | N/A     | Python           | 2025-11-01 |

**Notes:**
- K001: NumPy vectorization; 1.82× speedup on large workloads (64×4096)
- K002: NumPy vectorization; **7.46× speedup** (exceptional performance gain)
- K003: Built-in sum() already optimal; NumPy conversion adds overhead

---

## TODO Tags Reference

Search for `# TODO(K001)`, `# TODO(K002)`, etc. in the codebase to find call sites marked for optimization tracking.

---

**Last Updated**: 2025-11-01  
**Manifest Version**: 1.0  
**Total Kernels Cataloged**: 5 (5 Python, 0 CUDA, 0 HIP, 0 JAX, 0 Triton)
