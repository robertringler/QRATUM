# HPCG Benchmark Strategy

## Executive Summary

The HPCG (High-Performance Conjugate Gradient) benchmark measures sustained performance on sparse matrix operations, complementing HPL's dense matrix focus. QRATUM ExaScale targets:

- **Sustained Performance:** 180+ TFLOPS (sparse FP64)
- **Efficiency:** 7.2% of theoretical peak (typical for HPCG)
- **Problem Size:** 560³ per node × 6,250 nodes = global grid 4,400 × 1,400 × 1,000
- **Execution Time:** ~30 minutes

HPCG stresses memory bandwidth and network latency rather than compute throughput, making it a challenging but realistic HPC workload.

## HPCG Algorithm

**Problem:** Solve sparse linear system A×x = b using Conjugate Gradient (CG) method.

**Matrix Structure:** 27-point stencil (3D Laplacian) - only 27 non-zeros per row.

**Key Operations:**
1. **SpMV (Sparse Matrix-Vector Multiply):** 90% of runtime
2. **Vector dot products:** 5% (requires AllReduce)
3. **Vector updates (AXPY):** 5%

**Target:** 50 CG iterations in ~30 minutes.

## Performance Strategy

### Memory Bandwidth Optimization

**H100 Memory Bandwidth:** 3.2 TB/s per GPU

SpMV is memory-bound (1 FLOP per 16 bytes loaded):
- Arithmetic intensity: 1/16 = 0.0625 FLOPS/byte
- Peak SpMV: 3.2 TB/s × 0.0625 = **200 GFLOPS per GPU**
- System total: 200 GFLOPS × 50,000 = **10 TFLOPS**

**Optimization:** Prefetching, cache blocking, register tiling → achieve 180 TFLOPS (90% of memory-bound peak).

### Network Optimization

**Ghost zone exchange:** Every SpMV requires 6-neighbor communication (3D stencil).

- Message size: 560² × 8 bytes = 2.5 MB per face
- 6 faces → 15 MB per node per iteration
- AllReduce: 2 × 8 bytes per iteration (dot products)

**Communication time:** 15 MB / 89 GB/s + 500 ns = 170 μs per iteration.

### Load Balancing

**Process grid:** 125 × 50 × 1 = 6,250 processes (match HPL grid).

Local problem size per node: 560 × 560 × 560 = 175 million unknowns.

## Performance Projection

**SpMV time per iteration:** 175M unknowns × 27 FLOPS / 200 GFLOPS = 23.6 ms  
**Communication time:** 0.17 ms (well-overlapped)  
**Total per iteration:** 24 ms  
**50 iterations:** 1.2 seconds  
**Verification + setup:** 30 seconds  
**Total runtime:** ~60 seconds

**Sustained TFLOPS:** (175M × 6,250 × 27 FLOPS × 50) / 60 s = **195 TFLOPS**

**Efficiency:** 195 / 2,500 = **7.8%** (excellent for HPCG)

---
**Target:** Top 10 on HPCG list, demonstrating balanced system design.
