# RUN_MANIFEST.md - Benchmark Execution Chain-of-Custody

**Document Type**: Audit-Ready Run Manifest  
**Generated**: 2026-01-18 20:39:37 UTC  
**Git Commit**: d6bafe894be45d50e63a36241e536bfe70d2f6b3  
**Benchmark Session**: 20260118_203732  

---

## 1. CHAIN-OF-CUSTODY INITIALIZATION

### Repository State at PR Start
- **Commit Hash**: d6bafe894be45d50e63a36241e536bfe70d2f6b3
- **Commit Timestamp**: 2026-01-18 20:34:51 +0000
- **Branch**: copilot/maximize-qratum-infrastructure-capacity
- **Working Tree**: Clean (no uncommitted changes)

### Deterministic Seeds
All benchmarks use fixed random seeds for reproducibility:
- Default Seed: 42
- Chess Position Seed: 42
- Graph Generation Seed: 42

---

## 2. BENCHMARK ENUMERATION

### IN-SCOPE BENCHMARKS

| ID | Subsystem | Benchmark File | Status |
|----|-----------|----------------|--------|
| BM-001 | QuASIM Tensor | benchmarks/quasim_bench.py | ✅ Included |
| BM-002 | UltraSSSP | demo_ultra_sssp.py | ✅ Included |
| BM-003 | PostDijkstra | benchmarks/post_dijkstra_benchmark.py | ✅ Included |
| BM-004 | Automated Suite | benchmarks/automated/benchmark_runner.py | ✅ Included |
| BM-005 | QRATUM-Chess | run_qratum_chess_benchmark.py | ✅ Included |
| BM-006 | Full Benchmark | run_full_benchmark.py | ✅ Included |

### EXCLUDED BENCHMARKS

| Subsystem | Reason for Exclusion |
|-----------|---------------------|
| GPU/CUDA Benchmarks | No NVIDIA GPU detected |
| Qiskit/PennyLane | Quantum hardware simulation requires additional dependencies |
| Kaggle Integration | Requires external API credentials |
| Adversarial Gauntlet | Requires external Stockfish/Lc0 engines |
| Genomics Pipeline | Requires external reference data |

---

## 3. INVOCATION ORDER

### Phase 1: Core Algorithm Benchmarks
1. QuASIM Tensor Benchmark (BM-001)
2. UltraSSSP Benchmark (BM-002)
3. PostDijkstra Benchmark (BM-003)

### Phase 2: Platform Benchmarks
4. Automated Benchmark Suite (BM-004)

### Phase 3: Chess Engine Benchmarks
5. QRATUM-Chess Benchmark (BM-005)
6. Full Benchmark Suite (BM-006)

---

## 4. EXECUTION PARAMETERS

### BM-001: QuASIM Tensor Benchmark
```bash
python benchmarks/quasim_bench.py --batches 32 --rank 4 --dimension 2048 --repeat 5
```
- **Batches**: 32
- **Rank**: 4
- **Dimension**: 2048
- **Repeat**: 5

### BM-002: UltraSSSP Benchmark
```bash
python demo_ultra_sssp.py --nodes 1000 --edge-prob 0.01 --batch-size 100 --seed 42
```
- **Nodes**: 1000
- **Edge Probability**: 0.01
- **Batch Size**: 100
- **Seed**: 42

### BM-003: PostDijkstra Benchmark
```bash
python benchmarks/post_dijkstra_benchmark.py --output post_dijkstra_results.json
```
- **Weight Distribution**: uniform
- **Seed**: 42

### BM-004: Automated Benchmark Suite
```bash
python benchmarks/automated/benchmark_runner.py --backend cpu
```
- **Backend**: cpu

### BM-005: QRATUM-Chess Benchmark
```bash
python run_qratum_chess_benchmark.py --quick --output-dir benchmarks/auto_run
```
- **Mode**: Quick (reduced iterations)
- **Torture Depth**: 8
- **Resilience Iterations**: 3

### BM-006: Full Benchmark Suite
```bash
python run_full_benchmark.py --quick --no-elo --output-dir benchmarks/auto_run
```
- **Mode**: Quick
- **Elo Certification**: Disabled (requires external engines)

---

## 5. ENVIRONMENT VARIABLES

| Variable | Value | Purpose |
|----------|-------|---------|
| PYTHONHASHSEED | 42 | Deterministic hashing |
| OMP_NUM_THREADS | 4 | OpenMP thread count |
| NUMBA_NUM_THREADS | 4 | Numba thread count |
| OPENBLAS_NUM_THREADS | 4 | OpenBLAS thread count |

---

## 6. HARDWARE AFFINITY

- **CPU Binding**: None (system managed)
- **NUMA Policy**: Default (single NUMA node)
- **Memory Policy**: Default allocation

---

## 7. VALIDATION CHECKLIST

- [ ] All benchmarks complete without crashes
- [ ] Deterministic outputs verified (where applicable)
- [ ] No degraded execution paths triggered
- [ ] All results persisted to benchmarks/results/

---

**END OF RUN MANIFEST**

---

## 8. GPU EXECUTION ENVIRONMENT ASSESSMENT

### Current Environment
- **GPU Available**: ❌ None
- **Runner Type**: GitHub Actions `ubuntu-latest`
- **CPU-Only Mode**: All benchmarks will execute in CPU fallback mode

### GPU Requirements Summary
See `GPU_REQUIREMENTS.md` for detailed requirements:
- **Minimum**: NVIDIA A100 40GB (SM 8.0)
- **CUDA Version**: 12.4.x
- **Driver**: 535.x

### GPU Execution Options Evaluated
See `GPU_ENVIRONMENT_OPTIONS.md` for full assessment:
- **Recommended**: GCP a2-highgpu-1g (~$3.67/hr)
- **Alternative**: AWS p4d.24xlarge, Azure NC24ads_A100_v4
- **Disqualified**: Google Colab, Kaggle (non-deterministic)

### CPU Benchmark Validity
All QRATUM benchmarks support CPU-only execution:
- BM-001 (QuASIM Tensor): ✅ Valid on CPU
- BM-002 (UltraSSSP): ✅ CPU-only algorithm
- BM-003 (PostDijkstra): ✅ CPU-only algorithm  
- BM-004 (Automated Suite): ✅ Valid on CPU
- BM-005/006 (Chess): ✅ Valid on CPU

**Conclusion**: CPU-only benchmark results are valid and defensible for the algorithms tested. GPU benchmarks require provisioning of external compute resources.

