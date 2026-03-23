# Ansys-QuASIM Performance Comparison Methodology

**Document Version:** 1.0.0  
**Date:** 2025-12-13  
**Status:** Production-Ready  
**Target Audience:** Ansys Engineering, Industrial Validation Partners

---

## Executive Summary

This document specifies the performance validation methodology for Ansys-QuASIM integration. It ensures **reproducible, statistically rigorous, and adversarially defensible** performance comparisons between Ansys Mechanical (CPU baseline) and QuASIM (GPU-accelerated) solvers.

**Key Principles:**

1. **Bit-exact reproducibility** - SHA-256 hash verification of results
2. **Statistical rigor** - 5 runs per benchmark with confidence intervals
3. **Controlled hardware** - Fixed CPU/GPU configuration, cooldown periods
4. **Accuracy first** - Performance irrelevant if accuracy fails
5. **Transparent reporting** - All data, outliers, and failures documented

---

## 1. Hardware Configuration Specifications

### 1.1 CPU Baseline Configuration

**Required for Ansys reference:**

| Component | Specification | Justification |
|-----------|--------------|---------------|
| **CPU** | Intel Xeon Gold 6248R | Industry-standard datacenter CPU |
| Cores | 16 cores @ 3.0 GHz | Typical workstation configuration |
| Memory | 64 GB DDR4-2933 | Sufficient for 100k element models |
| Storage | NVMe SSD (1 TB) | Fast I/O for result files |
| OS | Ubuntu 22.04 LTS | Stable, reproducible environment |
| Kernel | 5.15+ | Long-term support |

**Alternatives (with performance normalization):**

- AMD EPYC 7763 (64-core) → normalize to 16-core equivalent
- Intel Xeon Platinum 8280 → similar performance to Gold 6248R
- ARM-based (AWS Graviton3) → not recommended (different ISA)

**Forbidden:**

- Laptop CPUs (thermal throttling)
- Overclocked systems (non-deterministic)
- Shared cloud instances (noisy neighbors)

### 1.2 GPU Accelerated Configuration

**Required for QuASIM:**

| Component | Specification | Justification |
|-----------|--------------|---------------|
| **GPU** | NVIDIA A100 80GB | Flagship datacenter GPU |
| Memory | 80 GB HBM2e | Handles largest benchmarks |
| CUDA Cores | 6912 | Maximum parallelism |
| Tensor Cores | 432 (3rd gen) | cuQuantum acceleration |
| Interconnect | NVLink 3.0 (600 GB/s) | Multi-GPU scaling |
| CPU | Same as baseline | Fair comparison |
| OS | Same as baseline | |

**Alternatives (with caveats):**

- NVIDIA A100 40GB → BM_005 may fail (OOM)
- NVIDIA H100 80GB → 1.5-2x faster (report separately)
- NVIDIA V100 32GB → Older architecture (report separately)
- AMD MI250X → ROCm support experimental

**Multi-GPU Configuration (for BM_005):**

- 4x NVIDIA A100 80GB
- NVLink mesh topology (all-to-all)
- PCIe Gen4 x16 fallback if NVLink unavailable

### 1.3 Network and Environment

**Network Isolation:**

- Dedicated VLAN or air-gapped network
- No internet access during benchmarking
- NTP time sync disabled (use system clock)

**System State:**

- No GUI (text console only)
- All non-essential services stopped
- CPU governor: `performance` (no frequency scaling)
- GPU persistence mode: enabled (`nvidia-smi -pm 1`)

**Thermal Management:**

- CPU temperature < 75°C (monitor with `sensors`)
- GPU temperature < 80°C (monitor with `nvidia-smi`)
- 60-second cooldown between runs

---

## 2. Software Version Requirements

### 2.1 Ansys Mechanical

**Primary Version:**

- **Ansys Mechanical 2024 R1** (version 24.1)
- MAPDL solver build: 24.1.0
- License: Mechanical Pro or Enterprise

**Compatibility:**

- 2024 R2: Supported (may have minor differences)
- 2023 R2: Supported (some features unavailable)
- 2023 R1 and earlier: **Not validated**

**Configuration:**

```bash
# Ansys environment variables
export ANSYS_RELEASE=241  # 2024 R1
export ANSYS_VERSION=v241
export ANSYS_ROOT=/ansys_inc/v241/ansys
export ANSYS_SYSDIR=linx64

# Disable telemetry
export ANSYS_NO_TELEMETRY=1

# Single-node execution (no MPI for baseline)
export ANSYS_MPI=0
```

### 2.2 QuASIM Software Stack

**Core Components:**

| Component | Version | Purpose |
|-----------|---------|---------|
| **QuASIM Core** | ≥0.1.0 | Simulation runtime |
| **QuASIM Ansys Adapter** | ≥1.0.0 | PyMAPDL integration |
| Python | 3.10+ (3.12 recommended) | Scripting |
| NumPy | ≥1.24.0 | Array operations |
| PyMAPDL | ≥0.68.0 | Ansys integration |
| PyYAML | ≥6.0 | Config parsing |

**GPU Stack:**

| Component | Version | Purpose |
|-----------|---------|---------|
| NVIDIA Driver | ≥535.86 | GPU support |
| CUDA Toolkit | 12.2+ | GPU programming |
| cuDNN | 8.9+ | Neural network primitives |
| cuQuantum | 23.10+ | Quantum tensor backend |

**Installation Verification:**

```bash
# Verify Python environment
python3 --version  # Should be ≥3.10
python3 -c "import numpy; print(numpy.__version__)"

# Verify GPU stack
nvidia-smi  # Check driver version and GPU
nvcc --version  # Check CUDA version

# Verify QuASIM installation
python3 -m quasim_ansys_adapter.test_installation
```

---

## 3. Pre-Execution Checklist

**System Preparation:**

- [ ] Verify hardware configuration matches spec
- [ ] Install all required software versions
- [ ] Run installation test (`test_installation.py`)
- [ ] Set CPU governor to `performance`
- [ ] Enable GPU persistence mode
- [ ] Disable swap (for deterministic memory behavior)
- [ ] Clear filesystem cache (`sync; echo 3 > /proc/sys/vm/drop_caches`)

**Benchmark Preparation:**

- [ ] Download `benchmark_definitions.yaml`
- [ ] Verify YAML syntax (`python3 -c "import yaml; yaml.safe_load(open('benchmark_definitions.yaml'))"`)
- [ ] Create output directory (`mkdir -p results/`)
- [ ] Set random seeds in config files

**Environment Variables:**

```bash
# Deterministic execution
export OMP_NUM_THREADS=16  # Match CPU core count
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0  # Single GPU for most benchmarks

# Disable GPU power management
export CUDA_AUTO_BOOST=0

# QuASIM configuration
export QUASIM_SEED=42
export QUASIM_LOG_LEVEL=INFO
export QUASIM_CACHE_DIR=/tmp/quasim_cache
```

---

## 4. Execution Sequence

### 4.1 Benchmark Execution Protocol

**For each benchmark (BM_001 through BM_005):**

1. **Cooldown Period** (60 seconds)
   - Let CPU/GPU temperatures stabilize
   - Monitor with `sensors` and `nvidia-smi`

2. **Run Ansys Baseline** (5 runs)

   ```bash
   for i in {1..5}; do
       echo "Ansys baseline run $i/5"
       python3 performance_runner.py \
           --benchmark BM_001 \
           --solver ansys \
           --output results/BM_001_ansys_run${i}.json
       sleep 60  # Cooldown
   done
   ```

3. **Compute Reference Hash**

   ```bash
   python3 performance_runner.py \
       --benchmark BM_001 \
       --compute-hash \
       --output results/BM_001_reference_hash.txt
   ```

4. **Run QuASIM Solver** (5 runs)

   ```bash
   for i in {1..5}; do
       echo "QuASIM run $i/5"
       python3 performance_runner.py \
           --benchmark BM_001 \
           --solver quasim \
           --device gpu \
           --output results/BM_001_quasim_run${i}.json
       sleep 60  # Cooldown
   done
   ```

5. **Validate Results**

   ```bash
   python3 performance_runner.py \
       --benchmark BM_001 \
       --validate \
       --ansys-results results/BM_001_ansys_*.json \
       --quasim-results results/BM_001_quasim_*.json
   ```

### 4.2 Timing Methodology

**What to Measure:**

- **Setup time:** Mesh import, material initialization (exclude from speedup)
- **Solve time:** Newton-Raphson iteration (include in speedup)
- **Postprocessing time:** Result extraction (exclude from speedup)
- **Total time:** Setup + Solve + Postprocessing (for reference only)

**How to Measure:**

```python
import time

# Start timing (after mesh import)
solve_start = time.time()

# Solver execution
solver.solve()

# Stop timing (before result export)
solve_time = time.time() - solve_start
```

**Clock Source:**

- Use `time.perf_counter()` for high-resolution timing
- Do NOT use `time.time()` (wall clock, subject to NTP adjustments)

### 4.3 Memory Measurement

**GPU Memory:**

```bash
# Peak memory usage during solve
nvidia-smi --query-gpu=memory.used --format=csv -lms 100 > gpu_mem.log &
GPU_MON_PID=$!

# Run solver
python3 performance_runner.py ...

# Stop monitoring
kill $GPU_MON_PID

# Extract peak
awk -F, 'NR>1 {if($1>max) max=$1} END {print max}' gpu_mem.log
```

**CPU Memory:**

```bash
# Use `/usr/bin/time -v` for detailed memory stats
/usr/bin/time -v python3 performance_runner.py ... 2>&1 | grep "Maximum resident"
```

---

## 5. Reproducibility Protocol

### 5.1 Deterministic Execution

**Requirements:**

- **Fixed random seed:** Set in benchmark YAML and environment variable
- **Single-threaded reduction:** Use deterministic summation (Kahan algorithm)
- **Sorted iteration order:** Iterate over elements/nodes in ascending ID order
- **Disabled dynamic scheduling:** Force static OpenMP schedule

**Verification:**

```bash
# Run same benchmark twice
python3 performance_runner.py --benchmark BM_001 --solver quasim --seed 42 --output run1.json
python3 performance_runner.py --benchmark BM_001 --solver quasim --seed 42 --output run2.json

# Compare hashes
diff <(jq -r '.state_hash' run1.json) <(jq -r '.state_hash' run2.json)
# Should output nothing (hashes identical)
```

### 5.2 SHA-256 Hash Verification

**What to Hash:**

- Nodal displacements (full precision, all nodes)
- Element stresses (integration point values)
- Convergence history (residual norms)

**How to Compute:**

```python
import hashlib
import numpy as np

def compute_state_hash(displacements: np.ndarray) -> str:
    """Compute SHA-256 hash of displacement field."""
    # Use full precision bytes representation
    data = displacements.astype(np.float64).tobytes()
    return hashlib.sha256(data).hexdigest()
```

**Acceptance Criteria:**

- All 5 Ansys runs must have **identical hash** (bit-exact reproducibility)
- All 5 QuASIM runs must have **identical hash**
- QuASIM hash must **differ** from Ansys hash (due to algorithmic differences)
- Displacement L2 error must be < 2% (despite different hash)

---

## 6. Accuracy Metrics

### 6.1 Displacement Error

**Metric:**

```
L2 displacement error = ||u_quasim - u_ansys||₂ / ||u_ansys||₂

where:
  u_quasim = QuASIM displacement field (num_nodes × 3)
  u_ansys  = Ansys reference displacement field
  || · ||₂ = Euclidean norm (L2)
```

**Computation:**

```python
def compute_displacement_error(u_quasim: np.ndarray, u_ansys: np.ndarray) -> float:
    """Compute L2 displacement error."""
    numerator = np.linalg.norm(u_quasim - u_ansys)
    denominator = np.linalg.norm(u_ansys)
    return numerator / denominator
```

**Threshold:**

- **Tier-0 acceptance:** < 2% for all benchmarks
- **Warning level:** 1-2% (acceptable, but investigate)
- **Failure:** > 2% (solver tuning required)

### 6.2 Stress Error

**Metric:**

```
Element-wise von Mises stress error:
  σ_vm = sqrt(1/2 * [(σ₁-σ₂)² + (σ₂-σ₃)² + (σ₃-σ₁)²])
  
Relative error per element:
  ε_i = |σ_vm_quasim_i - σ_vm_ansys_i| / σ_vm_ansys_i
  
Mean error:
  ε_mean = (1/N) Σ ε_i
```

**Threshold:**

- **Tier-0 acceptance:** < 5% mean error
- **Per-element tolerance:** < 15% (99th percentile)

### 6.3 Energy Conservation

**Metric:**

```
Strain energy error:
  E_strain = ∫ W(F) dV  (integrate strain energy density over volume)
  
Relative error:
  ε_energy = |E_quasim - E_ansys| / E_ansys
```

**Threshold:**

- **Tier-0 acceptance:** < 1e-6 (strain energy balance)

### 6.4 Contact Force Error (for BM_002, BM_005)

**Metric:**

```
Reaction force error:
  F_reaction = Σ (reaction forces at constrained nodes)
  
Relative error:
  ε_force = |F_quasim - F_ansys| / F_ansys
```

**Threshold:**

- **Tier-0 acceptance:** < 5% for normal direction
- **Tangential forces:** < 10% (friction is harder to match)

---

## 7. Performance Metrics

### 7.1 Speedup Definition

**Primary Metric:**

```
Speedup = T_ansys_median / T_quasim_median

where:
  T_ansys_median  = median solve time from 5 Ansys runs
  T_quasim_median = median solve time from 5 QuASIM runs
```

**Why Median?**

- Robust to outliers (single slow run doesn't skew results)
- More stable than mean for small sample sizes
- Industry-standard practice

**Reporting:**

- Report median ± 95% confidence interval
- Report min/max for transparency
- Flag outliers (> 1.5 × IQR from median)

### 7.2 Iteration Count

**Metric:**

```
Iteration efficiency = N_ansys / N_quasim

where:
  N_ansys  = Newton-Raphson iterations (Ansys)
  N_quasim = Newton-Raphson iterations (QuASIM)
```

**Target:**

- **Ideal:** N_quasim ≤ N_ansys (QuASIM converges in fewer iterations)
- **Acceptable:** N_quasim ≤ 1.2 × N_ansys (20% more iterations OK if faster per-iteration)

### 7.3 Memory Efficiency

**Metric:**

```
Memory overhead = M_quasim / M_ansys

where:
  M_ansys  = Peak CPU memory usage (Ansys)
  M_quasim = Peak GPU memory usage (QuASIM)
```

**Note:**

- CPU and GPU memory are not directly comparable
- Report absolute values (GB) for transparency
- GPU memory limit is hard constraint (OOM = failure)

### 7.4 Scaling Efficiency (Multi-GPU)

**Weak Scaling (BM_005 only):**

```
Efficiency_weak = T_1GPU / T_NGPU

Target: > 75% for N ≤ 4 GPUs
```

**Strong Scaling:**

```
Efficiency_strong = (T_1GPU / N) / T_NGPU

Target: > 65% for N ≤ 4 GPUs
```

---

## 8. Statistical Analysis

### 8.1 Sample Size Justification

**Why 5 runs?**

- Minimum for meaningful statistics (central limit theorem)
- Balances rigor with practicality (10 runs = 2 hours per benchmark)
- Sufficient to detect outliers (modified Z-score method)

**If variance is high:**

- Increase to 7-10 runs
- Investigate source of variance (thermal throttling? background processes?)

### 8.2 Outlier Detection

**Modified Z-Score Method:**

```python
def detect_outliers(times: list[float], threshold: float = 3.5) -> list[int]:
    """Detect outliers using modified Z-score."""
    median = np.median(times)
    mad = np.median(np.abs(times - median))  # Median absolute deviation
    modified_z = 0.6745 * (times - median) / mad
    return [i for i, z in enumerate(modified_z) if abs(z) > threshold]
```

**Action on Outliers:**

- **1 outlier:** Report but include in statistics
- **2+ outliers:** Investigate (hardware issue? thermal throttling?)
- **All outliers:** Discard entire benchmark run, repeat with different hardware

### 8.3 Confidence Intervals

**Bootstrap Method (for small samples):**

```python
def bootstrap_ci(times: list[float], n_bootstrap: int = 10000) -> tuple[float, float]:
    """Compute 95% confidence interval via bootstrap."""
    medians = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(times, size=len(times), replace=True)
        medians.append(np.median(sample))
    return np.percentile(medians, [2.5, 97.5])
```

**Reporting Format:**

```
Speedup: 4.2x (95% CI: [3.8x, 4.6x])
```

### 8.4 Statistical Significance Test

**Null Hypothesis:**

```
H₀: T_quasim ≥ T_ansys  (QuASIM is not faster)
H₁: T_quasim < T_ansys  (QuASIM is faster)
```

**Mann-Whitney U Test:**

```python
from scipy.stats import mannwhitneyu

def test_speedup_significance(ansys_times, quasim_times) -> tuple[float, str]:
    """Test if QuASIM is significantly faster than Ansys."""
    statistic, p_value = mannwhitneyu(quasim_times, ansys_times, alternative='less')
    
    if p_value < 0.05:
        return p_value, "SIGNIFICANT (p < 0.05)"
    else:
        return p_value, "NOT SIGNIFICANT"
```

---

## 9. Profiling and Diagnostics

### 9.1 Ansys Profiling

**Enable Ansys Timing Output:**

```apdl
/CONFIG,TIMINT,1  ! Enable timing
/STATUS,SOLU      ! Print solver status
```

**Parse Ansys Output:**

```python
def parse_ansys_timing(output_file: str) -> dict:
    """Extract timing breakdown from Ansys output."""
    with open(output_file) as f:
        content = f.read()
    
    # Look for timing sections
    # "Time spent in solution = 180.5 seconds"
    # "Time spent in contact detection = 42.3 seconds"
    # etc.
    
    timings = {}
    # TODO: Regex parsing of Ansys output
    return timings
```

### 9.2 QuASIM GPU Profiling

**NVIDIA Nsight Systems:**

```bash
# Profile QuASIM execution
nsys profile -o quasim_profile python3 performance_runner.py \
    --benchmark BM_001 --solver quasim

# View timeline
nsys-ui quasim_profile.nsys-rep
```

**NVIDIA Nsight Compute (kernel-level):**

```bash
# Profile specific kernels
ncu --set full -o kernel_profile python3 performance_runner.py \
    --benchmark BM_001 --solver quasim

# View kernel metrics
ncu-ui kernel_profile.ncu-rep
```

**What to Look For:**

- GPU utilization (should be > 80% during solve)
- Memory bandwidth utilization (> 60% is good)
- Kernel launch overhead (should be < 5% of total time)
- CPU-GPU transfer bottlenecks

---

## 10. Validation Criteria (Tier-0 Acceptance)

### 10.1 Per-Benchmark Acceptance

**For each benchmark (BM_001 through BM_005):**

| Criterion | Threshold | Action if Failed |
|-----------|-----------|------------------|
| **Accuracy** | | |
| Displacement error | < 2% | Tune solver parameters |
| Stress error | < 5% | Refine mesh, check material params |
| Energy conservation | < 1e-6 | Check Jacobian accuracy |
| **Performance** | | |
| Speedup | ≥ 3x | Profile, optimize hotspots |
| Iteration count | ≤ 1.2 × Ansys | Improve preconditioner |
| Memory usage | < GPU limit | Reduce precision, multi-GPU |
| **Stability** | | |
| Convergence failures | 0 | Reduce load step, enable line search |
| Reproducibility | 100% | Fix random seed, check determinism |

### 10.2 Overall Acceptance (All Benchmarks)

**Tier-0 acceptance requires:**

- [ ] All 5 benchmarks pass accuracy thresholds
- [ ] All 5 benchmarks achieve ≥ 3x speedup
- [ ] Zero convergence failures across all runs
- [ ] 100% reproducibility (identical hash for same seed)
- [ ] Statistical significance (p < 0.05 for all speedup claims)

**If any benchmark fails:**

1. Analyze failure mode (accuracy? performance? convergence?)
2. Implement fix (algorithm tuning, hardware upgrade, etc.)
3. Re-run failed benchmark(s)
4. Document failure and fix in report

---

## 11. Reporting Requirements

### 11.1 Report Formats

**CSV (machine-readable):**

```csv
Benchmark,Solver,Run,SolveTime,Iterations,MemoryGB,DisplacementError,StressError
BM_001,ansys,1,180.2,6,4.2,N/A,N/A
BM_001,ansys,2,182.1,6,4.2,N/A,N/A
...
BM_001,quasim,1,45.3,7,8.1,0.015,0.032
...
```

**JSON (detailed metrics):**

```json
{
  "benchmark": "BM_001",
  "solver": "quasim",
  "runs": [
    {
      "run_id": 1,
      "solve_time": 45.3,
      "iterations": 7,
      "memory_gb": 8.1,
      "convergence_history": [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.003],
      "state_hash": "a3f2c1..."
    }
  ],
  "accuracy": {
    "displacement_error": 0.015,
    "stress_error": 0.032,
    "energy_error": 5.2e-7
  },
  "performance": {
    "speedup": 4.2,
    "speedup_ci": [3.8, 4.6],
    "p_value": 0.008,
    "significance": "SIGNIFICANT"
  }
}
```

**HTML (human-readable):**

- Executive summary (pass/fail for each benchmark)
- Performance plots (speedup bar chart, convergence curves)
- Statistical analysis (confidence intervals, significance tests)
- Detailed tables (all runs, outliers flagged)

### 11.2 Required Plots

**Speedup Comparison (Bar Chart):**

```
Benchmark Speedup vs Ansys Baseline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BM_001 ████████████ 4.2x ± 0.4x
BM_002 ██████████   5.1x ± 0.6x
BM_003 ███████████  3.9x ± 0.3x
BM_004 ████████████████ 6.5x ± 0.8x
BM_005 █████████    3.7x ± 0.5x
        
       0x  2x  4x  6x  8x
```

**Accuracy Error Bars:**

```
Displacement Error (%) by Benchmark
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BM_001 ●━━━━━━━━━━━━━━━━━━━━━━━ 1.5%
BM_002 ●━━━━━━━━━━━━━━━━━━━━━━━ 1.8%
BM_003 ●━━━━━━━━━━━━━━━━━━━━━━━ 1.2%
BM_004 ●━━━━━━━━━━━━━━━━━━━━━━━ 2.0%  (at threshold)
BM_005 ●━━━━━━━━━━━━━━━━━━━━━━━ 1.6%

       0%   1%   2%   3%
       └─────┬─────┘
        Threshold (2%)
```

**Convergence History:**

```
Newton-Raphson Convergence (BM_001)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
10⁰  ●
     |\
10⁻¹ | \●
     |  \
10⁻² |   \●
     |    \
10⁻³ |     ●●●  Ansys (6 iter)
     |     
     └──────●●●  QuASIM (7 iter)
     
     1  2  3  4  5  6  7  8
     Iteration Number
```

---

## 12. Continuous Validation and Regression Testing

### 12.1 Regression Test Suite

**Automated Execution:**

- **Trigger:** Every merge to `main` branch
- **Frequency:** Weekly full benchmark suite
- **Hardware:** Dedicated CI server (fixed config)

**Regression Detection:**

```python
def check_performance_regression(
    current_speedup: float,
    baseline_speedup: float,
    tolerance: float = 0.05
) -> bool:
    """Check if performance has regressed."""
    if current_speedup < baseline_speedup * (1 - tolerance):
        return True  # Regression detected (>5% slower)
    return False
```

**Action on Regression:**

1. Bisect commits to find culprit
2. Profile regression case
3. Fix or revert
4. Update golden reference data

### 12.2 Notification and Reporting

**Success Notification:**

- Green checkmark on commit status
- Update performance badge in README

**Failure Notification:**

- Red X on commit status
- Email to engineering team
- Slack message to #quasim-ansys channel

**Failure Report Contents:**

- Which benchmark failed
- Accuracy or performance regression
- Comparison against previous run
- Profiling data (if available)

---

## 13. Partner Validation Protocol

### 13.1 Partner Execution

**Provided to Partners:**

- Benchmark definitions YAML
- Performance runner script
- Reference data (Ansys hashes, expected speedups)
- Installation instructions
- Troubleshooting guide

**Partner Responsibilities:**

- Run benchmarks on internal hardware
- Report results (CSV/JSON)
- Document any issues encountered

### 13.2 Partner Acceptance Gates

**Minimum Requirements:**

- 3+ Fortune-50 partners validate successfully
- No critical issues (zero convergence failures)
- Median speedup ≥ 3x across all partners

**Critical Issues:**

- Convergence failure (any benchmark)
- Accuracy violation (> 5% error)
- Crash or GPU driver error

**Non-Critical Issues:**

- Speedup < 3x (but > 2x, acceptable with investigation)
- Installation difficulties (docs improvement needed)

---

## 14. Troubleshooting Guide

### 14.1 Common Issues

**Issue:** Ansys solve time varies widely (> 20% between runs)

**Diagnosis:**

- Check CPU temperature (thermal throttling?)
- Check for background processes (`top`, `htop`)
- Check CPU frequency scaling (`cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`)

**Fix:**

- Set CPU governor to `performance`
- Kill background processes
- Increase cooldown period to 120 seconds

---

**Issue:** QuASIM GPU memory error (OOM)

**Diagnosis:**

- Check GPU memory usage (`nvidia-smi`)
- Check model size (element count)

**Fix:**

- Reduce precision (FP32 instead of FP64)
- Use multi-GPU mode
- Upgrade to 80GB GPU (from 40GB)

---

**Issue:** QuASIM convergence failure

**Diagnosis:**

- Check convergence history (diverging residual?)
- Check material parameters (negative modulus?)
- Check mesh quality (negative Jacobian?)

**Fix:**

- Reduce load step size (increase substeps)
- Enable line search
- Fallback to CPU (more robust numerics)

---

**Issue:** Accuracy failure (> 2% displacement error)

**Diagnosis:**

- Check mesh convergence (refine and re-run)
- Check Ansys reference hash (reproducible?)
- Check material parameter conversion (units correct?)

**Fix:**

- Refine mesh by 2x
- Re-run Ansys with tighter tolerance
- Verify material parameter mapping

---

## 15. Conclusion

This methodology ensures **reproducible, statistically rigorous, and adversarially defensible** performance validation of Ansys-QuASIM integration. By following this protocol, industrial partners can independently verify QuASIM performance claims and make informed decisions about pilot deployment.

**Key Takeaways:**

1. **Accuracy first** - Performance is irrelevant if accuracy fails
2. **Statistical rigor** - 5 runs, confidence intervals, significance tests
3. **Transparency** - Report all data, outliers, and failures
4. **Reproducibility** - SHA-256 hash verification, deterministic execution
5. **Adversarial mindset** - Assume skeptical reviewers

If all benchmarks pass Tier-0 acceptance criteria, QuASIM is **production-ready for pilot deployment** at Fortune-50 industrial partners.

---

**Document Control:**

- **Revision History:**
  - v1.0.0 (2025-12-13): Initial release
- **Next Review:** 2025-03-15 (quarterly)
