# GPU_REQUIREMENTS.md - QRATUM GPU Infrastructure Requirements

**Document Type**: GPU Requirements Specification  
**Generated**: 2026-01-18 20:45:00 UTC  
**Git Commit**: d6bafe894be45d50e63a36241e536bfe70d2f6b3  
**Status**: Discovery Phase (No Benchmarking)  

---

## 1. EXECUTIVE SUMMARY

QRATUM is a multi-subsystem platform with **heterogeneous GPU requirements**. Different subsystems have varying GPU needs, from none (CPU-only) to high-end datacenter GPUs. This document derives requirements from codebase inspection.

---

## 2. SUPPORTED GPU BACKENDS

| Backend | Status | Implementation | Files |
|---------|--------|----------------|-------|
| **NVIDIA CUDA** | ✅ Primary | Production-ready | `QuASIM/src/cuda/*.cu`, `quasim/qc/simulator.py` |
| **AMD ROCm/HIP** | 🟡 Stub | Placeholder only | `quasim/hcal/backends/amd_rocm.py` |
| **CPU Fallback** | ✅ Complete | Full functionality | All modules |
| **Apple Metal** | ❌ Not Supported | N/A | N/A |

### CUDA Implementation Evidence
```
QuASIM/src/cuda/tensor_solve.cu
QuASIM/src/cuda/vjp.cu
QuASIM/src/cuda/ftq_kernels.cu
```

### Qiskit Aer GPU Backend
```python
# From quasim/qc/simulator.py
AerSimulator(method='statevector', device='GPU')
```

---

## 3. CUDA VERSION REQUIREMENTS

### Minimum Requirements (from Dockerfile.cuda)
| Dependency | Version | Source |
|------------|---------|--------|
| CUDA Toolkit | 12.4.1 | `nvidia/cuda:12.4.1-devel-ubuntu22.04` |
| Driver | 535.x+ | Implied by CUDA 12.4 |
| Ubuntu | 22.04 | Dockerfile base |
| Python | 3.x | System default |
| pybind11 | 2.11.1 | Requirements |
| NumPy | 1.26.4 | CUDA-compatible version |

### Compute Capability
| Source | Required SM Version | Notes |
|--------|---------------------|-------|
| `quasim/cli/main.py` | SM 8.0 (Ampere) | Explicit config |
| SAXPY kernel | Any | Basic CUDA |
| DeepVariant (VITRA) | SM 8.0+ | A100 optimized |

**Minimum Compute Capability: SM 8.0 (Ampere architecture)**

---

## 4. SUBSYSTEM-SPECIFIC REQUIREMENTS

### 4.1 QuASIM Tensor Simulation
- **GPU Required**: Optional
- **Benefit**: Faster tensor contractions
- **Minimum**: Any CUDA-capable GPU
- **Recommended**: A100 40GB

### 4.2 QRATUM-Chess (BOB Engine)
- **GPU Required**: Optional
- **Benefit**: Neural network inference acceleration
- **Minimum**: 4GB VRAM
- **Recommended**: RTX 3080 or better

### 4.3 Quantum Circuit Simulation (Qiskit Aer)
- **GPU Required**: Optional
- **Benefit**: 10-100x speedup for >20 qubits
- **Minimum**: 8GB VRAM
- **Recommended**: A100 40GB for 30+ qubit simulation

### 4.4 VITRA-E0 Genomics Pipeline
- **GPU Required**: Strongly Recommended
- **Benefit**: 45min vs 8+ hours for 30x WGS
- **Minimum**: A100 40GB (Parabricks requirement)
- **Recommended**: A100 80GB
- **CUDA Version**: 12.4.x + Driver 535.x (locked for determinism)

### 4.5 ANSYS/Elastomer Benchmarks
- **GPU Required**: Optional (CPU reference available)
- **Minimum**: A100 40GB
- **Recommended**: A100 80GB
- **Memory**: 16-24GB GPU VRAM

---

## 5. MEMORY REQUIREMENTS

| Workload | GPU VRAM | System RAM |
|----------|----------|------------|
| Chess Engine | 4-8 GB | 16 GB |
| Quantum Sim (20 qubits) | 8 GB | 32 GB |
| Quantum Sim (30 qubits) | 40 GB | 128 GB |
| Genomics (30x WGS) | 40-80 GB | 64-128 GB |
| ANSYS Elastomer | 16-24 GB | 64 GB |

---

## 6. DRIVER AND TOOLKIT DEPENDENCIES

### Required Software Stack
```
┌─────────────────────────────────────┐
│ Application Layer (Python 3.10+)    │
├─────────────────────────────────────┤
│ Qiskit Aer 0.13.0+                  │
│ PyTorch 2.1+ (optional)             │
│ cuQuantum (optional)                │
├─────────────────────────────────────┤
│ CUDA Toolkit 12.4.x                 │
├─────────────────────────────────────┤
│ NVIDIA Driver 535.x+               │
├─────────────────────────────────────┤
│ Linux Kernel 5.15+                  │
└─────────────────────────────────────┘
```

### Version Pinning for Determinism
VITRA-E0 requires exact version pinning:
- CUDA 12.4.x (not 12.5 or 11.x)
- Driver 535.x specifically
- This ensures PTX kernel reproducibility

---

## 7. HARD DEPENDENCIES

### Mandatory for GPU Execution
1. NVIDIA GPU with SM 8.0+ (Ampere or newer)
2. CUDA Toolkit 12.4.x
3. Compatible NVIDIA driver
4. pybind11 2.11.1
5. NumPy 1.26.4 (CUDA-compatible)

### Optional Enhancements
1. cuQuantum SDK (quantum acceleration)
2. NCCL (multi-GPU communication)
3. NVIDIA Parabricks (genomics)

---

## 8. SUPPORTED GPU MODELS

### Tier 1: Recommended (Full Support)
| Model | VRAM | SM | Use Cases |
|-------|------|-----|-----------|
| NVIDIA A100 80GB | 80 GB | 8.0 | All workloads |
| NVIDIA A100 40GB | 40 GB | 8.0 | Most workloads |
| NVIDIA H100 | 80 GB | 9.0 | All workloads |

### Tier 2: Supported (Limited Workloads)
| Model | VRAM | SM | Use Cases |
|-------|------|-----|-----------|
| NVIDIA A10 | 24 GB | 8.6 | Chess, small quantum |
| NVIDIA T4 | 16 GB | 7.5 | Chess, inference only |
| NVIDIA V100 | 32 GB | 7.0 | Legacy support |

### Tier 3: Consumer (Development Only)
| Model | VRAM | SM | Use Cases |
|-------|------|-----|-----------|
| RTX 4090 | 24 GB | 8.9 | Development |
| RTX 3090 | 24 GB | 8.6 | Development |
| RTX 3080 | 10 GB | 8.6 | Chess only |

---

## 9. CPU FALLBACK CAPABILITIES

All QRATUM subsystems support CPU-only execution:

| Subsystem | CPU Support | Performance Impact |
|-----------|-------------|-------------------|
| QuASIM Tensor | ✅ Full | 10-100x slower |
| QRATUM-Chess | ✅ Full | 2-5x slower |
| Quantum Sim | ✅ Full | 10-1000x slower |
| VITRA-E0 | ✅ Partial | GPU required for production |
| UltraSSSP | ✅ Full | No GPU benefit |
| PostDijkstra | ✅ Full | No GPU benefit |

---

## 10. DETERMINISM REQUIREMENTS

For audit-compliant benchmarking with GPU:

1. **Fixed CUDA Epoch**: Lock to CUDA 12.4.x
2. **PTX Verification**: Hash compiled kernels
3. **Driver Pinning**: Use driver 535.x
4. **Seed Management**: Fixed seeds (42)
5. **Environment Variables**:
   ```bash
   CUDA_VISIBLE_DEVICES=0
   CUBLAS_WORKSPACE_CONFIG=:4096:8
   ```

---

## 11. BENCHMARK-SPECIFIC GPU REQUIREMENTS

### BM-001: QuASIM Tensor
- **Minimum**: CPU (functional)
- **Recommended**: Any CUDA GPU
- **Optimal**: A100

### BM-002: UltraSSSP
- **GPU**: Not beneficial
- **CPU-only**: Recommended

### BM-003: PostDijkstra
- **GPU**: Not beneficial
- **CPU-only**: Recommended

### BM-004: Automated Suite
- **GPU**: Optional
- **Backend Config**: `--backend cpu` or `--backend cuda`

### BM-005/BM-006: QRATUM-Chess
- **Minimum**: CPU (functional)
- **Recommended**: 8GB+ GPU
- **Neural Network**: Benefits from GPU

---

## 12. CONCLUSION

### Minimum Viable GPU Configuration
- **Model**: NVIDIA A100 40GB
- **CUDA**: 12.4.x
- **Driver**: 535.x
- **Compute Capability**: SM 8.0

### Optimal Configuration
- **Model**: NVIDIA A100 80GB or H100
- **CUDA**: 12.4.x (pinned)
- **Multi-GPU**: NVLink for large quantum simulations

### CPU-Only Viability
All core benchmarks can execute on CPU. GPU is:
- **Required**: Only for VITRA-E0 production genomics
- **Recommended**: For quantum simulation >20 qubits
- **Optional**: For all other workloads

---

**END OF GPU REQUIREMENTS DOCUMENT**
