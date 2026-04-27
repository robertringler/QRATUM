# GPU_ENVIRONMENT_OPTIONS.md - Execution Environment Discovery

**Document Type**: GPU Environment Assessment  
**Generated**: 2026-01-18 20:45:00 UTC  
**Git Commit**: d6bafe894be45d50e63a36241e536bfe70d2f6b3  
**Status**: Discovery Phase (No Benchmarking)  

---

## 1. CURRENT ENVIRONMENT ASSESSMENT

### GitHub Actions Standard Runner (Current)

| Property | Value |
|----------|-------|
| Runner | `ubuntu-latest` |
| GPU | ❌ None |
| CPU | AMD EPYC 7763 (4 vCPUs) |
| RAM | 16 GB |
| Storage | 72 GB SSD |

**Verdict**: CPU-only benchmarks viable. GPU benchmarks impossible.

---

## 2. GPU EXECUTION ENVIRONMENT OPTIONS

### Option A: GitHub Actions GPU Runners

#### A.1: GitHub-Hosted Larger Runners (Enterprise)

| Property | Value |
|----------|-------|
| Availability | GitHub Enterprise Cloud only |
| GPU Models | T4, A10G (not A100) |
| VRAM | 16-24 GB |
| Cost | ~$0.07/min (T4), ~$0.15/min (A10G) |
| Setup | Requires Enterprise subscription |

**Reproducibility**:

- ✅ Hardware identifiable via runner labels
- ⚠️ Shared infrastructure, potential variability
- ✅ Logs preserved in GitHub

**Suitability**: 🟡 Partial - T4/A10G lack SM 8.0 for full VITRA support

#### A.2: Self-Hosted GPU Runner

| Property | Value |
|----------|-------|
| Availability | User-provisioned |
| GPU Models | Any (user choice) |
| Cost | Infrastructure cost only |
| Setup | Requires runner registration |

**Reproducibility**:

- ✅ Full hardware control
- ✅ Deterministic environment
- ✅ All logs preserved

**Suitability**: ✅ Ideal if A100 hardware available

---

### Option B: Major Cloud Providers

#### B.1: AWS EC2 GPU Instances

| Instance | GPU | VRAM | Cost/hr | Spot Cost |
|----------|-----|------|---------|-----------|
| p4d.24xlarge | 8× A100 40GB | 320 GB | ~$32.77 | ~$10-15 |
| p4de.24xlarge | 8× A100 80GB | 640 GB | ~$40.97 | ~$12-18 |
| p3.2xlarge | 1× V100 | 16 GB | ~$3.06 | ~$0.92 |
| g4dn.xlarge | 1× T4 | 16 GB | ~$0.526 | ~$0.16 |
| g5.xlarge | 1× A10G | 24 GB | ~$1.006 | ~$0.30 |

**Reproducibility Assessment**:

- ✅ Hardware unambiguously identified via instance metadata
- ✅ Runs repeatable with same instance type
- ✅ CloudWatch logs preserved
- ⚠️ Spot instances may be terminated (use On-Demand for benchmarks)

**Suitability for QRATUM**:

- p4d/p4de: ✅ Full support (A100)
- g5 (A10G): 🟡 Partial (SM 8.6, not A100)
- g4dn (T4): ⚠️ Limited (SM 7.5)
- p3 (V100): ⚠️ Legacy support only

#### B.2: Google Cloud Platform (GCP)

| Machine Type | GPU | VRAM | Cost/hr |
|--------------|-----|------|---------|
| a2-highgpu-1g | 1× A100 40GB | 40 GB | ~$3.67 |
| a2-ultragpu-1g | 1× A100 80GB | 80 GB | ~$5.01 |
| a2-megagpu-16g | 16× A100 40GB | 640 GB | ~$58.72 |
| n1 + T4 | 1× T4 | 16 GB | ~$0.35 |

**Reproducibility Assessment**:

- ✅ Hardware identifiable via metadata server
- ✅ Deterministic with same machine type
- ✅ Cloud Logging preserved
- ✅ Preemptible instances ~70% discount

**Suitability for QRATUM**: ✅ a2-highgpu-1g ideal for single-GPU benchmarks

#### B.3: Microsoft Azure

| VM Size | GPU | VRAM | Cost/hr |
|---------|-----|------|---------|
| Standard_NC24ads_A100_v4 | 1× A100 80GB | 80 GB | ~$3.67 |
| Standard_ND96asr_v4 | 8× A100 40GB | 320 GB | ~$27.20 |
| Standard_NC6s_v3 | 1× V100 | 16 GB | ~$3.06 |
| Standard_NC4as_T4_v3 | 1× T4 | 16 GB | ~$0.526 |

**Reproducibility Assessment**:

- ✅ Hardware identifiable via Azure IMDS
- ✅ Deterministic with same VM size
- ✅ Azure Monitor logs preserved

**Suitability for QRATUM**: ✅ NC24ads_A100_v4 ideal

---

### Option C: Academic/Research Free-Tier GPU

#### C.1: Google Colab

| Property | Value |
|----------|-------|
| GPU Models | T4 (free), A100 (Colab Pro+) |
| VRAM | 16 GB (T4), 40 GB (A100) |
| Cost | Free / $49.99/mo (Pro+) |
| Time Limit | ~12 hours |

**Reproducibility Assessment**:

- ⚠️ Hardware NOT guaranteed (may get different GPU)
- ❌ Runs NOT repeatable (random GPU assignment)
- ⚠️ Logs lost on disconnect
- ❌ Throttling unpredictable

**Suitability**: ❌ **DISQUALIFIED** - Hardware variability invalidates claims

#### C.2: Kaggle Notebooks

| Property | Value |
|----------|-------|
| GPU Models | T4 (dual), P100 |
| VRAM | 16 GB (T4), 16 GB (P100) |
| Cost | Free (30 hrs/week) |
| Time Limit | 12 hours per session |

**Reproducibility Assessment**:

- ⚠️ Hardware selection not guaranteed
- ⚠️ Shared environment
- ✅ Notebooks saved

**Suitability**: ❌ **DISQUALIFIED** - Cannot guarantee consistent hardware

#### C.3: Lambda Labs Cloud

| Property | Value |
|----------|-------|
| GPU Models | A100 40GB, A100 80GB, H100 |
| Cost | $1.10/hr (A100 40GB), $1.99/hr (H100) |
| Availability | Often limited |

**Reproducibility Assessment**:

- ✅ Hardware clearly specified
- ✅ Dedicated instances
- ⚠️ Availability constraints

**Suitability**: ✅ Cost-effective A100 access

#### C.4: Paperspace

| Property | Value |
|----------|-------|
| GPU Models | A100, V100, RTX |
| Cost | $2.30/hr (A100) |

**Reproducibility Assessment**:

- ✅ Dedicated hardware
- ✅ Persistent storage

**Suitability**: ✅ Viable alternative

---

### Option D: Local Workstation

| Property | Value |
|----------|-------|
| GPU | User-dependent |
| Cost | Existing hardware |
| Availability | Immediate |

**Reproducibility Assessment**:

- ✅ Full hardware identification (nvidia-smi)
- ✅ Complete control over environment
- ✅ All logs preserved locally
- ⚠️ Results specific to hardware

**Suitability**: ✅ Ideal for development; requires A100 for production claims

---

## 3. AUDITABILITY ASSESSMENT MATRIX

| Environment | HW Identifiable | Repeatable | Logs Preserved | Defensible | Verdict |
|-------------|-----------------|------------|----------------|------------|---------|
| GitHub GPU (Enterprise) | ✅ | ✅ | ✅ | 🟡 | Partial |
| Self-Hosted Runner | ✅ | ✅ | ✅ | ✅ | **Ideal** |
| AWS p4d (A100) | ✅ | ✅ | ✅ | ✅ | **Ideal** |
| AWS g5 (A10G) | ✅ | ✅ | ✅ | 🟡 | Partial |
| GCP a2-highgpu | ✅ | ✅ | ✅ | ✅ | **Ideal** |
| Azure NC24ads_A100 | ✅ | ✅ | ✅ | ✅ | **Ideal** |
| Google Colab | ❌ | ❌ | ⚠️ | ❌ | **Disqualified** |
| Kaggle | ❌ | ❌ | ⚠️ | ❌ | **Disqualified** |
| Lambda Labs | ✅ | ✅ | ✅ | ✅ | **Viable** |
| Local (with A100) | ✅ | ✅ | ✅ | ✅ | **Ideal** |

---

## 4. DISQUALIFIED ENVIRONMENTS

### Google Colab

**Reason**: Hardware selection is non-deterministic. Performance claims would be invalid because:

1. GPU model varies between sessions
2. Throttling applied unpredictably
3. Shared resources cause variability

### Kaggle Notebooks

**Reason**: Similar to Colab - hardware not guaranteed. Benchmark results could vary 2-5x between runs due to:

1. Different GPU assignments
2. Background resource contention
3. Session time limits

---

## 5. ENVIRONMENT RANKING (Most to Least Defensible)

### Tier 1: Fully Defensible (A100 Hardware)

1. **Self-Hosted Runner with A100** - Complete control
2. **AWS p4d.24xlarge (On-Demand)** - Documented, repeatable
3. **GCP a2-highgpu-1g** - Cost-effective A100
4. **Azure NC24ads_A100_v4** - Enterprise-friendly
5. **Lambda Labs A100** - Budget option

### Tier 2: Partially Defensible (Non-A100)

6. **AWS g5.xlarge (A10G)** - SM 8.6, limited VRAM
2. **GitHub GPU Runners (T4/A10G)** - Convenient but limited

### Tier 3: Development Only

8. **Local RTX 3090/4090** - Consumer hardware
2. **AWS p3 (V100)** - Legacy

### Tier 4: Disqualified

- Google Colab
- Kaggle Notebooks
- Any shared/variable hardware

---

## 6. PRIMARY RECOMMENDATION

### Recommended Environment: **GCP a2-highgpu-1g**

**Technical Justification**:

1. **Hardware**: NVIDIA A100 40GB (SM 8.0) - meets all QRATUM requirements
2. **Cost**: ~$3.67/hr - affordable for benchmark runs
3. **Reproducibility**:
   - Same hardware guaranteed via machine type
   - Metadata server provides hardware attestation
   - Cloud Logging preserves all output
4. **Auditability**:
   - Instance metadata documents exact hardware
   - Timestamped logs in Cloud Logging
   - Boot disk can be preserved as snapshot
5. **Setup Complexity**: Low - standard GCP VM workflow

### Alternative: **AWS p4d.24xlarge (Spot)**

- **Use Case**: Multi-GPU benchmarks
- **Cost**: ~$10-15/hr (Spot)
- **Risk**: Spot termination possible

---

## 7. VALID AND INVALID PERFORMANCE CLAIMS

### In GCP a2-highgpu-1g Environment

#### VALID Claims

- "QRATUM achieved X throughput on NVIDIA A100 40GB"
- "Quantum simulation of 25 qubits completed in Y seconds"
- "VITRA-E0 processed 30x WGS in Z minutes"
- "Chess engine evaluated N nodes/second with GPU acceleration"

#### INVALID Claims

- "QRATUM is X% faster than competitor" (no competitor baseline)
- "Will achieve same performance on H100" (extrapolation)
- "Scalable to 8-GPU" (single GPU tested)
- "Production-ready performance" (benchmark environment only)

### Performance Claim Template

```
BENCHMARK: [Name]
HARDWARE: NVIDIA A100-SXM4-40GB
CUDA: 12.4.1
DRIVER: 535.x
INSTANCE: GCP a2-highgpu-1g
RESULT: [Metric] = [Value] ± [Variance]
RUNS: [N] repetitions
DETERMINISM: [Hash if applicable]
```

---

## 8. NEXT PR REQUIRED TO ENABLE GPU BENCHMARKING

### Required CI/Infra Changes

#### Option A: GCP Integration (Recommended)

1. **Create GCP Service Account**
   - Role: `compute.instanceAdmin.v1`
   - Role: `logging.logWriter`

2. **Add GitHub Secrets**

   ```
   GCP_PROJECT_ID: qratum-benchmarks
   GCP_SA_KEY: [base64-encoded service account JSON]
   GCP_ZONE: us-central1-a
   ```

3. **Create Workflow File**: `.github/workflows/gpu-benchmark.yml`

   ```yaml
   name: GPU Benchmarks
   on:
     workflow_dispatch:
     schedule:
       - cron: '0 0 * * 0'  # Weekly
   
   jobs:
     gpu-benchmark:
       runs-on: ubuntu-latest
       steps:
         - uses: google-github-actions/auth@v2
           with:
             credentials_json: ${{ secrets.GCP_SA_KEY }}
         
         - name: Create GPU VM
           run: |
             gcloud compute instances create benchmark-runner \
               --machine-type=a2-highgpu-1g \
               --zone=${{ secrets.GCP_ZONE }} \
               --image-family=pytorch-latest-gpu \
               --image-project=deeplearning-platform-release \
               --maintenance-policy=TERMINATE
         
         - name: Run Benchmarks
           run: |
             gcloud compute ssh benchmark-runner --command="..."
         
         - name: Destroy VM
           if: always()
           run: |
             gcloud compute instances delete benchmark-runner --quiet
   ```

4. **Update `.gitignore`**

   ```
   # GCP credentials (never commit)
   *.json
   !package*.json
   ```

#### Expected Risks

| Risk | Mitigation |
|------|------------|
| Cloud cost overrun | Set budget alerts, use preemptible |
| Secret exposure | Use GitHub encrypted secrets |
| VM not destroyed | Add `if: always()` cleanup |
| GPU quota limits | Request quota increase |

#### Credentials Needed

- GCP Service Account JSON key
- Project ID with billing enabled
- Sufficient GPU quota in target zone

---

## 9. IMMEDIATE NEXT STEPS

1. **No Action Required Now** - CPU benchmarks are valid
2. **For GPU Benchmarks**:
   - Decision: Choose GCP, AWS, or self-hosted
   - Action: Provision credentials and update CI
   - Timeline: Separate PR after this discovery phase

---

**END OF GPU ENVIRONMENT OPTIONS DOCUMENT**
