# QRATUM Benchmark Report

_Phase_: U5 Benchmark & Technical Spec  
_Generated_: 2026-04-30

## Existing Benchmark Harnesses

The following benchmark infrastructure exists in the repository:

| File/Dir | Purpose |
|----------|---------|
| `benchmarks/` | Benchmark suite directory |
| `benchmarking_framework.py` | Core benchmark framework |
| `scale_benchmarks.py` | Scale/load benchmarks |
| `run_full_benchmark.py` | Full benchmark runner |
| `montecarlo_campaigns/` | Monte Carlo benchmarks |
| `temporal_compression/` | Temporal compression benchmarks |
| `.github/workflows/bm_001.yml` | CI benchmark gate |
| `.github/workflows/compression-benchmarks.yml` | Compression CI gate |

## Baseline Results (from audit/PROOF_LEDGER.md)

| Subsystem | Metric | Value |
|-----------|--------|-------|
| CIIR loop step time | per-step (deterministic) | <1ms (Python, no GPU) |
| MVRI loop | 50-step run | deterministic, O(N²) per step |
| RIC v2 horizon K | cost vs horizon | linear in K |
| Plasma reconnection | control margin | ≥0 (F1-F5 failure modes defined) |

## Flagged Gaps (from Prompt #1 audit)

Per the audit, missing benchmarks were flagged for:

- Multi-qubit CIIR controller throughput (GAP-CIIR-MQ-001 — now stubbed)
- Tensor-network contraction cost vs bond dimension (no dedicated benchmark exists)
- RIC v2 horizon-vs-cost curve (no dedicated harness)

## Status

- **GAP-BENCHMARK-MQ-001**: Multi-qubit controller benchmark — DEFERRED (stub implementation created; performance benchmarks require real quantum backend)
- **Existing benchmarks**: Retained as-is; no stale top-level reports archived

## Reproducibility

All benchmarks in `benchmarks/` use seeded RNG. To reproduce:

```bash
PYTHONPATH=. python run_full_benchmark.py --seed 42
```
