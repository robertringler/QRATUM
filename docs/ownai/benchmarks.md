# QuASIM-Own Benchmark Results

Generated: 2026-01-11T03:46:31.919394

Total runs: 25

## Summary by Task and Model

### tabular-cls

| Model | Primary Metric | Secondary Metric | Latency (ms) | Stability | Deterministic |
|-------|---------------|------------------|--------------|-----------|---------------|
| logreg | 0.6600 ± 0.0000 | 0.6610 ± 0.0000 | 0.09 | 1.000 | ✅ |
| mlp | 0.8770 ± 0.0160 | 0.8773 ± 0.0161 | 0.15 | 0.982 | ❌ |
| rf | 0.7850 ± 0.0138 | 0.7850 ± 0.0140 | 3.62 | 0.982 | ❌ |
| slt | 0.7780 ± 0.0186 | 0.7779 ± 0.0185 | 5.74 | 0.976 | ❌ |

### text-cls

| Model | Primary Metric | Secondary Metric | Latency (ms) | Stability | Deterministic |
|-------|---------------|------------------|--------------|-----------|---------------|
| slt | 0.9910 ± 0.0020 | 0.9910 ± 0.0020 | 14.06 | 0.998 | ❌ |

## Reliability-per-Watt Ranking

Computed as: `(stability × primary_metric) / energy_proxy`

| Rank | Task | Model | Reliability-per-Watt |
|------|------|-------|---------------------|
| 1 | tabular-cls | logreg | 114.708874 |
| 2 | tabular-cls | mlp | 86.116542 |
| 3 | tabular-cls | rf | 3.281507 |
| 4 | tabular-cls | slt | 2.035947 |
| 5 | text-cls | slt | 1.081949 |
