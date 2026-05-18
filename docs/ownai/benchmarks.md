# QuASIM-Own Benchmark Results

Generated: 2026-05-18T06:08:51.496888

Total runs: 25

## Summary by Task and Model

### tabular-cls

| Model | Primary Metric | Secondary Metric | Latency (ms) | Stability | Deterministic |
|-------|---------------|------------------|--------------|-----------|---------------|
| logreg | 0.6600 ± 0.0000 | 0.6610 ± 0.0000 | 0.07 | 1.000 | ✅ |
| mlp | 0.8760 ± 0.0153 | 0.8763 ± 0.0154 | 0.13 | 0.983 | ❌ |
| rf | 0.7850 ± 0.0138 | 0.7850 ± 0.0140 | 3.18 | 0.982 | ❌ |
| slt | 0.7780 ± 0.0186 | 0.7779 ± 0.0185 | 5.51 | 0.976 | ❌ |

### text-cls

| Model | Primary Metric | Secondary Metric | Latency (ms) | Stability | Deterministic |
|-------|---------------|------------------|--------------|-----------|---------------|
| slt | 0.9910 ± 0.0020 | 0.9910 ± 0.0020 | 13.69 | 0.998 | ❌ |

## Reliability-per-Watt Ranking

Computed as: `(stability × primary_metric) / energy_proxy`

| Rank | Task | Model | Reliability-per-Watt |
|------|------|-------|---------------------|
| 1 | tabular-cls | logreg | 150.081603 |
| 2 | tabular-cls | mlp | 105.843929 |
| 3 | tabular-cls | rf | 3.726213 |
| 4 | tabular-cls | slt | 2.119924 |
| 5 | text-cls | slt | 1.111021 |
