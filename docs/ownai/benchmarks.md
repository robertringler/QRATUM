# QuASIM-Own Benchmark Results

Generated: 2026-01-10T03:21:09.962127

Total runs: 25

## Summary by Task and Model

### tabular-cls

| Model | Primary Metric | Secondary Metric | Latency (ms) | Stability | Deterministic |
|-------|---------------|------------------|--------------|-----------|---------------|
| logreg | 0.6600 ± 0.0000 | 0.6610 ± 0.0000 | 0.10 | 1.000 | ✅ |
| mlp | 0.8760 ± 0.0153 | 0.8763 ± 0.0154 | 0.17 | 0.983 | ❌ |
| rf | 0.7850 ± 0.0138 | 0.7850 ± 0.0140 | 4.42 | 0.982 | ❌ |
| slt | 0.7780 ± 0.0186 | 0.7779 ± 0.0185 | 7.16 | 0.976 | ❌ |

### text-cls

| Model | Primary Metric | Secondary Metric | Latency (ms) | Stability | Deterministic |
|-------|---------------|------------------|--------------|-----------|---------------|
| slt | 0.9910 ± 0.0020 | 0.9910 ± 0.0020 | 15.29 | 0.998 | ❌ |

## Reliability-per-Watt Ranking

Computed as: `(stability × primary_metric) / energy_proxy`

| Rank | Task | Model | Reliability-per-Watt |
|------|------|-------|---------------------|
| 1 | tabular-cls | logreg | 100.801000 |
| 2 | tabular-cls | mlp | 79.957777 |
| 3 | tabular-cls | rf | 2.683938 |
| 4 | tabular-cls | slt | 1.631281 |
| 5 | text-cls | slt | 0.995170 |
