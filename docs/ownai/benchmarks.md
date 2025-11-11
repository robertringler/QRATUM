# QuASIM-Own Benchmark Results

Generated: 2025-11-11T03:11:17.776394

Total runs: 25

## Summary by Task and Model

### tabular-cls

| Model | Primary Metric | Secondary Metric | Latency (ms) | Stability | Deterministic |
|-------|---------------|------------------|--------------|-----------|---------------|
| logreg | 0.6600 ± 0.0000 | 0.6610 ± 0.0000 | 0.10 | 1.000 | ✅ |
| mlp | 0.8770 ± 0.0160 | 0.8773 ± 0.0161 | 0.17 | 0.982 | ❌ |
| rf | 0.7850 ± 0.0138 | 0.7850 ± 0.0140 | 4.60 | 0.982 | ❌ |
| slt | 0.7780 ± 0.0186 | 0.7779 ± 0.0185 | 7.42 | 0.976 | ❌ |

### text-cls

| Model | Primary Metric | Secondary Metric | Latency (ms) | Stability | Deterministic |
|-------|---------------|------------------|--------------|-----------|---------------|
| slt | 0.9910 ± 0.0020 | 0.9910 ± 0.0020 | 15.66 | 0.998 | ❌ |

## Reliability-per-Watt Ranking

Computed as: `(stability × primary_metric) / energy_proxy`

| Rank | Task | Model | Reliability-per-Watt |
|------|------|-------|---------------------|
| 1 | tabular-cls | logreg | 100.584417 |
| 2 | tabular-cls | mlp | 78.022781 |
| 3 | tabular-cls | rf | 2.577607 |
| 4 | tabular-cls | slt | 1.573837 |
| 5 | text-cls | slt | 0.971876 |
