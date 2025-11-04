# QuASIM Benchmark Report

## Environment

- **Commit**: `455b6295`
- **Branch**: `copilot/execute-quasim-benchmark-suite-again`
- **Dirty**: True

- **OS**: Linux
- **Python**: 3.12.3


## Summary

- **Total Kernels**: 3
- **Successful**: 3
- **Failed**: 0

## Performance Leaderboard

| Kernel | Backend | Precision | p50 (ms) | p90 (ms) | Throughput (ops/s) |
| --- | --- | --- | --- | --- | --- |
| autonomous_systems | jax | fp32 | 0.011 | 0.015 | 77186.70 |
| quasim_runtime | cpu | fp32 | 37.667 | 37.812 | 26.53 |
| pressure_poisson | cuda | fp32 | 44.207 | 44.719 | 22.60 |

## Resource Usage

No memory data available.

## Key Findings

- **Fastest Kernel**: `autonomous_systems` (0.011 ms p50)
- **Highest Throughput**: `autonomous_systems` (77186.70 ops/s)
- **Backends Tested**: cpu, cuda, jax

## Recommendations

- **High Variance Detected**: 1 kernel(s) show >10% stddev. Consider investigating sources of non-determinism.
- **Precision Testing**: Consider testing additional precisions (FP16, FP8) for speed vs. accuracy trade-offs.
