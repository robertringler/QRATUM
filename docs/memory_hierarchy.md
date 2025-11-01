# Memory Hierarchy

| Level | Size | Assoc | Latency (ns) | Bandwidth |
|-------|------|-------|--------------|-----------|
| L1I/L1D | 128 KB per core | 4-way | 1.2 | 6.4 TB/s aggregate |
| L2 | 4 MB slice (per 4 cores) | 8-way | 6.5 | 1.8 TB/s aggregate |
| L3 | 96 MB shared | 16-way | 18.0 | 0.9 TB/s |
| GPU SM | 256 KB register file + 128 KB shared mem | -- | 0.8 | 14.0 TB/s |
| HBM/LPDDR5x | 128 GB | -- | 78.0 | 273 GB/s |

The coherent NVLink-C2C fabric exposes a unified memory model. CPU and GPU agents participate in directory-based coherence using a MOESI variant. The memory controller employs adaptive write-combining and command reordering to maximize LPDDR5x bandwidth while preserving QoS for latency-sensitive tensor kernels.

Tensor workloads benefit from explicit residency hints communicated through the `gb10_mm` driver, which maps large pages and shares residency metadata with the runtime scheduler.
