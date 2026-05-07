# P0.6 — Divergence Policy

A run is *divergent* iff the observed envelope exceeds the declared
bounds in any single dimension.

## Default bounds (`Envelope::defaults`)

| Dimension          | Bound (cycles) |
|--------------------|----------------|
| irq_jitter_cyc     | 10 000         |
| apic_latency_cyc   | 50 000         |
| replay_drift_cyc   | 1 000          |
| queue_pressure     | 64             |

These are **policy** values, not measured ceilings.

## Action on divergence

INV-Y trips the `DivergenceEnvelopeBreached` invariant and emits an
audit frame. The kernel halts in release; debug panics with the
breached dimensions.

## Tightening

Bounds may only be tightened, never widened, without a recorded
freeze amendment under `ARCHITECTURE_FREEZE.md`.
