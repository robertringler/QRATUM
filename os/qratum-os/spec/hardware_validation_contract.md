# P0.6 — Hardware Validation Contract

## Scope

This document distinguishes claims that have been **measured on real
hardware**, claims that are **simulated/inferred**, and claims that
are **deferred** until SMP-real bring-up.

## Categories

| Claim                                       | Status     | Mechanism                              |
|---------------------------------------------|------------|----------------------------------------|
| Single-core boot (UEFI)                     | measured   | QEMU + bare metal smoke (P0.4)         |
| AP startup receipt emission                 | inferred   | Host model + kernel stub               |
| LAPIC/x2APIC equivalence                    | inferred   | Pure-fn equivalence proof              |
| Work-steal determinism                      | measured   | `deterministic_work_steal` host tests  |
| Receipt Merkle root reproducibility         | measured   | `receipt_merkle_integrity`             |
| Divergence envelope detection               | measured   | `divergence_envelope_tests`            |
| Real INIT/SIPI on real silicon              | deferred   | gated by `cfg(target_feature="smp_real")` |
| TSC drift bound on heterogeneous packages   | deferred   | requires multi-socket lab              |

## Forbidden claims

This phase MUST NOT claim hardware-jitter measurements taken on real
silicon. The `divergence_envelope` defaults are *policy bounds*, not
measured ceilings.
