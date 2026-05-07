# Formal Model Limits

P0.5 formal models are intentionally bounded. This document records
their proof envelope and what they do **not** establish.

## Bounded constants

| Model                              | Constant   | Bound used in CI |
|------------------------------------|------------|------------------|
| `canonical_equivalence_boundary`   | `MaxLen`   | 64               |
| `canonical_equivalence_boundary`   | `Inputs`   | 8 atoms          |
| `migration_consistency`            | `MaxLen`   | 64               |
| `migration_consistency`            | `Tasks`    | 8                |
| `migration_consistency`            | `Cores`    | 4                |
| `multicore_replay`                 | `Cores`    | 4                |
| `multicore_replay`                 | `Tasks`    | 8                |
| `multicore_replay`                 | `MaxLen`   | 16               |

## State-space pruning

- Symmetry reduction over `Cores` and `Tasks` is enabled via TLC
  `SYMMETRY` declarations in the corresponding `.cfg` files.
- Migration model excludes `src = dst` transitions at the action
  guard level.

## Properties NOT proved

1. Real wall-clock determinism on hardware. The TLA+ models reason
   over discrete steps; cycle-level invariance is asserted only via
   the runtime witness tests in `qratum-arbiter`.
2. Termination under adversarial liveness violations. Models check
   safety only.
3. Cross-architecture portability. Bounds are validated for x86_64;
   ARM64 + RISC-V claims are deferred to P0.6.

## Runtime witness obligations

Every formal property here MUST have a corresponding runtime witness
test:

| Formal property             | Runtime witness test                  |
|-----------------------------|---------------------------------------|
| `CanonicalEquivalence`      | `corpus_1000_equivalence.rs`          |
| `ReplayStable`              | `migration_ordering.rs`               |
| `OrderIndependent`          | `live_multicore_hash.rs`              |

The traceability matrix's "formal model" column enforces this link.

## P0.6 model limits

| Model                              | Bound                                  |
|------------------------------------|----------------------------------------|
| deterministic_work_steal.tla       | Cores=1..8, MaxDepth=0..32, Steps=32   |
| receipt_merkle_consistency.tla     | Leaves=symbolic; Hash injective        |
| replay_checkpoint_recovery.tla     | MaxEpoch=64                            |
| intercore_ordering.tla             | MaxEpoch=8, MaxSeq=64                  |
| divergence_envelope.tla            | Bounds=symbolic                        |
| hardware_replay_consistency.tla    | Events=symbolic; H injective           |
