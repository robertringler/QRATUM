# Formal Models — P0.4

Three TLA+ specifications cover the externally-verifiable invariants.

| File                            | Property                              | Maps to    |
|---------------------------------|---------------------------------------|------------|
| `execution_equivalence.tla`     | `replayed = fold_log`                 | INV-J      |
| `arbitration_invariance.tla`    | `verdict_raw = verdict_folded`        | INV-N      |
| `replay_consistency.tla`        | identical migration log → identical hash | INV-L  |
| `replay_consistency.tla`        | no orphaned execution states          | INV-K      |

## Running TLC

```bash
# Linux/macOS
java -cp tla2tools.jar tlc2.TLC -workers auto -config <name>.cfg <name>.tla

# Windows PowerShell
java -cp tla2tools.jar tlc2.TLC -workers auto -config <name>.cfg <name>.tla
```

## State-space limits

| Model                        | Constants                              | Bound (states) |
|------------------------------|----------------------------------------|----------------|
| `execution_equivalence`      | `\|Intents\|≤16, MaxFolds≤32`          | ≤ 5 × 10⁵      |
| `arbitration_invariance`     | `\|Intents\|≤16`                       | ≤ 1 × 10⁵      |
| `replay_consistency`         | `\|Tasks\|≤16, \|Cores\|≤4`            | ≤ 1 × 10⁶      |

All models terminate within the listed bound on commodity hardware.

## Proof assumptions

- `SchedulerHash` is a pure deterministic function of the migration
  log. Real implementation uses FNV-1a-64 over fixed-iteration
  bytes — verified by `tests/migration_replay.rs`.
- `Verdict` depends only on the SET of canonical keys (no ordering,
  no count). Real implementation: `state_machine::arbitrate`.
- `fold_log` order matches arrival order. Real implementation: AHTC-K
  `RunQueue` preserves insertion order via `MemberRef`s.

## Failure conditions

If TLC reports any of:

- `ReplayLossless` violated → INV-J broken; fold/expand asymmetry.
- `ArbitrationInvariant` violated → fold changed verdict; INV-N broken.
- `ReplayConsistency` violated → migration log non-deterministic.
- `NoOrphanedStates` violated → INV-K broken (dual residency).

…the corresponding host test in
`crates/qratum-arbiter/tests/` is REQUIRED to also fail. CI gate
`P0.4 — formal model exec invariants` chains both checks.
