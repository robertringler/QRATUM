# P0.6 — Deterministic Multicore Model

## Model

Execution is the projection of a totally-ordered *event stream*
`(epoch, seq, cpu_id, payload)` onto N cores. The projection is
defined by `kernel::smp::deterministic_work_steal::pick_steal` —
purely a function of the queue-depth snapshot.

## Ordering rules

1. Within a core, events are ordered by `seq`.
2. Across cores, events are ordered by `(epoch, seq)`.
3. Migration commits are tagged with the current `migration_epoch::current()`.

## Equivalence

Two executions are *equivalent* if their receipt Merkle roots match.
Two cores are *interchangeable* if they share the same canonical
`(cpu_id, apic_id)` projection.

## Interaction with AHTC-K

R1 (`expand(fold(S)) == S`) and R2 (`arbiter(fold(S)) == arbiter(S)`)
extend trivially: the arbiter and AHTC-K operate on the canonical
event stream, never on per-core physical ordering.
