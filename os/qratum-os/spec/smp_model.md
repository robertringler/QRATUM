# QRATUM SMP Model (P0.3)

## 1. Per-core state

Every core owns:

```
struct PerCpu {
    cpu_id        : u8,
    sched_state   : SchedState,        // private TCB array, runqueue
    audit_segment : SegmentBuffer,     // private 4 KiB writer
    ipi_inbox     : RingBuf<Ipi>,      // lock-free SPSC from peers
    invariants    : InvariantCounters, // per-core hard-fail counters
}
```

There is **no shared scheduler queue**. Tasks migrate by IPI only.

## 2. Inter-processor interrupt model (IPI)

```
enum Ipi {
    Halt,                      // urgent
    SchedRebalance(TaskId),    // migrate task to this core
    AuditFlush,                // flush my audit segment now
    InvariantViolation(u32),   // remote invariant tripped
}
```

The kernel's `smp::ipi_stub` exposes `send(target_cpu, ipi)` and
`drain_inbox()`; on x86 this is to be backed by `xAPIC ICR` writes
in P0.3.1. P0.3 ships a single-core stub that pushes into the local
inbox so the rest of the kernel can be written against the final API.

## 3. Lock migration rules

The arbiter's `LockState` is **per-core** during P0.3. Cross-core
verdicts go through one of two protocols:

| Protocol | When                                  | Mechanism                                  |
|----------|---------------------------------------|--------------------------------------------|
| Local    | intent.scope ∈ {Process, Session}     | served by current core                     |
| Migrate  | intent.scope ∈ {Machine, Fleet}       | IPI to the BSP (cpu_id=0); BSP arbitrates  |

This avoids cross-core lock contention while preserving global
serializability for fleet-scope intents — the BSP becomes the
canonical witness, and only its `LockState` participates in
`Machine`/`Fleet` decisions.

## 4. Audit aggregation

Each core's audit_segment chains independently. On flush, segments
are tagged with `cpu_id` and sealed into a shared on-disk log under
the format v2 layout. Replay merges by `(cpu_id, seq)` and asserts
chain integrity per core.

## 5. Determinism caveat

Cross-core ordering is not globally deterministic without a global
time source. The P0.3 contract is therefore weaker than P0.2:

* **Per-core determinism** is a hard invariant.
* **Global determinism** holds only for `Machine`/`Fleet` scopes
  (which serialize through the BSP).

## 6. Status

* `smp/mod.rs` — `init()`, `current_cpu()`, `per_cpu()` accessors.
  Single-core implementation: returns cpu_id=0 unconditionally.
* `smp/ipi_stub.rs` — `send()`, `drain_inbox()`, `Ipi` enum.
  Single-core: same-core delivery via local ring.
* xAPIC backend, AP boot, BSP-vs-AP role split → P0.3.1.
