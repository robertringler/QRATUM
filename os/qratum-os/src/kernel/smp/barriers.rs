//! P0.4 — Cross-core barriers + memory ordering documentation.
//!
//! Two primitives:
//!
//!   * `with_migration_barrier(f)` — wraps a critical section that
//!     touches more than one core's run queue. Implemented as a
//!     ticket spinlock with `Acquire/Release` ordering.
//!   * `quiesce_for_replay()` — drains pending IPIs and asserts every
//!     core has observed all writes up to the call. Used by
//!     `replay::expand` to guarantee bytewise replay.
//!
//! Memory ordering rationale:
//!   * Writes to per-core run queues use `Release`; reads from the
//!     reconciliation path use `Acquire`. This pairs with x86's
//!     program-order guarantees for data dependencies — the only
//!     reorder we must defeat is store-load (handled by the lock
//!     `Release/Acquire` pairing).
//!
//! Lock ordering (global) — see `core_local::ORDER_DOC` mirror:
//!     Affinity → Source RQ → Dest RQ (lower id) → MigrationLog → Audit
//!
//! Crossing this ordering trips INV-K.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU32, Ordering};

pub const ORDER_DOC: &str =
    "AFFINITY < SRC_RQ < DST_RQ(lower-id) < MIGRATION_LOG < AUDIT";

static TICKET_NEXT:    AtomicU32 = AtomicU32::new(0);
static TICKET_SERVING: AtomicU32 = AtomicU32::new(0);
static HOLDER:         AtomicU32 = AtomicU32::new(u32::MAX);

#[inline]
pub fn is_locked() -> bool {
    HOLDER.load(Ordering::Acquire) != u32::MAX
}

pub fn with_migration_barrier<R>(f: impl FnOnce() -> R) -> R {
    let me = TICKET_NEXT.fetch_add(1, Ordering::AcqRel);
    while TICKET_SERVING.load(Ordering::Acquire) != me {
        core::hint::spin_loop();
    }
    HOLDER.store(super::current_cpu() as u32, Ordering::Release);
    let r = f();
    HOLDER.store(u32::MAX, Ordering::Release);
    TICKET_SERVING.store(me.wrapping_add(1), Ordering::Release);
    r
}

/// Drain every IPI inbox and assert ordering. P0.4 single-core has
/// nothing to drain; the function still emits a single fence so
/// callers depend on the *contract*, not the absence of pending work.
pub fn quiesce_for_replay() {
    super::ipi_stub::drain_all();
    core::sync::atomic::fence(Ordering::SeqCst);
}
