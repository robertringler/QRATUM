//! P0.4 — Per-core local storage.
//!
//! `CoreLocal<T>` is a fixed-array `[T; MAX_CPUS]` with a stable
//! per-core view. Access is via `with(cpu, f)` — never raw indexing —
//! so future percpu-segment migrations are a one-line change.
//!
//! Lock ordering (global, see `barriers::ORDER_DOC`):
//!   1. `Affinity` table  (read-mostly)
//!   2. `CoreLocal<RunQueue>` of source core
//!   3. `CoreLocal<RunQueue>` of destination core (lower CPU id first)
//!   4. `MigrationLog`
//!   5. Audit chain
//!
//! Crossing this order is INV-K territory.

#![allow(dead_code)]

use core::cell::UnsafeCell;
use super::MAX_CPUS;

pub struct CoreLocal<T> { slots: [UnsafeCell<T>; MAX_CPUS] }

// SAFETY: callers must serialize per-slot access externally (mutex or
// per-core invariant). The type is a primitive — it adds no synchro.
unsafe impl<T: Send> Sync for CoreLocal<T> {}

impl<T: Copy> CoreLocal<T> {
    pub const fn new(init: T) -> Self {
        Self { slots: [
            UnsafeCell::new(init), UnsafeCell::new(init),
            UnsafeCell::new(init), UnsafeCell::new(init),
            UnsafeCell::new(init), UnsafeCell::new(init),
            UnsafeCell::new(init), UnsafeCell::new(init),
        ] }
    }
}

impl<T> CoreLocal<T> {
    /// Per-core mutable access. Caller asserts no other core is in
    /// the same slot — enforced by either (a) the global migration
    /// barrier (see `barriers::with_migration_barrier`) or
    /// (b) lock ordering above.
    #[inline]
    pub unsafe fn with<R>(&self, cpu: u8, f: impl FnOnce(&mut T) -> R) -> R {
        let i = (cpu as usize).min(MAX_CPUS - 1);
        f(&mut *self.slots[i].get())
    }

    /// Read-only snapshot used by reconciliation paths. Iterates
    /// 0..MAX_CPUS in order so the snapshot is byte-stable.
    #[inline]
    pub fn snapshot<R: Default>(&self, mut f: impl FnMut(usize, &T) -> R) -> [R; MAX_CPUS] {
        let mut out: [R; MAX_CPUS] = Default::default();
        for i in 0..MAX_CPUS {
            // SAFETY: snapshot reads are valid as long as the global
            // barrier is held; callers must enforce.
            let r = unsafe { &*self.slots[i].get() };
            out[i] = f(i, r);
        }
        out
    }
}
