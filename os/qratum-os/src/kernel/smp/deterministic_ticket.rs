//! P0.5 — Deterministic ticket spinlock (replay-stable lock order).
//!
//! Acquisition order is identical across runs of the same workload
//! because every caller draws a monotonic ticket from a global
//! counter. The ticket value is recorded in
//! `contention_trace::record_acquire`.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug)]
pub struct DeterministicTicket {
    next:    AtomicU64,
    serving: AtomicU64,
    id:      u32,
}

impl DeterministicTicket {
    pub const fn new(id: u32) -> Self {
        Self { next: AtomicU64::new(0), serving: AtomicU64::new(0), id }
    }

    pub fn acquire(&self) -> u64 {
        let t = self.next.fetch_add(1, Ordering::AcqRel);
        // Spin until we are served. In single-core builds this loop
        // executes zero times; on real SMP it is the standard ticket
        // wait. No fairness anomalies because tickets are monotone.
        while self.serving.load(Ordering::Acquire) != t {
            core::hint::spin_loop();
        }
        super::contention_trace::record_acquire(self.id, t);
        t
    }

    pub fn release(&self, ticket: u64) {
        super::contention_trace::record_release(self.id, ticket);
        self.serving.store(ticket.wrapping_add(1), Ordering::Release);
    }

    #[inline]
    pub fn next_ticket(&self) -> u64 { self.next.load(Ordering::Acquire) }
}
