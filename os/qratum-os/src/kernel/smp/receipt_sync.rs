//! P0.6 — Receipt synchronization barrier.
//!
//! Ensures all per-CPU receipt heads observe the same monotone epoch
//! before the next batch may proceed.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

use super::MAX_CPUS;

static EPOCH: AtomicU64 = AtomicU64::new(0);
static OBSERVED: [AtomicU64; MAX_CPUS] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
];

#[inline]
pub fn epoch() -> u64 { EPOCH.load(Ordering::Acquire) }

pub fn advance() -> u64 { EPOCH.fetch_add(1, Ordering::AcqRel) + 1 }

pub fn observe(cpu_id: u8, e: u64) {
    if (cpu_id as usize) < MAX_CPUS {
        OBSERVED[cpu_id as usize].store(e, Ordering::Release);
    }
}

/// True when all online CPUs have observed an epoch ≥ `target`.
pub fn synchronized(target: u64) -> bool {
    for i in 0..MAX_CPUS {
        if OBSERVED[i].load(Ordering::Acquire) < target { return false; }
    }
    true
}

pub fn reset() {
    EPOCH.store(0, Ordering::Release);
    for s in &OBSERVED { s.store(0, Ordering::Release); }
}
