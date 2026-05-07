//! P0.6 — Execution epoch. INV monotonicity: epochs only advance.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

static EPOCH: AtomicU64 = AtomicU64::new(0);

#[inline]
pub fn current() -> u64 { EPOCH.load(Ordering::Acquire) }

pub fn advance() -> u64 {
    let prev = EPOCH.fetch_add(1, Ordering::AcqRel);
    prev + 1
}

#[inline]
pub fn is_monotone(prev: u64, next: u64) -> bool { next > prev }

pub fn reset() { EPOCH.store(0, Ordering::Release); }
