//! P0.6 — IRQ jitter profiler. Tracks min/max gap between IRQs.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

static LAST: AtomicU64 = AtomicU64::new(0);
static MIN_GAP: AtomicU64 = AtomicU64::new(u64::MAX);
static MAX_GAP: AtomicU64 = AtomicU64::new(0);
static SAMPLES: AtomicU64 = AtomicU64::new(0);

pub fn record(tsc: u64) {
    let prev = LAST.swap(tsc, Ordering::AcqRel);
    if prev == 0 { return; }
    let gap = tsc.saturating_sub(prev);
    SAMPLES.fetch_add(1, Ordering::Relaxed);
    let mut m = MIN_GAP.load(Ordering::Relaxed);
    while gap < m {
        match MIN_GAP.compare_exchange_weak(m, gap, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break, Err(v) => m = v,
        }
    }
    let mut m = MAX_GAP.load(Ordering::Relaxed);
    while gap > m {
        match MAX_GAP.compare_exchange_weak(m, gap, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break, Err(v) => m = v,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct IrqJitter { pub samples: u64, pub min_gap: u64, pub max_gap: u64 }

pub fn summary() -> IrqJitter {
    let s = SAMPLES.load(Ordering::Relaxed);
    IrqJitter {
        samples: s,
        min_gap: if s == 0 { 0 } else { MIN_GAP.load(Ordering::Relaxed) },
        max_gap: MAX_GAP.load(Ordering::Relaxed),
    }
}

pub fn reset() {
    LAST.store(0, Ordering::Release);
    MIN_GAP.store(u64::MAX, Ordering::Release);
    MAX_GAP.store(0, Ordering::Release);
    SAMPLES.store(0, Ordering::Release);
}
