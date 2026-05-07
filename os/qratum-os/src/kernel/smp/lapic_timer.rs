//! P0.6 — LAPIC timer calibration + deterministic envelope.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct TimerEnvelope {
    pub min_cyc:    u64,
    pub max_cyc:    u64,
    pub samples:    u64,
}

static SAMPLES: AtomicU64 = AtomicU64::new(0);
static MIN_CYC: AtomicU64 = AtomicU64::new(u64::MAX);
static MAX_CYC: AtomicU64 = AtomicU64::new(0);

pub fn record(cyc: u64) {
    SAMPLES.fetch_add(1, Ordering::Relaxed);
    let mut cur = MIN_CYC.load(Ordering::Relaxed);
    while cyc < cur {
        match MIN_CYC.compare_exchange_weak(cur, cyc, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break, Err(v) => cur = v,
        }
    }
    let mut cur = MAX_CYC.load(Ordering::Relaxed);
    while cyc > cur {
        match MAX_CYC.compare_exchange_weak(cur, cyc, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break, Err(v) => cur = v,
        }
    }
}

pub fn envelope() -> TimerEnvelope {
    let s = SAMPLES.load(Ordering::Relaxed);
    let mn = if s == 0 { 0 } else { MIN_CYC.load(Ordering::Relaxed) };
    TimerEnvelope { min_cyc: mn, max_cyc: MAX_CYC.load(Ordering::Relaxed), samples: s }
}

/// Bounded jitter: `max - min <= bound`.
#[inline]
pub fn within(bound: u64) -> bool {
    let e = envelope();
    if e.samples == 0 { return true; }
    e.max_cyc.saturating_sub(e.min_cyc) <= bound
}

pub fn reset() {
    SAMPLES.store(0, Ordering::Release);
    MIN_CYC.store(u64::MAX, Ordering::Release);
    MAX_CYC.store(0, Ordering::Release);
}
