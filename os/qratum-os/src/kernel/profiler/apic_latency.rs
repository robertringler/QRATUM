//! P0.6 — APIC IPI latency profiler.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

static SAMPLES: AtomicU64 = AtomicU64::new(0);
static SUM:     AtomicU64 = AtomicU64::new(0);
static MAX:     AtomicU64 = AtomicU64::new(0);

pub fn record(cyc: u64) {
    SAMPLES.fetch_add(1, Ordering::Relaxed);
    SUM.fetch_add(cyc, Ordering::Relaxed);
    let mut m = MAX.load(Ordering::Relaxed);
    while cyc > m {
        match MAX.compare_exchange_weak(m, cyc, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break, Err(v) => m = v,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct ApicLatency { pub samples: u64, pub avg: u64, pub max: u64 }

pub fn summary() -> ApicLatency {
    let s = SAMPLES.load(Ordering::Relaxed);
    let total = SUM.load(Ordering::Relaxed);
    ApicLatency {
        samples: s, max: MAX.load(Ordering::Relaxed),
        avg: if s == 0 { 0 } else { total / s },
    }
}

pub fn reset() {
    SAMPLES.store(0, Ordering::Release);
    SUM.store(0, Ordering::Release);
    MAX.store(0, Ordering::Release);
}
