//! P0.6 — Run-queue pressure profiler.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

static MAX_DEPTH: AtomicU64 = AtomicU64::new(0);
static SAMPLES:   AtomicU64 = AtomicU64::new(0);
static SUM:       AtomicU64 = AtomicU64::new(0);

pub fn note(depth: u64) {
    SAMPLES.fetch_add(1, Ordering::Relaxed);
    SUM.fetch_add(depth, Ordering::Relaxed);
    let mut m = MAX_DEPTH.load(Ordering::Relaxed);
    while depth > m {
        match MAX_DEPTH.compare_exchange_weak(m, depth, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break, Err(v) => m = v,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Pressure { pub samples: u64, pub avg: u64, pub max: u64 }

pub fn summary() -> Pressure {
    let s = SAMPLES.load(Ordering::Relaxed);
    Pressure {
        samples: s, max: MAX_DEPTH.load(Ordering::Relaxed),
        avg: if s == 0 { 0 } else { SUM.load(Ordering::Relaxed) / s },
    }
}

pub fn reset() {
    MAX_DEPTH.store(0, Ordering::Release);
    SAMPLES.store(0, Ordering::Release);
    SUM.store(0, Ordering::Release);
}
