//! P0.6 — Replay drift profiler. Difference between live TSC and
//! replay TSC must remain bounded.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

static MAX_DRIFT: AtomicU64 = AtomicU64::new(0);
static SAMPLES:   AtomicU64 = AtomicU64::new(0);

pub fn record(live_cyc: u64, replay_cyc: u64) {
    SAMPLES.fetch_add(1, Ordering::Relaxed);
    let drift = if live_cyc > replay_cyc { live_cyc - replay_cyc } else { replay_cyc - live_cyc };
    let mut m = MAX_DRIFT.load(Ordering::Relaxed);
    while drift > m {
        match MAX_DRIFT.compare_exchange_weak(m, drift, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break, Err(v) => m = v,
        }
    }
}

#[inline]
pub fn max_drift() -> u64 { MAX_DRIFT.load(Ordering::Relaxed) }

#[inline]
pub fn samples() -> u64 { SAMPLES.load(Ordering::Relaxed) }

#[inline]
pub fn within(bound: u64) -> bool { max_drift() <= bound }

pub fn reset() {
    MAX_DRIFT.store(0, Ordering::Release);
    SAMPLES.store(0, Ordering::Release);
}
