//! P0.5 — Migration cost estimator.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

static TOTAL_CYC: AtomicU64 = AtomicU64::new(0);
static SAMPLES:   AtomicU64 = AtomicU64::new(0);
static MAX_CYC:   AtomicU64 = AtomicU64::new(0);

pub fn record(start: u64, end: u64) {
    let dt = end.wrapping_sub(start);
    TOTAL_CYC.fetch_add(dt, Ordering::Relaxed);
    SAMPLES.fetch_add(1, Ordering::Relaxed);
    let mut cur = MAX_CYC.load(Ordering::Relaxed);
    while dt > cur {
        match MAX_CYC.compare_exchange_weak(cur, dt, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(v) => cur = v,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct MigrationCostSummary {
    pub samples:    u64,
    pub total_cyc:  u64,
    pub max_cyc:    u64,
    pub avg_cyc:    u64,
}

pub fn summary() -> MigrationCostSummary {
    let s = SAMPLES.load(Ordering::Relaxed);
    let t = TOTAL_CYC.load(Ordering::Relaxed);
    let m = MAX_CYC.load(Ordering::Relaxed);
    let avg = if s == 0 { 0 } else { t / s };
    MigrationCostSummary { samples: s, total_cyc: t, max_cyc: m, avg_cyc: avg }
}

pub fn reset() {
    TOTAL_CYC.store(0, Ordering::Release);
    SAMPLES.store(0, Ordering::Release);
    MAX_CYC.store(0, Ordering::Release);
}
