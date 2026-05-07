//! P0.4 — Context switch cycle counter.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

static SWITCHES:    AtomicU64 = AtomicU64::new(0);
static TOTAL_CYC:   AtomicU64 = AtomicU64::new(0);
static MIN_CYC:     AtomicU64 = AtomicU64::new(u64::MAX);
static MAX_CYC:     AtomicU64 = AtomicU64::new(0);

#[inline]
pub fn record(start: u64, end: u64) {
    let d = end.saturating_sub(start);
    SWITCHES.fetch_add(1, Ordering::Relaxed);
    TOTAL_CYC.fetch_add(d, Ordering::Relaxed);
    let mut m = MIN_CYC.load(Ordering::Relaxed);
    while d < m {
        match MIN_CYC.compare_exchange_weak(m, d, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break, Err(cur) => m = cur,
        }
    }
    let mut x = MAX_CYC.load(Ordering::Relaxed);
    while d > x {
        match MAX_CYC.compare_exchange_weak(x, d, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break, Err(cur) => x = cur,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct ContextSwitchSummary {
    pub switches:   u64,
    pub total_cyc:  u64,
    pub min_cyc:    u64,
    pub max_cyc:    u64,
    pub avg_cyc:    u64,
}

pub fn summary() -> ContextSwitchSummary {
    let s = SWITCHES.load(Ordering::Relaxed);
    let t = TOTAL_CYC.load(Ordering::Relaxed);
    let mn = MIN_CYC.load(Ordering::Relaxed);
    ContextSwitchSummary {
        switches: s, total_cyc: t,
        min_cyc:  if mn == u64::MAX { 0 } else { mn },
        max_cyc:  MAX_CYC.load(Ordering::Relaxed),
        avg_cyc:  if s == 0 { 0 } else { t / s },
    }
}

pub fn reset() {
    SWITCHES.store(0, Ordering::Relaxed);
    TOTAL_CYC.store(0, Ordering::Relaxed);
    MIN_CYC.store(u64::MAX, Ordering::Relaxed);
    MAX_CYC.store(0, Ordering::Relaxed);
}
