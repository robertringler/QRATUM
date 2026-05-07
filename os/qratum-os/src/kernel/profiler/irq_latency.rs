//! P0.4 — IRQ latency histogram (RDTSC-based).
//!
//! `enter()` records the entry timestamp; `exit()` reads the matching
//! entry, computes delta, bins it. Histogram uses 16 power-of-two
//! buckets in cycles. Bucket index = floor(log2(cycles)).

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

const BUCKETS: usize = 16;
static HIST: [AtomicU64; BUCKETS] = {
    const Z: AtomicU64 = AtomicU64::new(0);
    [Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z]
};
static TOTAL_CYCLES: AtomicU64 = AtomicU64::new(0);
static SAMPLES:      AtomicU64 = AtomicU64::new(0);
static LAST_ENTER:   AtomicU64 = AtomicU64::new(0);

#[inline]
pub fn enter() -> u64 {
    let t = super::cycle_counter::rdtsc();
    LAST_ENTER.store(t, Ordering::Relaxed);
    t
}

#[inline]
pub fn exit(entered_at: u64) {
    let now = super::cycle_counter::rdtsc();
    let d = now.saturating_sub(entered_at);
    let bucket = if d == 0 { 0 } else { (d.ilog2() as usize).min(BUCKETS - 1) };
    HIST[bucket].fetch_add(1, Ordering::Relaxed);
    TOTAL_CYCLES.fetch_add(d, Ordering::Relaxed);
    SAMPLES.fetch_add(1, Ordering::Relaxed);
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct IrqLatencySummary {
    pub samples:        u64,
    pub total_cycles:   u64,
    pub histogram:      [u64; BUCKETS],
    pub p50_bucket:     u8,
    pub p99_bucket:     u8,
}

pub fn summary() -> IrqLatencySummary {
    let mut hist = [0u64; BUCKETS];
    for (i, c) in HIST.iter().enumerate() { hist[i] = c.load(Ordering::Relaxed); }
    let total: u64 = hist.iter().sum();
    let p50_th = total / 2;
    let p99_th = total * 99 / 100;
    let (mut p50, mut p99, mut acc) = (0u8, 0u8, 0u64);
    for (i, &c) in hist.iter().enumerate() {
        acc += c;
        if acc >= p50_th && p50 == 0 { p50 = i as u8; }
        if acc >= p99_th { p99 = i as u8; break; }
    }
    IrqLatencySummary {
        samples: SAMPLES.load(Ordering::Relaxed),
        total_cycles: TOTAL_CYCLES.load(Ordering::Relaxed),
        histogram: hist, p50_bucket: p50, p99_bucket: p99,
    }
}

pub fn reset() {
    for c in &HIST { c.store(0, Ordering::Relaxed); }
    TOTAL_CYCLES.store(0, Ordering::Relaxed);
    SAMPLES.store(0, Ordering::Relaxed);
}
