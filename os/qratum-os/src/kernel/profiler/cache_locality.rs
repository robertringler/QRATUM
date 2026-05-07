//! P0.5 — Cache-locality scoring (kernel side, software-counted).

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

static REUSE_HOT:    AtomicU64 = AtomicU64::new(0);
static REUSE_WARM:   AtomicU64 = AtomicU64::new(0);
static REUSE_COLD:   AtomicU64 = AtomicU64::new(0);

#[inline] pub fn note_hot()  { REUSE_HOT.fetch_add(1, Ordering::Relaxed); }
#[inline] pub fn note_warm() { REUSE_WARM.fetch_add(1, Ordering::Relaxed); }
#[inline] pub fn note_cold() { REUSE_COLD.fetch_add(1, Ordering::Relaxed); }

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct LocalityScore {
    pub hot:        u64,
    pub warm:       u64,
    pub cold:       u64,
    /// Weighted score in [0, 1_000_000]. Hot = 1, warm = 0.5, cold = 0.
    pub score_q:    u64,
}

pub const SCALE: u64 = 1_000_000;

pub fn score() -> LocalityScore {
    let h = REUSE_HOT.load(Ordering::Relaxed);
    let w = REUSE_WARM.load(Ordering::Relaxed);
    let c = REUSE_COLD.load(Ordering::Relaxed);
    let denom = h + w + c;
    let score_q = if denom == 0 { 0 }
                  else { (2 * h + w) * SCALE / (2 * denom) };
    LocalityScore { hot: h, warm: w, cold: c, score_q }
}

pub fn reset() {
    REUSE_HOT.store(0, Ordering::Release);
    REUSE_WARM.store(0, Ordering::Release);
    REUSE_COLD.store(0, Ordering::Release);
}
