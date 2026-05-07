//! P0.4 — Cache-locality proxy.
//!
//! True L1/L2/L3 metrics require PMU MSRs (perfmon). This module
//! provides a deterministic *proxy*: per-batch reuse distance over
//! AHTC-K canonical batches. Higher reuse = better locality.
//!
//! Concretely:
//!   - On every AhtcFold hit, increment `reuse_hits`.
//!   - On every AhtcFold create, increment `cold_inserts`.
//!   - locality_score = reuse_hits / (reuse_hits + cold_inserts).
//!
//! When PMU is later wired (DEFERRED to P0.5), the proxy stays as the
//! kernel-portable metric and PMU readings become the secondary
//! observation. Both are surfaced via `summary()`.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

static REUSE_HITS:   AtomicU64 = AtomicU64::new(0);
static COLD_INSERTS: AtomicU64 = AtomicU64::new(0);

#[inline]
pub fn note_reuse() { REUSE_HITS.fetch_add(1, Ordering::Relaxed); }
#[inline]
pub fn note_cold()  { COLD_INSERTS.fetch_add(1, Ordering::Relaxed); }

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct CacheSummary {
    pub reuse_hits:    u64,
    pub cold_inserts:  u64,
    /// reuse / (reuse + cold), scaled by 1_000_000.
    pub locality_q:    u64,
}

pub const LOCALITY_SCALE: u64 = 1_000_000;

pub fn summary() -> CacheSummary {
    let r = REUSE_HITS.load(Ordering::Relaxed);
    let c = COLD_INSERTS.load(Ordering::Relaxed);
    let denom = r + c;
    let q = if denom == 0 { 0 } else { r * LOCALITY_SCALE / denom };
    CacheSummary { reuse_hits: r, cold_inserts: c, locality_q: q }
}

pub fn reset() {
    REUSE_HITS.store(0, Ordering::Relaxed);
    COLD_INSERTS.store(0, Ordering::Relaxed);
}
