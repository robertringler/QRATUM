//! Per-trace-point coverage counters + Shannon-entropy helpers.
//!
//! Entropy is defined IDENTICALLY between kernel and host:
//!
//!     H(stream) = -Σ p_i · log2(p_i)
//!
//! over the empirical distribution of canonical-batch occupancies in
//! the AHTC-K run queue. The two implementations differ only in their
//! storage backing (kernel: fixed-array, host: `Vec`); the math is
//! the same, asserted by `tests/entropy_consistency_tests.rs`.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

use super::trace_point::{TracePointKind, TRACE_POINTS};

/// Fixed-point scaling for entropy reporting in no_std contexts where
/// `f64::log2` is unavailable. Host crate uses the same scale so
/// comparisons are bytewise.
pub const ENTROPY_SCALE: u64 = 1_000_000;

/// One u64 per trace point. 16 slots leaves 5 spare for P0.4 growth.
const COUNTER_SLOTS: usize = 16;
static COUNTERS: [AtomicU64; COUNTER_SLOTS] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
];

#[inline]
pub fn bump(kind: TracePointKind) {
    let idx = kind as u8 as usize;
    if idx < COUNTER_SLOTS {
        COUNTERS[idx].fetch_add(1, Ordering::Relaxed);
    }
}

pub fn count(kind: TracePointKind) -> u64 {
    COUNTERS[kind as u8 as usize].load(Ordering::Relaxed)
}

pub fn coverage_pct() -> u32 {
    let mut hit = 0u32;
    for spec in TRACE_POINTS {
        if count(spec.kind) > 0 { hit += 1; }
    }
    (hit * 100) / (TRACE_POINTS.len() as u32)
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct CompressionMetrics {
    /// Numerator: input events. Denominator: distinct canonical batches.
    /// Stored as scaled integer (× ENTROPY_SCALE).
    pub event_reduction_ratio_q:        u64,
    pub scheduling_reduction_ratio_q:   u64,
    pub memory_reduction_ratio_q:       u64,
    pub arbitration_equivalence_ratio_q: u64,
    /// Shannon entropy of the empirical batch-count distribution
    /// (× ENTROPY_SCALE bits).
    pub entropy_before_q: u64,
    pub entropy_after_q:  u64,
}

/// Integer Shannon entropy approximation, identical between kernel and
/// host. `counts` must be the per-batch occupancy histogram. Returns
/// entropy in *milli-bits* (× ENTROPY_SCALE).
///
/// Implementation: integer log2 of the rational p_i = count_i / total
/// using `u64::ilog2`. This is exact for power-of-two probabilities
/// and a deterministic lower bound otherwise — same on both sides.
pub fn shannon_entropy_bits(counts: &[u64]) -> u64 {
    let total: u128 = counts.iter().map(|&c| c as u128).sum();
    if total == 0 { return 0; }
    let mut acc: u128 = 0;
    for &c in counts {
        if c == 0 { continue; }
        // p = c / total. log2(p) = log2(c) - log2(total).
        // We want -p * log2(p) = (c/total) * (log2(total) - log2(c)).
        let log_total = (total as u64).ilog2() as u128;
        let log_c     = (c as u64).max(1).ilog2() as u128;
        let term = (c as u128) * (log_total - log_c) * (ENTROPY_SCALE as u128);
        acc += term / total;
    }
    acc as u64
}
