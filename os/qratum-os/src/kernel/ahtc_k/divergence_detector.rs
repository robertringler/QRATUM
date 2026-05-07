//! P0.4 — Divergence detector.
//!
//! Consumed by `submit` and the host adversarial corpus. Two
//! complementary detectors:
//!
//!   1. `report_collision(a, b)`: when two `ShadowKey`s with
//!      identical primary canonical key disagree on any shadow
//!      dimension. Hard violation of INV-N — trips
//!      `InvariantCode::SemanticFalseFold`.
//!   2. `entropy_alarm(before, after, threshold_q)`: when the
//!      Shannon-entropy delta between input and reduced graph
//!      exceeds a configured tolerance — indicating that the fold
//!      collapsed information that wasn't actually duplicate.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

use super::equivalence_boundary::{EquivalenceVerdict, ShadowKey, compare};

static FALSE_POSITIVE_COUNT: AtomicU64 = AtomicU64::new(0);
static ENTROPY_ALARMS:       AtomicU64 = AtomicU64::new(0);

pub fn report_collision(a: &ShadowKey, b: &ShadowKey) -> EquivalenceVerdict {
    let v = compare(a, b);
    if !super::equivalence_boundary::is_safe_merge(&v) {
        FALSE_POSITIVE_COUNT.fetch_add(1, Ordering::Relaxed);
        // Hard invariant: the kernel must not silently merge
        // semantically distinct graphs. INV-N enforcement.
        super::super::invariants::trip(
            super::super::invariants::InvariantCode::SemanticFalseFold,
            b"ahtc_k:semantic_false_fold",
        );
    }
    v
}

/// Soft predicate — does NOT trip an invariant. Used by host-side
/// adversarial tests to assess fold quality.
pub fn safe_compare(a: &ShadowKey, b: &ShadowKey) -> EquivalenceVerdict {
    compare(a, b)
}

/// Entropy alarm. Returns true if (entropy_before - entropy_after)
/// exceeds `threshold_q`. The fold is allowed to *reduce* entropy
/// (that's the point); we alarm on *over*-reduction.
pub fn entropy_alarm(entropy_before_q: u64, entropy_after_q: u64, threshold_q: u64) -> bool {
    let drop = entropy_before_q.saturating_sub(entropy_after_q);
    let alarm = drop > threshold_q;
    if alarm { ENTROPY_ALARMS.fetch_add(1, Ordering::Relaxed); }
    alarm
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct DivergenceSummary {
    pub false_positive_count: u64,
    pub entropy_alarms:       u64,
}

pub fn summary() -> DivergenceSummary {
    DivergenceSummary {
        false_positive_count: FALSE_POSITIVE_COUNT.load(Ordering::Relaxed),
        entropy_alarms:       ENTROPY_ALARMS.load(Ordering::Relaxed),
    }
}

pub fn reset() {
    FALSE_POSITIVE_COUNT.store(0, Ordering::Relaxed);
    ENTROPY_ALARMS.store(0, Ordering::Relaxed);
}
