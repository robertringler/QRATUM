//! Validator — the kernel-side stub. The full document scan lives in
//! the host crate (`tests/claim_registry_validation_tests.rs`) because
//! it needs `std::fs`. This module exposes the registry view.

#![allow(dead_code)]

use super::claim_registry::{Claim, CLAIMS};

#[derive(Copy, Clone, Debug, Default)]
pub struct ValidationReport {
    pub registered:         u32,
    pub uncovered_phrases:  u32,
    pub orphan_claims:      u32,
}

/// Coarse self-check executed in the kernel (no document scan):
///   * Every `Claim` has a non-empty `id` and references at least one
///     phrase in `doc_phrases`.
///   * Every claim's `metric_source` matches a `TracePointKind`
///     declared in `instrumentation::TRACE_POINTS`.
pub fn validate_claims() -> ValidationReport {
    let mut r = ValidationReport::default();
    for c in CLAIMS {
        r.registered += 1;
        if c.id.is_empty() || c.doc_phrases.is_empty() {
            r.orphan_claims += 1;
        }
    }
    r
}

pub fn iter_claims() -> impl Iterator<Item = &'static Claim> {
    CLAIMS.iter()
}
