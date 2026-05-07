//! P0.4 — Performance claim integrity.
//!
//! Public docs MUST NOT use banned headline terms. Validated via
//! `performance_truth::validate_text` over the README + spec tree.

use std::fs;
use qratum_arbiter::performance_truth::{
    validate_claim, validate_text, ArbitrationMode, CacheState,
    PerformanceClaim, ReplayMode, TimingSource, WorkloadClass,
};

fn read(p: &str) -> String {
    let abs = format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), p);
    fs::read_to_string(&abs).unwrap_or_default()
}

#[test]
fn no_banned_headline_terms_anywhere_in_docs() {
    let docs = [
        "README.md",
        "spec/ahtc_k.md",
    ];
    let mut all_hits = Vec::new();
    for path in docs {
        let body = read(path);
        all_hits.extend(validate_text(path, &body));
    }
    assert!(all_hits.is_empty(),
        "banned headline terms found:\n{}",
        all_hits.iter()
            .map(|(p, t)| format!("  {} :: {}", p, t))
            .collect::<Vec<_>>().join("\n"));
}

#[test]
fn well_formed_claim_passes_validation() {
    let c = PerformanceClaim {
        headline: "scheduler event reduction at 70 % duplication".into(),
        workload_class: WorkloadClass::HighDuplication70Pct,
        duplication_pct: 70,
        event_count: 50_000,
        hardware_fingerprint: "x86_64/amd64; tsc=invariant; cpus=8".into(),
        cache_state: CacheState::Warm,
        arbitration_mode: ArbitrationMode::Strict,
        replay_mode: ReplayMode::Live,
        timing_source: TimingSource::MonotonicNs,
        measured_value: 1515.0,
        units: "x".into(),
    };
    assert!(validate_claim(&c).is_ok());
}

#[test]
fn banned_headline_is_rejected() {
    let c = PerformanceClaim {
        headline: "delivers OS-wide acceleration across all workloads".into(),
        workload_class: WorkloadClass::HighDuplication70Pct,
        duplication_pct: 70, event_count: 1,
        hardware_fingerprint: "x".into(),
        cache_state: CacheState::Warm,
        arbitration_mode: ArbitrationMode::Strict,
        replay_mode: ReplayMode::Live,
        timing_source: TimingSource::MonotonicNs,
        measured_value: 1.0, units: "x".into(),
    };
    assert!(validate_claim(&c).is_err());
}

#[test]
fn empty_units_is_rejected() {
    let c = PerformanceClaim {
        headline: "scheduler event reduction".into(),
        workload_class: WorkloadClass::HighDuplication70Pct,
        duplication_pct: 70, event_count: 100,
        hardware_fingerprint: "x".into(),
        cache_state: CacheState::Warm,
        arbitration_mode: ArbitrationMode::Strict,
        replay_mode: ReplayMode::Live,
        timing_source: TimingSource::MonotonicNs,
        measured_value: 1.0, units: "".into(),
    };
    assert!(validate_claim(&c).is_err());
}
