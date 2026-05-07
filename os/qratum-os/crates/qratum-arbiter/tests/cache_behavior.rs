//! P0.4 — Cache locality measurement under AHTC-K vs baseline.

use qratum_arbiter::ahtc_k::{AhtcEngine, FoldOutcome};
use qratum_arbiter::hardware_profile::{locality_improved, HardwareProfiler};
use qratum_arbiter::performance::workload;

#[test]
fn ahtc_k_improves_locality_proxy() {
    let stream = workload(10_000, 70, 0xCACE_CACE_CACE_CACE);

    // Baseline: every event is a "cold insert".
    let mut bp = HardwareProfiler::new();
    for _ in &stream { bp.note_cold(); }
    let baseline = bp.finalize();

    // AHTC-K: hits → reuse, creates → cold.
    let mut ap = HardwareProfiler::new();
    let mut eng = AhtcEngine::new();
    for env in &stream {
        match eng.fold_one(env) {
            FoldOutcome::Created     { .. } => ap.note_cold(),
            FoldOutcome::Incremented { .. } => ap.note_reuse(),
        }
    }
    let ahtc = ap.finalize();

    // 70 % duplication should yield > 25 % locality boost.
    assert!(locality_improved(&baseline, &ahtc, 250_000),
        "baseline locality_q={}, ahtc locality_q={}",
        baseline.cache_locality_q, ahtc.cache_locality_q);
}

#[test]
fn locality_is_well_defined_for_both_workloads() {
    // Both 70 % and 0 % synthetic workloads exhibit high locality
    // because the COLD_POOL is bounded at 32 distinct keys. Assert
    // that the locality metric is well-formed (in [0, 1_000_000])
    // and that AHTC-K is producing real fold reuse for both.
    let measure = |dup: u32| -> u64 {
        let stream = workload(2_000, dup, 0xC0DE_C0DE);
        let mut p = HardwareProfiler::new();
        let mut e = AhtcEngine::new();
        for env in &stream {
            match e.fold_one(env) {
                FoldOutcome::Created     { .. } => p.note_cold(),
                FoldOutcome::Incremented { .. } => p.note_reuse(),
            }
        }
        p.finalize().cache_locality_q
    };
    let lq_hi = measure(70);
    let lq_lo = measure(0);
    assert!(lq_hi <= 1_000_000 && lq_lo <= 1_000_000);
    assert!(lq_hi > 500_000, "70%% locality must be > 50%%, got {}", lq_hi);
    assert!(lq_lo > 0, "even 0%% workload must have some reuse via cold pool");
}
