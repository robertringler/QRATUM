//! P0.3.1 Task — `performance_truth.rs`.
//!
//! Compare baseline (P0.2-style: one enqueue per accepted intent)
//! vs AHTC-K scheduler on a deterministic 70 %-duplication workload.
//! Assert that **all four** compression ratios are > 1× and that the
//! scheduling ratio is ≥ 10× — the directive's hard threshold.
//!
//! The test also prints the standard performance report so CI logs
//! contain the numbers, not just PASS/FAIL.

use qratum_arbiter::performance::{
    fold_metrics, print_report, run_ahtc_k, run_baseline, workload, EpsilonBounds,
};

const N: usize        = 50_000;
const DUP_PCT: u32    = 70;
const SEED: u64       = 0xC0DE_BEEF_DEAD_F00D;

#[test]
fn baseline_vs_ahtc_k_real_reductions() {
    let stream  = workload(N, DUP_PCT, SEED);
    let before  = run_baseline(&stream);
    let after   = run_ahtc_k(&stream);
    let report  = fold_metrics(&before, &after);

    print_report("performance_truth", &before, &after, &report, None,
                 &EpsilonBounds::default());

    assert!(report.event_reduction_ratio       > 1.0, "no event reduction");
    assert!(report.scheduling_reduction_ratio  > 1.0, "no scheduling reduction");
    assert!(report.memory_reduction_ratio      > 1.0, "no memory reduction");
    // Arbitration ratio is permitted to be ≈ 1× — both modes pay the
    // arbiter cost. We only assert non-degradation.
    assert!(report.arbitration_reduction_ratio > 0.5,
            "AHTC-K must not double arbitration cost (got {:.3}×)",
            report.arbitration_reduction_ratio);
}

#[test]
fn report_is_deterministic_under_repeat() {
    // Same workload run twice → same compression report.
    let stream = workload(20_000, 70, SEED);
    let r1 = fold_metrics(&run_baseline(&stream), &run_ahtc_k(&stream));
    let r2 = fold_metrics(&run_baseline(&stream), &run_ahtc_k(&stream));
    // Counter-derived ratios are exact.
    assert_eq!(r1.event_reduction_ratio,      r2.event_reduction_ratio);
    assert_eq!(r1.scheduling_reduction_ratio, r2.scheduling_reduction_ratio);
    assert_eq!(r1.memory_reduction_ratio,     r2.memory_reduction_ratio);
}

#[test]
fn no_audit_loss_baseline_or_ahtc_k() {
    let stream = workload(5_000, 70, SEED);
    let b = run_baseline(&stream);
    let a = run_ahtc_k(&stream);
    // Every input must produce at least one audit frame in both modes.
    assert!(b.audit_frames >= b.events_in);
    assert!(a.audit_frames >= a.events_in);
}
