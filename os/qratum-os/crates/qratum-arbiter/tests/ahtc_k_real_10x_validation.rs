//! P0.3.1 Task — `ahtc_k_real_10x_validation.rs`.
//!
//! HARD GATE. Forces a 70 % duplication workload at 100 000 events and
//! asserts the scheduling reduction ratio is ≥ 10.0×. Build fails if
//! the directive's published 10× claim is not numerically met.

use qratum_arbiter::performance::{
    fold_metrics, run_ahtc_k, run_baseline, workload,
};

const N: usize     = 100_000;
const DUP_PCT: u32 = 70;
const SEED: u64    = 0x1010_1010_1010_1010u64;

#[test]
fn ahtc_k_meets_strict_10x_at_70pct_duplication() {
    let stream = workload(N, DUP_PCT, SEED);
    let before = run_baseline(&stream);
    let after  = run_ahtc_k(&stream);
    let report = fold_metrics(&before, &after);

    println!("[10x-validator] baseline.enqueues={} ahtc_k.enqueues={} ratio={:.3}",
             before.enqueues, after.enqueues, report.scheduling_reduction_ratio);

    assert!(report.meets_10x_at_70pct,
            "DIRECTIVE FAILURE: scheduling reduction = {:.3}× (< 10× required at 70 % dup). \
             baseline.enqueues={}, ahtc_k.enqueues={}, events_in={}",
            report.scheduling_reduction_ratio,
            before.enqueues, after.enqueues, before.events_in);
    assert!(report.scheduling_reduction_ratio >= 10.0);
}

#[test]
fn ratio_grows_monotone_above_50pct_dup() {
    // The 10× threshold is established at 70 %; the ratio must never
    // *fall* as duplication rises beyond 50 %.
    let mut prev = 0.0_f64;
    for dup in [50u32, 70, 80, 90, 95] {
        let s = workload(20_000, dup, 0xA5A5_A5A5_A5A5_A5A5);
        let r = fold_metrics(&run_baseline(&s), &run_ahtc_k(&s));
        assert!(r.scheduling_reduction_ratio + 1e-9 >= prev,
                "ratio collapsed at dup={dup}: prev={prev:.3} now={:.3}",
                r.scheduling_reduction_ratio);
        prev = r.scheduling_reduction_ratio;
    }
}
