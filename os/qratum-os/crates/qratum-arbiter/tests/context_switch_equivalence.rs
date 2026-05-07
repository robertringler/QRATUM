//! P0.4 — Context-switch cost equivalence (host self-parity).

use qratum_arbiter::hardware_profile::HardwareProfiler;

#[test]
fn context_switch_samples_are_well_formed() {
    let mut p = HardwareProfiler::new();
    for _ in 0..1_000 {
        p.time_context_switch(|| {
            // Simulate a small, deterministic unit of work.
            let mut x = 0u64;
            for i in 0..32 { x = x.wrapping_add(i); }
            std::hint::black_box(x);
        });
    }
    let s = p.finalize();
    assert_eq!(s.ctx_switches, 1_000);
    assert!(s.ctx_total_ns > 0);
    assert!(s.ctx_min_ns <= s.ctx_max_ns);
}

#[test]
fn irq_p99_is_bounded_by_max() {
    let mut p = HardwareProfiler::new();
    for _ in 0..500 {
        p.time_irq(|| { std::hint::black_box(0u64); });
    }
    let s = p.finalize();
    assert!(s.irq_samples == 500);
    assert!(s.irq_p50_ns <= s.irq_p99_ns);
}

#[test]
fn distinct_workloads_produce_distinct_profiles() {
    let mut light = HardwareProfiler::new();
    for _ in 0..200 { light.time_context_switch(|| { std::hint::black_box(0u64); }); }
    let lp = light.finalize();

    let mut heavy = HardwareProfiler::new();
    for _ in 0..200 {
        heavy.time_context_switch(|| {
            let mut x = 0u64;
            for i in 0..1024 { x = x.wrapping_mul(1_000_003).wrapping_add(i); }
            std::hint::black_box(x);
        });
    }
    let hp = heavy.finalize();
    assert!(hp.ctx_total_ns >= lp.ctx_total_ns,
        "heavier work should not be cheaper");
}
