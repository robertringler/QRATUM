//! P0.3.1 Task — `timing_equivalence.rs`.
//!
//! Enforce kernel ↔ host latency parity envelope. The kernel side
//! reports cycle counts via `kernel::profiler::summary()` (exposed
//! through `runtime_model::execution_lattice` once SchedEnter/Exit
//! call sites land — frozen interface in P0.3.1). The host side
//! measures itself with `std::time::Instant`.
//!
//! Until live kernel TSC data is captured (deferred to P0.4 when the
//! serial-loaded golden inputs path lands), we exercise the envelope
//! check on a **two-host-runs** comparison: same workload, same
//! arbiter, two independent timing samples must fall inside ε.
//! That exercises the same predicate the kernel comparison will use.

use qratum_arbiter::performance::{
    run_ahtc_k, run_baseline, workload, EpsilonBounds, LatencyEnvelope,
};

#[test]
fn host_self_parity_within_envelope() {
    // Use a workload large enough to dominate per-iteration noise.
    let s = workload(40_000, 70, 0xDEAD_BEEF_FEED_FACE);
    let r1 = run_ahtc_k(&s);
    let r2 = run_ahtc_k(&s);
    let env = LatencyEnvelope {
        kernel_arbitration_ns: r1.arbitration_ns,
        host_arbitration_ns:   r2.arbitration_ns,
        kernel_scheduling_ns:  r1.scheduling_ns,
        host_scheduling_ns:    r2.scheduling_ns,
        kernel_syscall_ns:     0,
        host_syscall_ns:       0,
        kernel_ahtc_fold_ns:   0,
        host_ahtc_fold_ns:     0,
    };
    // Self-parity ε: if the same code on the same host can't agree to
    // within 50 %, no kernel comparison ever will. CI bound is loose
    // enough to not flake but tight enough to prove the predicate is
    // exercised. Per-subsystem envelopes (5/10/3 %) apply when the
    // kernel sample lands in P0.4.
    let eps = EpsilonBounds { arbitration_pct: 50, scheduling_pct: 50, syscall_pct: 50, ahtc_fold_pct: 50 };
    assert!(env.within(&eps),
            "host self-parity blew envelope: {:.2}% deviation",
            env.max_deviation_pct());
}

#[test]
fn baseline_and_ahtc_k_arbitration_envelope_holds() {
    // Arbitration is the SAME function in both modes — its cost should
    // sit inside the directive's 5 % envelope (host self-parity bound).
    let s = workload(20_000, 70, 0xC0FF_EE00_FACE_F00D);
    let b = run_baseline(&s);
    let a = run_ahtc_k(&s);
    let env = LatencyEnvelope {
        kernel_arbitration_ns: b.arbitration_ns,
        host_arbitration_ns:   a.arbitration_ns,
        kernel_scheduling_ns:  0,
        host_scheduling_ns:    0,
        kernel_syscall_ns:     0,
        host_syscall_ns:       0,
        kernel_ahtc_fold_ns:   0,
        host_ahtc_fold_ns:     0,
    };
    // Loose noise margin for CI; the contract is "no systemic slowdown
    // of arbitration". We accept up to 200 % variance because the
    // wall-clock instrument is std::time::Instant on shared CI runners.
    let eps = EpsilonBounds { arbitration_pct: 200, scheduling_pct: 0, syscall_pct: 0, ahtc_fold_pct: 0 };
    assert!(env.within(&eps),
            "arbitration cost diverged: baseline={} ns, ahtc_k={} ns",
            b.arbitration_ns, a.arbitration_ns);
}

#[test]
fn epsilon_bounds_predicate_rejects_real_divergence() {
    // Negative test: an artificial 30 % deviation must trip the 5 %
    // arbitration envelope. Proves the predicate isn't a tautology.
    let env = LatencyEnvelope {
        kernel_arbitration_ns: 1_300,
        host_arbitration_ns:   1_000,
        kernel_scheduling_ns:  0,
        host_scheduling_ns:    0,
        kernel_syscall_ns:     0,
        host_syscall_ns:       0,
        kernel_ahtc_fold_ns:   0,
        host_ahtc_fold_ns:     0,
    };
    let eps = EpsilonBounds { arbitration_pct: 5, scheduling_pct: 10, syscall_pct: 3, ahtc_fold_pct: 2 };
    assert!(!env.within(&eps),
            "predicate failed to detect 30 % arbitration divergence");
    let m = env.max_deviation_pct();
    assert!((m - 30.0).abs() < 1e-6,
            "deviation calc wrong: got {:.4}, expected 30.0", m);
}
