//! P0.4 — Latency stability: measured arbitration / scheduling time
//! must not vary by more than the configured envelope across runs.

use qratum_arbiter::performance::{run_ahtc_k, workload, EpsilonBounds, LatencyEnvelope};

fn within(a: u64, b: u64, pct: u32) -> bool {
    if a == 0 && b == 0 { return true; }
    let hi = a.max(b) as u128;
    let lo = a.min(b) as u128;
    if hi == 0 { return true; }
    ((hi - lo) * 100) <= (hi * pct as u128)
}

#[test]
fn run_to_run_arbitration_is_within_envelope() {
    let s = workload(20_000, 70, 0xFEED_FACE_C0DE_BABE);
    let r1 = run_ahtc_k(&s);
    let r2 = run_ahtc_k(&s);
    let env = LatencyEnvelope {
        kernel_arbitration_ns: r1.arbitration_ns,
        host_arbitration_ns:   r2.arbitration_ns,
        kernel_scheduling_ns:  r1.scheduling_ns,
        host_scheduling_ns:    r2.scheduling_ns,
        kernel_syscall_ns:     0, host_syscall_ns: 0,
        kernel_ahtc_fold_ns:   0, host_ahtc_fold_ns: 0,
    };
    let eps = EpsilonBounds { arbitration_pct: 60, scheduling_pct: 60,
                              syscall_pct: 0, ahtc_fold_pct: 0 };
    assert!(env.within(&eps),
        "run-to-run blew envelope: {:.2}%", env.max_deviation_pct());
}

#[test]
fn semantic_results_are_byte_identical_across_runs() {
    // Audit-frame and enqueue counts MUST be deterministic;
    // wall-clock variance is allowed within envelope.
    let s = workload(5_000, 70, 0x12345678_9ABCDEF0);
    let r1 = run_ahtc_k(&s);
    let r2 = run_ahtc_k(&s);
    assert_eq!(r1.events_in,    r2.events_in);
    assert_eq!(r1.enqueues,     r2.enqueues);
    assert_eq!(r1.batches,      r2.batches);
    assert_eq!(r1.audit_frames, r2.audit_frames);
    assert_eq!(r1.memory_bytes, r2.memory_bytes);
}

#[test]
fn invariant_tsc_proxy_is_monotone() {
    // Successive Instant samples must be monotone; this is the
    // host-side proxy for the kernel's `invariant TSC` requirement.
    let mut last = std::time::Instant::now();
    let mut hits = 0u32;
    for _ in 0..1_000 {
        let t = std::time::Instant::now();
        if t >= last { hits += 1; }
        last = t;
    }
    assert_eq!(hits, 1_000, "monotonic source was not monotone — fallback path required");
    let _ = within(0, 0, 0);
}
