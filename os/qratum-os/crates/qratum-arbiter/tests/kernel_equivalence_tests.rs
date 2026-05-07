//! Kernel ↔ arbiter equivalence — the directive's bit-for-bit gate.
//!
//! Six tests collectively prove:
//!   * Every one of the 270 corpus scenarios round-trips through
//!     `synthesize_kernel_state → verify_kernel_equivalence`.
//!   * The 4-hash determinism receipt is byte-stable across runs.
//!   * Projection hash is bytewise-stable.
//!   * Mutating the kernel state OFF-by-one (any field) is detected.

use qratum_arbiter::corpus::build_corpus;
use qratum_arbiter::kernel_equivalence::{
    compute_receipt, synthesize_kernel_state, verify_corpus_equivalence,
    verify_kernel_equivalence, DeterminismReceipt, ExecutionState,
};

#[test]
fn corpus_270_kernel_arbiter_equivalence() {
    let mismatches = verify_corpus_equivalence();
    assert_eq!(mismatches, 0,
               "kernel↔arbiter equivalence FAIL: {mismatches} mismatched scenarios");
}

#[test]
fn synthesize_then_verify_per_scenario() {
    for s in build_corpus() {
        let synth = synthesize_kernel_state(&s.ops);
        assert!(
            verify_kernel_equivalence(&synth, &s.ops),
            "scenario `{}` failed kernel-equivalence check", s.name
        );
    }
}

#[test]
fn projection_hash_is_byte_stable() {
    let corpus = build_corpus();
    let s = &corpus[0];
    let h1 = synthesize_kernel_state(&s.ops).projection_hash();
    let h2 = synthesize_kernel_state(&s.ops).projection_hash();
    assert_eq!(h1, h2, "projection_hash not deterministic");

    // Mutate a single field — hash must diverge.
    let mut state = synthesize_kernel_state(&s.ops);
    state.arbiter.inflight = state.arbiter.inflight.wrapping_add(1);
    assert_ne!(state.projection_hash(), h1,
               "projection_hash failed to detect inflight mutation");
}

#[test]
fn determinism_receipt_is_byte_stable() {
    let r1 = compute_receipt();
    let r2 = compute_receipt();
    assert_eq!(r1, r2, "determinism receipt not byte-stable across runs");
    assert_eq!(r1.kernel_trace_hash.len(), 32);
}

#[test]
fn off_by_one_mutation_is_detected() {
    // Take a non-trivial scenario.
    let s = &build_corpus()[10];
    let mut state = synthesize_kernel_state(&s.ops);

    // Tampering with inflight must break verify.
    state.arbiter.inflight = state.arbiter.inflight.wrapping_add(1);
    assert!(!verify_kernel_equivalence(&state, &s.ops),
            "mutated inflight not detected");

    // Reset, then tamper with ahtc_k closure.
    let mut state2 = synthesize_kernel_state(&s.ops);
    state2.ahtc_k.fold_hits = state2.ahtc_k.fold_hits.wrapping_add(1);
    assert!(!verify_kernel_equivalence(&state2, &s.ops),
            "broken ahtc_k closure not detected");

    // Reset, then tamper with audit live_seq below total ops.
    let mut state3 = synthesize_kernel_state(&s.ops);
    state3.audit.live_seq = 0;
    if !s.ops.is_empty() {
        assert!(!verify_kernel_equivalence(&state3, &s.ops),
                "audit live_seq violation not detected");
    }
}

#[test]
fn receipt_distinct_axes_have_distinct_hashes() {
    let r: DeterminismReceipt = compute_receipt();
    assert_ne!(r.kernel_trace_hash,    r.arbiter_trace_hash);
    assert_ne!(r.kernel_trace_hash,    r.smp_recompose_hash);
    assert_ne!(r.kernel_trace_hash,    r.ahtc_fold_expand_hash);
    assert_ne!(r.arbiter_trace_hash,   r.smp_recompose_hash);
    assert_ne!(r.arbiter_trace_hash,   r.ahtc_fold_expand_hash);
    assert_ne!(r.smp_recompose_hash,   r.ahtc_fold_expand_hash);
}

#[test]
fn execution_state_default_projects_to_empty_lockstate() {
    let s = ExecutionState::default();
    let l = s.project_arbiter();
    assert_eq!(l.inflight, 0);
    assert!(l.holder_authority.is_none());
}
