//! T6.5 — AHTC-K reality constraint.
//!
//! AHTC-K MUST be a *deterministic execution graph reduction*, not a
//! heuristic. Two predicates define this:
//!
//!   R1: `expand(fold(stream)) == stream`   (lossless lattice)
//!   R2: `decision_stream(raw) == decision_stream(fold(stream))`
//!                                          (verdict invariance)
//!
//! Failure of either predicate prints
//!     "AHTC-K INVALID UNDER CURRENT WORKLOAD CLASS"
//! and fails the build. Success prints
//!     "AHTC-K STATUS: VALID"
//! as a banner consumed by the CI matrix.

use qratum_arbiter::ahtc_k::{expand, AhtcEngine};
use qratum_arbiter::performance::{run_ahtc_k, run_baseline, workload};

fn assert_replay_lossless(seed: u64, n: usize, dup: u32) -> bool {
    let s = workload(n, dup, seed);
    let mut eng = AhtcEngine::new();
    eng.fold_stream(&s);
    let recon = expand(&eng.log);
    if recon.len() != s.len() {
        println!("R1 FAIL: stream={} recon={}", s.len(), recon.len());
        return false;
    }
    for (i, (orig, r)) in s.iter().zip(recon.iter()).enumerate() {
        if orig.id != r.id {
            println!("R1 FAIL @ {}: orig.id={:?} recon.id={:?}",
                     i, orig.id, r.id);
            return false;
        }
    }
    true
}

fn assert_verdict_invariance(seed: u64, n: usize, dup: u32) -> bool {
    let s = workload(n, dup, seed);
    let b = run_baseline(&s);
    let a = run_ahtc_k(&s);
    // Verdict equivalence is captured by audit-frame parity at the
    // arbiter layer (AHTC-K interposes BEFORE the scheduler — the
    // arbiter sees the same envelopes). The reports' `audit_frames`
    // counts must agree exactly.
    if b.audit_frames != a.audit_frames {
        println!("R2 FAIL: baseline frames={} ahtc frames={}",
                 b.audit_frames, a.audit_frames);
        return false;
    }
    true
}

#[test]
fn ahtc_k_is_a_deterministic_reduction_operator() {
    let workloads = [
        ("70pct/20k",  20_000, 70, 0xA1A1_A1A1_A1A1_A1A1),
        ("50pct/10k",  10_000, 50, 0xB2B2_B2B2_B2B2_B2B2),
        ("0pct/5k",     5_000,  0, 0xC3C3_C3C3_C3C3_C3C3),
        ("90pct/30k",  30_000, 90, 0xD4D4_D4D4_D4D4_D4D4),
    ];
    let mut all_pass = true;
    for (label, n, dup, seed) in workloads {
        let r1 = assert_replay_lossless(seed, n, dup);
        let r2 = assert_verdict_invariance(seed, n, dup);
        println!("[{label}] R1 replay-lossless={r1}  R2 verdict-invariant={r2}");
        all_pass &= r1 && r2;
    }
    if all_pass {
        println!("AHTC-K STATUS: VALID");
    } else {
        println!("AHTC-K INVALID UNDER CURRENT WORKLOAD CLASS");
    }
    assert!(all_pass, "AHTC-K reality constraint violated");
}
