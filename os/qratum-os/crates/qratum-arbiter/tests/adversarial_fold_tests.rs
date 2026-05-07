//! P0.4 — Adversarial fold corpus (INV-N).
//!
//! AHTC-K must NEVER fold semantically distinct workloads. These
//! workloads attack the canonical-key surface from five angles.

use qratum_arbiter::ahtc_k::{canonicalize, AhtcEngine, FoldOutcome, IntentEnvelope};

fn env(name: &str, params: &[u8], authority: u8, priority: u8, lock_key: u64,
       id: u64) -> IntentEnvelope {
    let mut idb = [0u8; 16];
    idb[..8].copy_from_slice(&id.to_le_bytes());
    IntentEnvelope {
        id: idb, ts: id,
        authority, scope: 1, priority,
        lock_key, name: name.as_bytes().to_vec(),
        params: params.to_vec(),
    }
}

#[test]
fn near_collision_streams_yield_distinct_canonical_keys() {
    let a = env("svc.x", b"alpha",  2, 2, 0x1111, 1);
    let b = env("svc.x", b"alphaa", 2, 2, 0x1111, 2);
    assert_ne!(canonicalize(&a), canonicalize(&b),
        "INV-N: distinct params bytes must yield distinct canonical keys");
}

#[test]
fn authority_escalation_drift_does_not_collapse() {
    let lo = env("ctl.do", b"go", 1, 2, 0xAA, 1);
    let hi = env("ctl.do", b"go", 4, 2, 0xAA, 2);
    assert_ne!(canonicalize(&lo), canonicalize(&hi));
}

#[test]
fn priority_drift_does_not_collapse() {
    let a = env("p", b"", 2, 1, 0xAA, 1);
    let b = env("p", b"", 2, 4, 0xAA, 2);
    assert_ne!(canonicalize(&a), canonicalize(&b));
}

#[test]
fn lock_key_drift_does_not_collapse() {
    let a = env("k", b"", 2, 2, 0xAAAA, 1);
    let b = env("k", b"", 2, 2, 0xBBBB, 2);
    assert_ne!(canonicalize(&a), canonicalize(&b));
}

#[test]
fn timing_skew_alone_must_fold() {
    let a = env("noop", b"", 2, 2, 0xBEEF, 1);
    let b = env("noop", b"", 2, 2, 0xBEEF, 2);
    let mut eng = AhtcEngine::new();
    let r1 = eng.fold_one(&a);
    let r2 = eng.fold_one(&b);
    assert!(matches!(r1, FoldOutcome::Created { .. }));
    assert!(matches!(r2, FoldOutcome::Incremented { .. }));
    assert_eq!(eng.rq.batches.len(), 1);
}

#[test]
fn partial_order_perturbations_yield_identical_lattice() {
    let mk = |id: u64| env("svc", b"", 2, 2, 0x100 + id, id);
    let stream_a: Vec<_> = (1..=8u64).map(mk).collect();
    let mut stream_b = stream_a.clone(); stream_b.reverse();

    let mut a = AhtcEngine::new(); a.fold_stream(&stream_a);
    let mut b = AhtcEngine::new(); b.fold_stream(&stream_b);

    let mut ka: Vec<_> = a.rq.batches.iter().map(|x| x.key).collect();
    let mut kb: Vec<_> = b.rq.batches.iter().map(|x| x.key).collect();
    ka.sort_by(|x, y| (x.lock_key, x.name_hash).cmp(&(y.lock_key, y.name_hash)));
    kb.sort_by(|x, y| (x.lock_key, x.name_hash).cmp(&(y.lock_key, y.name_hash)));
    assert_eq!(ka, kb);
    assert_eq!(a.log.frames.len(), b.log.frames.len());
}

#[test]
fn scheduler_race_simulation_preserves_audit_frame_count() {
    let dup = env("dup", b"", 2, 2, 0xCAFE, 0);
    let mut interleaved: Vec<IntentEnvelope> = Vec::new();
    for i in 0..50u64 {
        let mut e = dup.clone();
        let mut idb = [0u8; 16]; idb[..8].copy_from_slice(&i.to_le_bytes());
        e.id = idb; e.ts = i;
        interleaved.push(e);
    }
    for i in 50..100u64 {
        interleaved.push(env("alt", b"", 2, 2, 0x1234, i));
    }
    let mut sequential = interleaved.clone();
    sequential.sort_by_key(|e| e.lock_key);

    let mut a = AhtcEngine::new(); a.fold_stream(&interleaved);
    let mut b = AhtcEngine::new(); b.fold_stream(&sequential);
    assert_eq!(a.log.frames.len(), b.log.frames.len());
    assert_eq!(a.rq.batches.len(), b.rq.batches.len());
}
