//! AHTC-K — Anti-Holographic Tensor Canonical Kernel Layer.
//!
//! Pure deterministic canonicalization + batching. Sits **between**
//! `intent::submit` (arbiter) and the future intent-driven scheduler
//! enqueue path. AHTC-K never sees a denied intent; the arbiter's
//! verdict is computed first and recorded in audit unchanged.
//!
//! See `spec/ahtc_k.md` for the formal contract. The five hard
//! constraints (K1..K5) are enforced by:
//!
//!   * K1 — `canonicalize` is total + pure; hashing is FNV-1a-64
//!     (`XXX-blake3` placeholder, parity-tested on host).
//!   * K2 — `BatchCreate` / `BatchIncrement` audit frames carry the
//!     original `intent_id`; `replay::expand` rebuilds `S` exactly.
//!   * K3 — call order in `submit` is arbiter → audit-verdict →
//!     canonicalize → fold; arbiter inputs are never mutated.
//!   * K4 — `lock_key` is supplied by the caller, not derived.
//!   * K5 — `CanonicalKey` `PartialEq` is bitwise.

#![allow(dead_code)]

pub mod batch;
pub mod canonical;
pub mod fold;
pub mod key;
pub mod replay;
pub mod equivalence_boundary;
pub mod canonical_proof;
pub mod divergence_detector;

pub use batch::{RunQueue, MAX_BATCHES};
pub use canonical::{canonicalize, CanonInputs};
pub use fold::{fold_or_insert, FoldOutcome};
pub use key::CanonicalKey;

use crate::shared::{IntentEnvelope, ResultEnvelope, Verdict};
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use super::intent;

/// Global AHTC-K run queue (single-CPU; per-CPU split lands with SMP
/// bring-up — see `spec/smp_model.md`).
static RUNQUEUE: spin::Mutex<RunQueue> = spin::Mutex::new(RunQueue::new());

/// Counters exposed for tests / qsh introspection.
static FOLD_HITS:    AtomicU64 = AtomicU64::new(0);
static FOLD_INSERTS: AtomicU64 = AtomicU64::new(0);
static SUBMITS:      AtomicU64 = AtomicU64::new(0);
static ENQUEUE_OPS:  AtomicU64 = AtomicU64::new(0);

/// Front-door for the canonicalized submit path.
///
/// Order is fixed by K3 — arbiter first, AHTC-K second. The arbiter
/// receives the **raw** envelope; AHTC-K receives the same raw envelope
/// only after an Accept verdict.
pub fn submit(env: &IntentEnvelope, lock_key: u64) -> ResultEnvelope {
    SUBMITS.fetch_add(1, Ordering::Relaxed);
    super::profiler::trace_event(super::profiler::TraceKind::IntentSubmit,
                                 env.authority as u64);

    // K3: arbitration runs first, in isolation, on raw envelope.
    let verdict = intent::submit(env);
    super::profiler::trace_event(super::profiler::TraceKind::ArbiterDecision,
                                 verdict.verdict as u64);

    if verdict.verdict != Verdict::Accepted as u8 {
        // K3: denied intents do not enter AHTC-K. No batch frames.
        return verdict;
    }

    // K4 / K5: lock_key is an input, not derived. Hashes are bytewise.
    let key = canonicalize(&CanonInputs {
        authority: env.authority,
        scope:     env.scope,
        priority:  env.priority,
        name:      env.name_str().as_bytes(),
        params:    &[], // P0.3 envelope has no params blob; empty bytes.
        lock_key,
    });

    let mut rq = RUNQUEUE.lock();
    let outcome = fold_or_insert(&mut rq, key, env.id, env.ts);
    drop(rq);

    match outcome {
        FoldOutcome::Created { batch_idx } => {
            FOLD_INSERTS.fetch_add(1, Ordering::Relaxed);
            ENQUEUE_OPS.fetch_add(1, Ordering::Relaxed);
            super::profiler::trace_event(super::profiler::TraceKind::AhtcFold,
                                         batch_idx as u64);
            audit_batch_create(&key, &env.id, env.ts, batch_idx);
            super::profiler::trace_event(super::profiler::TraceKind::AuditWrite, 1);
        }
        FoldOutcome::Incremented { batch_idx } => {
            FOLD_HITS.fetch_add(1, Ordering::Relaxed);
            super::profiler::trace_event(super::profiler::TraceKind::AhtcFold,
                                         batch_idx as u64);
            audit_batch_increment(&env.id, env.ts, batch_idx);
            super::profiler::trace_event(super::profiler::TraceKind::AuditWrite, 1);
        }
        FoldOutcome::QueueFull => {
            // Failure mode: invariants::trip — must not be silent.
            super::invariants::trip(
                super::invariants::InvariantCode::AuditDrop,
                b"ahtc_k:rq_full",
            );
        }
    }

    verdict
}

#[derive(Default, Copy, Clone)]
pub struct AhtcStats {
    pub submits:        u64,
    pub fold_hits:      u64,
    pub fold_inserts:   u64,
    pub enqueue_ops:    u64,
    pub queue_len:      u32,
}

pub fn stats() -> AhtcStats {
    let qlen = RUNQUEUE.lock().len() as u32;
    AhtcStats {
        submits:      SUBMITS.load(Ordering::Relaxed),
        fold_hits:    FOLD_HITS.load(Ordering::Relaxed),
        fold_inserts: FOLD_INSERTS.load(Ordering::Relaxed),
        enqueue_ops:  ENQUEUE_OPS.load(Ordering::Relaxed),
        queue_len:    qlen,
    }
}

/// Reset state — test/QSH support only. Not for production paths.
pub fn reset() {
    let mut rq = RUNQUEUE.lock();
    rq.clear();
    FOLD_HITS.store(0, Ordering::Relaxed);
    FOLD_INSERTS.store(0, Ordering::Relaxed);
    SUBMITS.store(0, Ordering::Relaxed);
    ENQUEUE_OPS.store(0, Ordering::Relaxed);
}

/// Helper used by `runtime_model::execution_lattice` so the kernel
/// lattice can read flushed-segment count without a hard dependency
/// loop. Returns 0 if `audit_persist` is not yet initialised.
pub fn audit_persist_flushed_or_zero() -> u64 {
    super::audit_persist::flushed_count()
}

// ===========================================================
// AHTC-K formal-constraint gate (P0.3 Task 3)
// ===========================================================

/// Reasons a gate check can refuse — every variant trips an invariant.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum GateFailure {
    NotReversible,
    AltersArbitration,
    NonDeterministicFold,
}

/// Property 1: counter closure. Every accepted submit produced exactly
/// one fold operation (insert OR hit), and `enqueue_ops` equals the
/// number of distinct keys (= `fold_inserts`).
fn is_reversible(s: &AhtcStats, accepted: u64) -> bool {
    s.fold_hits + s.fold_inserts == accepted
        && s.enqueue_ops == s.fold_inserts
}

/// Property 2: AHTC-K must not have raised any silent invariant.
/// `invariants::trip` panics on entry, so the only failure mode that
/// could leak through is an *uncounted* fold — captured by
/// `is_reversible`. This function additionally asserts
/// `submits >= accepted` (denied intents pass through `submits` but
/// not the fold counters).
fn preserves_arbitration(s: &AhtcStats, accepted: u64) -> bool {
    s.submits >= accepted
}

/// Property 3: deterministic fold-expand. The kernel-side check is
/// structural — `queue_len == fold_inserts`. The byte-level
/// `expand(fold(S)) == S` proof lives in the host crate test
/// `ahtc_k_tests::replay_equivalence_bit_for_bit`.
fn deterministic_fold_expand(s: &AhtcStats) -> bool {
    s.queue_len as u64 == s.fold_inserts
}

/// `ahtc_k_gate` — the directive's required invariant gate. Returns
/// `Ok(())` on success; on failure trips a hard invariant (no soft mode).
///
/// `accepted` is the number of arbiter Accept verdicts observed so far.
/// Caller pulls this from `ExecutionState` (not from inside AHTC-K) so
/// the check is genuinely external.
pub fn ahtc_k_gate(accepted: u64) -> Result<(), GateFailure> {
    let s = stats();
    if !is_reversible(&s, accepted) {
        super::invariants::trip(super::invariants::InvariantCode::AuditDrop,
                                 b"ahtc_k_gate:not_reversible");
    }
    if !preserves_arbitration(&s, accepted) {
        super::invariants::trip(super::invariants::InvariantCode::StateMachineMismatch,
                                 b"ahtc_k_gate:alters_arbitration");
    }
    if !deterministic_fold_expand(&s) {
        super::invariants::trip(super::invariants::InvariantCode::StateMachineMismatch,
                                 b"ahtc_k_gate:nondet_fold");
    }
    Ok(())
}

// -- audit emission --------------------------------------------------------

#[inline]
fn audit_batch_create(k: &CanonicalKey, id: &[u8; 16], ts: u64, batch_idx: u32) {
    let mut p = [0u8; 62];
    p[..16].copy_from_slice(id);
    p[16] = k.authority;
    p[17] = k.scope;
    p[18] = k.priority;
    p[19] = 0;
    p[20..28].copy_from_slice(&k.name_hash.to_le_bytes());
    p[28..36].copy_from_slice(&k.params_hash.to_le_bytes());
    p[36..44].copy_from_slice(&k.lock_key.to_le_bytes());
    p[44..52].copy_from_slice(&ts.to_le_bytes());
    p[52..56].copy_from_slice(&batch_idx.to_le_bytes());
    super::audit::append(super::audit::FrameKind::BatchCreate, &p[..56]);
}

#[inline]
fn audit_batch_increment(id: &[u8; 16], ts: u64, batch_idx: u32) {
    let mut p = [0u8; 62];
    p[..16].copy_from_slice(id);
    p[16..20].copy_from_slice(&batch_idx.to_le_bytes());
    p[20..28].copy_from_slice(&ts.to_le_bytes());
    super::audit::append(super::audit::FrameKind::BatchIncrement, &p[..28]);
}

// -- compile-time invariants ----------------------------------------------

const _: () = {
    // K-constants are part of the ABI surface.
    assert!(MAX_BATCHES <= u32::MAX as usize);
    assert!(core::mem::size_of::<CanonicalKey>() == 32);
};

#[allow(dead_code)]
static _AHTC_BATCHES_HINT: AtomicU32 = AtomicU32::new(MAX_BATCHES as u32);
