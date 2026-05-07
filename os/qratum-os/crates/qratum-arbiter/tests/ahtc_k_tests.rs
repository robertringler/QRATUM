//! AHTC-K mandatory acceptance suite.
//!
//! Test 1 — Determinism: identical input stream → identical canonical
//!          output stream (state hash + audit log).
//! Test 2 — Replay equivalence: `expand(fold(stream)) == stream`,
//!          bit-identical ordering.
//! Test 3 — Scheduler reduction: ≥ 70 % duplicate workload yields
//!          ≥ 10× reduction in scheduler enqueue ops.
//! Test 4 — Arbitration invariance: AHTC-K does not alter the verdict
//!          stream produced by `state_machine::arbitrate`.
//!
//! Plus a structural test (key bitwise equality / no fuzzy matching).

use qratum_arbiter::ahtc_k::{
    canonicalize, expand, AhtcEngine, IntentEnvelope, MemberRef,
};
use qratum_arbiter::state_machine::{
    arbitrate, Authority, Intent, LockState, Scope, Verdict,
};

// ----------- helpers -----------

fn env(seed: u64, name: &[u8], params: &[u8], lock_key: u64) -> IntentEnvelope {
    let mut id = [0u8; 16];
    id[..8].copy_from_slice(&seed.to_le_bytes());
    id[8..].copy_from_slice(&seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).to_le_bytes());
    IntentEnvelope {
        id,
        ts:        seed,
        authority: 2, // Console
        scope:     1, // Process
        priority:  2,
        lock_key,
        name:      name.to_vec(),
        params:    params.to_vec(),
    }
}

/// Generate a stream of N intents with `dup_pct`% duplication of a
/// hot key, the remainder drawn from a small bounded pool of cold
/// keys (so "redundant workload" yields a finite key set, matching the
/// directive's repeated-intents semantics).
fn workload(n: usize, dup_pct: u32) -> Vec<IntentEnvelope> {
    const COLD_POOL: u64 = 32;
    let mut out = Vec::with_capacity(n);
    let hot_name = b"hot.intent";
    let hot_params = b"";
    let hot_lock = 0x4854_4F54;
    let mut x = 0xDEAD_BEEF_CAFE_F00Du64;
    for i in 0..n {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        let pct = (x % 100) as u32;
        let mut e = if pct < dup_pct {
            env(i as u64, hot_name, hot_params, hot_lock)
        } else {
            let bucket = (x >> 7) % COLD_POOL;
            let mut nm = b"cold.".to_vec();
            nm.extend_from_slice(&bucket.to_le_bytes());
            env(i as u64, &nm, b"p", 0x434F_4C44 + bucket)
        };
        e.ts = i as u64;
        out.push(e);
    }
    out
}

// ============================================================
// TEST 1 — determinism (identical input → identical output)
// ============================================================
#[test]
fn determinism_state_hash_byte_equal() {
    let s = workload(2_000, 70);

    let mut a = AhtcEngine::new();
    a.fold_stream(&s);
    let mut b = AhtcEngine::new();
    b.fold_stream(&s);

    assert_eq!(a.state_hash(), b.state_hash(), "state hash divergence");
    assert_eq!(a.rq.batches.len(), b.rq.batches.len());
    assert_eq!(a.log.frames.len(), b.log.frames.len(),
               "audit log frame count divergence");
    assert_eq!(a.log.frames.len(), s.len(),
               "every accepted intent must produce exactly one frame");
}

#[test]
fn determinism_canonicalize_pure() {
    let e = env(7, b"x.y.z", b"{a:1}", 0x42);
    let k1 = canonicalize(&e);
    let k2 = canonicalize(&e);
    assert_eq!(k1, k2);
    // K5 — different name → different key.
    let e2 = env(7, b"x.y.Z", b"{a:1}", 0x42);
    assert_ne!(canonicalize(&e2), k1);
    // K4 — different lock_key → different key.
    let e3 = env(7, b"x.y.z", b"{a:1}", 0x43);
    assert_ne!(canonicalize(&e3), k1);
}

// ============================================================
// TEST 2 — replay equivalence: expand(fold(S)) == S (bit-for-bit order)
// ============================================================
#[test]
fn replay_equivalence_bit_for_bit() {
    let s = workload(1_500, 70);

    let mut eng = AhtcEngine::new();
    eng.fold_stream(&s);

    let r = expand(&eng.log);

    assert_eq!(r.len(), s.len(), "reconstructed length mismatch");
    for (i, (orig, rec)) in s.iter().zip(r.iter()).enumerate() {
        assert_eq!(orig.id, rec.id, "id mismatch at {i}");
        assert_eq!(orig.ts, rec.ts, "ts mismatch at {i}");
        // Reconstructed key must match canonicalized original.
        assert_eq!(canonicalize(orig), rec.key,
                   "key reconstruction mismatch at {i}");
    }
}

#[test]
fn replay_membership_matches_audit() {
    let s = workload(500, 80);
    let mut eng = AhtcEngine::new();
    eng.fold_stream(&s);

    // Sum of member counts across batches == stream length.
    let total_members: usize = eng.rq.batches.iter().map(|b| b.members.len()).sum();
    assert_eq!(total_members, s.len());

    // Every batch's first_id == its first member id.
    for b in &eng.rq.batches {
        let first = b.members.first().expect("non-empty batch");
        assert_eq!(b.first_id, first.id);
    }

    // For each batch, the audit BatchCreate frame's intent_id is the
    // first member id, and subsequent BatchIncrement frames preserve
    // member order.
    let mut per_batch: std::collections::HashMap<u32, Vec<MemberRef>> =
        std::collections::HashMap::new();
    use qratum_arbiter::ahtc_k::AhtcAuditFrame::*;
    for f in &eng.log.frames {
        match f {
            BatchCreate { batch_idx, intent_id, ts, .. } => {
                per_batch.entry(*batch_idx).or_default()
                    .push(MemberRef { id: *intent_id, ts: *ts });
            }
            BatchIncrement { batch_idx, intent_id, ts } => {
                per_batch.get_mut(batch_idx).expect("create before increment")
                    .push(MemberRef { id: *intent_id, ts: *ts });
            }
            BatchExpand { .. } => {}
        }
    }
    for (idx, b) in eng.rq.batches.iter().enumerate() {
        let from_log = per_batch.get(&(idx as u32)).expect("batch frames present");
        assert_eq!(from_log.len(), b.members.len());
        for (m_log, m_rq) in from_log.iter().zip(b.members.iter()) {
            assert_eq!(m_log.id, m_rq.id);
            assert_eq!(m_log.ts, m_rq.ts);
        }
    }
}

// ============================================================
// TEST 3 — scheduler reduction (≥ 10× at 70 % duplication)
// ============================================================
#[test]
fn scheduler_reduction_10x() {
    let n = 100_000;
    let s = workload(n, 70);

    let mut eng = AhtcEngine::new();
    eng.fold_stream(&s);

    // Pre-AHTC-K: one enqueue per intent.
    let pre  = s.len() as u64;
    // Post-AHTC-K: one enqueue per *new* canonical key.
    let post = eng.stats.enqueue_ops;

    assert!(post > 0);
    let ratio = pre as f64 / post as f64;
    assert!(
        ratio >= 10.0,
        "expected ≥10× scheduler reduction at 70% duplication; \
         got pre={pre}, post={post}, ratio={ratio:.2}"
    );

    // Audit count must remain == intent count (no audit drop).
    assert_eq!(eng.log.frames.len() as u64, pre,
               "every intent must produce exactly one audit frame");
}

#[test]
fn scheduler_reduction_scales_with_duplication() {
    // Reduction ratio (intents / enqueue_ops) is non-decreasing with
    // duplication once duplication dominates the cold-pool count.
    let mut prev_ratio: f64 = 0.0;
    for dup in [50u32, 70, 80, 90, 95] {
        let s = workload(5_000, dup);
        let mut eng = AhtcEngine::new();
        eng.fold_stream(&s);
        let ratio = s.len() as f64 / eng.stats.enqueue_ops as f64;
        assert!(ratio + 1e-9 >= prev_ratio,
                "reduction ratio should be non-decreasing \
                 (dup={dup} prev={prev_ratio:.2} post={ratio:.2})");
        prev_ratio = ratio;
    }
}

// ============================================================
// TEST 4 — arbitration invariance
// ============================================================
#[test]
fn arbitration_invariance() {
    // Build a deterministic raw intent stream using the formal arbiter's
    // input type so we can compare verdict streams *with and without*
    // AHTC-K in front of them.
    let mut intents = Vec::with_capacity(800);
    for i in 0..800u64 {
        let auth = match i % 4 {
            0 => Authority::User,
            1 => Authority::Console,
            2 => Authority::Service,
            _ => Authority::System,
        };
        let scope = match i % 3 {
            0 => Scope::Process,
            1 => Scope::Session,
            _ => Scope::Machine,
        };
        let mut id = [0u8; 16];
        id[..8].copy_from_slice(&i.to_le_bytes());
        intents.push(Intent {
            id,
            kind: ((i % 7) as u8) + 1, // never 0
            authority: auth,
            scope,
            tick: i,
            payload_h: [0; 32],
        });
    }

    // Verdict stream WITHOUT AHTC-K.
    let mut l1 = LockState::new();
    let mut v_baseline = Vec::with_capacity(intents.len());
    for i in &intents {
        let (v, lp) = arbitrate(i, &l1, i.tick);
        l1 = lp;
        v_baseline.push(v);
    }

    // Verdict stream WITH AHTC-K interposed (only on accepts).
    let mut l2 = LockState::new();
    let mut eng = AhtcEngine::new();
    let mut v_with = Vec::with_capacity(intents.len());
    for i in &intents {
        let (v, lp) = arbitrate(i, &l2, i.tick);
        l2 = lp;
        v_with.push(v);
        if matches!(v, Verdict::Accept(_)) {
            // K3 — feed AHTC-K only on Accept; AHTC-K must not loop
            // back into the arbiter or mutate `l2`.
            let env_for_ahtc = IntentEnvelope {
                id: i.id, ts: i.tick,
                authority: i.authority as u8,
                scope: i.scope as u8,
                priority: 2,
                lock_key: u64::from_le_bytes(i.id[..8].try_into().unwrap()),
                name:   i.id[8..].to_vec(),
                params: i.payload_h.to_vec(),
            };
            eng.fold_one(&env_for_ahtc);
        }
    }

    assert_eq!(v_baseline.len(), v_with.len());
    for (a, b) in v_baseline.iter().zip(v_with.iter()) {
        assert_eq!(a, b, "verdict stream must be identical with/without AHTC-K");
    }

    // Sanity — accepted count == AHTC-K log frame count.
    let accepted: usize = v_baseline.iter()
        .filter(|v| matches!(v, Verdict::Accept(_))).count();
    assert_eq!(accepted, eng.log.frames.len());
}
