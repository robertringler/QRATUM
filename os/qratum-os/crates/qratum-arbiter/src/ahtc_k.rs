//! Host-side reference of AHTC-K — Anti-Holographic Tensor Canonical
//! Kernel Layer.
//!
//! Authoritative for the determinism / replay-equivalence /
//! arbitration-invariance / scheduler-reduction tests. Mirrors
//! `os/qratum-os/src/kernel/ahtc_k/` modulo:
//!
//!   * Hash function — host uses real Blake3 truncated to 64-bit;
//!     kernel uses FNV-1a-64 (`XXX-blake3` placeholder, P0.3.1 swap).
//!   * Storage — host uses `Vec<IntentBatch>`; kernel uses fixed-array.
//!
//! Cross-impl hash parity is **not** required (see `spec/ahtc_k.md`
//! §2.1). Replay reconstructs from `intent_id`s in audit, not hashes.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------
// Canonical key + hashing
// ---------------------------------------------------------------------

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct CanonicalKey {
    pub authority:   u8,
    pub scope:       u8,
    pub priority:    u8,
    pub name_hash:   u64,
    pub params_hash: u64,
    pub lock_key:    u64,
}

/// Blake3 truncated to 64-bit little-endian.
pub fn blake3_64(bytes: &[u8]) -> u64 {
    let h = blake3::hash(bytes);
    let b = h.as_bytes();
    u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct IntentEnvelope {
    pub id:        [u8; 16],
    pub ts:        u64,
    pub authority: u8,
    pub scope:     u8,
    pub priority:  u8,
    /// Caller-furnished. AHTC-K never derives this (K4).
    pub lock_key:  u64,
    /// Content-addressed name + params bytes.
    pub name:      Vec<u8>,
    pub params:    Vec<u8>,
}

/// Pure canonicalization. K1, K4, K5.
pub fn canonicalize(env: &IntentEnvelope) -> CanonicalKey {
    CanonicalKey {
        authority:   env.authority,
        scope:       env.scope,
        priority:    env.priority,
        name_hash:   blake3_64(&env.name),
        params_hash: blake3_64(&env.params),
        lock_key:    env.lock_key,
    }
}

// ---------------------------------------------------------------------
// IntentBatch + RunQueue
// ---------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntentBatch {
    pub key:      CanonicalKey,
    pub count:    u32,
    pub first_ts: u64,
    pub last_ts:  u64,
    pub first_id: [u8; 16],
    /// Full membership in arrival order. K2: needed to reconstruct
    /// the original intent stream from the batch alone (in addition to
    /// the audit log path).
    pub members:  Vec<MemberRef>,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct MemberRef {
    pub id: [u8; 16],
    pub ts: u64,
}

#[derive(Default, Debug)]
pub struct RunQueue {
    pub batches: Vec<IntentBatch>,
}

// ---------------------------------------------------------------------
// Audit frames (host model)
// ---------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AhtcAuditFrame {
    BatchCreate    { batch_idx: u32, key: CanonicalKey, intent_id: [u8; 16], ts: u64 },
    BatchIncrement { batch_idx: u32, intent_id: [u8; 16], ts: u64 },
    BatchExpand    { batch_idx: u32, count: u32 },
}

#[derive(Default, Debug)]
pub struct AhtcAuditLog {
    pub frames: Vec<AhtcAuditFrame>,
}

#[derive(Default, Debug)]
pub struct AhtcStats {
    pub fold_hits:    u64,
    pub fold_inserts: u64,
    pub enqueue_ops:  u64,
}

#[derive(Default, Debug)]
pub struct AhtcEngine {
    pub rq:     RunQueue,
    pub log:    AhtcAuditLog,
    pub stats:  AhtcStats,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FoldOutcome {
    Created     { batch_idx: u32 },
    Incremented { batch_idx: u32 },
}

impl AhtcEngine {
    pub fn new() -> Self { Self::default() }

    /// Fold a single accepted intent. K3 — caller has already run the
    /// arbiter and confirmed Accept; AHTC-K does not see denials.
    pub fn fold_one(&mut self, env: &IntentEnvelope) -> FoldOutcome {
        let key = canonicalize(env);
        if let Some(idx) = self.rq.batches.iter().position(|b| b.key == key) {
            // Existing key → increment.
            let b = &mut self.rq.batches[idx];
            b.count = b.count.saturating_add(1);
            b.last_ts = env.ts;
            b.members.push(MemberRef { id: env.id, ts: env.ts });
            self.stats.fold_hits += 1;
            self.log.frames.push(AhtcAuditFrame::BatchIncrement {
                batch_idx: idx as u32,
                intent_id: env.id,
                ts:        env.ts,
            });
            FoldOutcome::Incremented { batch_idx: idx as u32 }
        } else {
            let batch_idx = self.rq.batches.len() as u32;
            let b = IntentBatch {
                key,
                count: 1,
                first_ts: env.ts,
                last_ts:  env.ts,
                first_id: env.id,
                members:  vec![MemberRef { id: env.id, ts: env.ts }],
            };
            self.rq.batches.push(b);
            self.stats.fold_inserts += 1;
            self.stats.enqueue_ops  += 1;
            self.log.frames.push(AhtcAuditFrame::BatchCreate {
                batch_idx, key, intent_id: env.id, ts: env.ts,
            });
            FoldOutcome::Created { batch_idx }
        }
    }

    pub fn fold_stream(&mut self, stream: &[IntentEnvelope]) {
        for e in stream { self.fold_one(e); }
    }

    /// Bytewise-stable digest of the canonical state (run queue + log).
    /// Used by determinism tests.
    pub fn state_hash(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        for b in &self.rq.batches {
            hasher.update(&[b.key.authority, b.key.scope, b.key.priority]);
            hasher.update(&b.key.name_hash.to_le_bytes());
            hasher.update(&b.key.params_hash.to_le_bytes());
            hasher.update(&b.key.lock_key.to_le_bytes());
            hasher.update(&b.count.to_le_bytes());
            hasher.update(&b.first_ts.to_le_bytes());
            hasher.update(&b.last_ts.to_le_bytes());
            hasher.update(&b.first_id);
            for m in &b.members {
                hasher.update(&m.id);
                hasher.update(&m.ts.to_le_bytes());
            }
        }
        *hasher.finalize().as_bytes()
    }
}

// ---------------------------------------------------------------------
// Replay — reconstruct original intent stream from audit alone (K2)
// ---------------------------------------------------------------------

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReconstructedIntent {
    pub id:        [u8; 16],
    pub ts:        u64,
    pub key:       CanonicalKey,
    pub batch_idx: u32,
}

/// Walk the audit log in order, emit `(intent_id, ts, key)` per frame.
///
/// `BatchCreate` registers the key for `batch_idx`; `BatchIncrement`
/// resolves its key by looking up the previously-seen `BatchCreate`.
/// `BatchExpand` frames are diagnostic and skipped.
pub fn expand(log: &AhtcAuditLog) -> Vec<ReconstructedIntent> {
    let mut keys: Vec<Option<CanonicalKey>> = Vec::new();
    let mut out: Vec<ReconstructedIntent> = Vec::with_capacity(log.frames.len());
    for f in &log.frames {
        match f {
            AhtcAuditFrame::BatchCreate { batch_idx, key, intent_id, ts } => {
                let bx = *batch_idx as usize;
                if keys.len() <= bx { keys.resize(bx + 1, None); }
                keys[bx] = Some(*key);
                out.push(ReconstructedIntent {
                    id: *intent_id, ts: *ts, key: *key, batch_idx: *batch_idx,
                });
            }
            AhtcAuditFrame::BatchIncrement { batch_idx, intent_id, ts } => {
                let bx = *batch_idx as usize;
                let k = keys.get(bx).and_then(|k| *k)
                    .expect("BatchIncrement before BatchCreate — invalid log");
                out.push(ReconstructedIntent {
                    id: *intent_id, ts: *ts, key: k, batch_idx: *batch_idx,
                });
            }
            AhtcAuditFrame::BatchExpand { .. } => {}
        }
    }
    out
}
