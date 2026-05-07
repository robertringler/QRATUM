//! `fold_or_insert` — the one-and-only mutator of the AHTC-K run queue.

use super::batch::{IntentBatch, RunQueue};
use super::key::CanonicalKey;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FoldOutcome {
    /// New canonical key — appended at the tail.
    Created     { batch_idx: u32 },
    /// Existing key — count incremented, last_ts updated.
    Incremented { batch_idx: u32 },
    /// Capacity exhausted. Must trip an invariant — never silent.
    QueueFull,
}

/// Fold an arbitrated intent into the run queue.
///
/// Rules (`spec/ahtc_k.md` §4):
///   * exact key match → increment `count`, update `last_ts`
///   * else            → append at the tail (no reordering)
///   * never           → cross-key merge or evict
///
/// `intent_id` and `ts` are caller-provided; AHTC-K does not mint them.
pub fn fold_or_insert(
    rq: &mut RunQueue,
    key: CanonicalKey,
    intent_id: [u8; 16],
    ts: u64,
) -> FoldOutcome {
    if let Some(idx) = rq.find_exact(&key) {
        // K5: exact bitwise match only (find_exact uses `==`).
        let b = rq.get_mut(idx).expect("idx in bounds");
        b.count = b.count.saturating_add(1);
        b.last_ts = ts;
        return FoldOutcome::Incremented { batch_idx: idx as u32 };
    }

    let new_b = IntentBatch {
        key,
        count: 1,
        first_ts: ts,
        last_ts: ts,
        first_id: intent_id,
    };
    match rq.push(new_b) {
        Some(idx) => FoldOutcome::Created { batch_idx: idx as u32 },
        None      => FoldOutcome::QueueFull,
    }
}
