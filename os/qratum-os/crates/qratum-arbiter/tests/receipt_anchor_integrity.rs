//! INV-Z — receipt anchors are append-only.

use qratum_arbiter::receipt_merkle::{ReceiptMerkle, leaf_for};
use qratum_arbiter::replay_checkpoint::CheckpointStore;

#[test]
fn anchors_are_append_only() {
    let mut m = ReceiptMerkle::new();
    let mut store = CheckpointStore::default();
    for i in 0u64..32 {
        m.append(leaf_for(&i.to_le_bytes()));
        store.save(i, &m, i, i ^ 0xA5);
    }
    // Verify no anchor gets overwritten — all seq values strictly
    // increasing, items vector length matches saves.
    assert_eq!(store.items.len(), 32);
    let seqs: Vec<u64> = store.items.iter().map(|c| c.seq).collect();
    let mut sorted = seqs.clone(); sorted.sort();
    assert_eq!(seqs, sorted);
    for w in seqs.windows(2) { assert!(w[1] > w[0]); }
}

#[test]
fn earlier_anchors_unmodified_by_later_appends() {
    let mut m = ReceiptMerkle::new();
    let mut store = CheckpointStore::default();
    m.append(leaf_for(b"first"));
    let s0 = store.save(0, &m, 0, 0);
    let r0 = store.load(s0).unwrap().merkle_root;

    for i in 0..10 {
        m.append(leaf_for(&(i as u64).to_le_bytes()));
        store.save(i + 1, &m, i + 1, 0);
    }

    assert_eq!(store.load(s0).unwrap().merkle_root, r0);
}
