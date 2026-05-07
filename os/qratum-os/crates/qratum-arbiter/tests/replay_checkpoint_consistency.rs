//! INV-U — replay checkpoint restore reproduces identical merkle root.

use qratum_arbiter::receipt_merkle::{ReceiptMerkle, leaf_for};
use qratum_arbiter::replay_checkpoint::CheckpointStore;

#[test]
fn save_then_restore_matches() {
    let mut m = ReceiptMerkle::new();
    for i in 0u32..16 {
        m.append(leaf_for(&i.to_le_bytes()));
    }
    let mut store = CheckpointStore::default();
    let seq = store.save(1, &m, 100, 0xCAFEBABE);

    // Restore: rebuild from same leaves.
    let mut restored = ReceiptMerkle::new();
    for i in 0u32..16 { restored.append(leaf_for(&i.to_le_bytes())); }

    assert!(store.restore_matches(seq, &restored));
}

#[test]
fn restore_with_perturbed_leaf_fails() {
    let mut m = ReceiptMerkle::new();
    for i in 0u32..8 { m.append(leaf_for(&i.to_le_bytes())); }
    let mut store = CheckpointStore::default();
    let seq = store.save(1, &m, 0, 0);

    let mut tampered = ReceiptMerkle::new();
    for i in 0u32..8 {
        let v = if i == 3 { 999 } else { i };
        tampered.append(leaf_for(&v.to_le_bytes()));
    }
    assert!(!store.restore_matches(seq, &tampered));
}
