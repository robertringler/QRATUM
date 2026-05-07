//! INV-U — replay resume reproduces identical state after checkpoint.

use qratum_arbiter::receipt_merkle::{ReceiptMerkle, leaf_for};
use qratum_arbiter::replay_checkpoint::CheckpointStore;

#[test]
fn resume_from_checkpoint_yields_identical_root() {
    let mut m = ReceiptMerkle::new();
    let mut store = CheckpointStore::default();
    for i in 0u64..10 { m.append(leaf_for(&i.to_le_bytes())); }
    let seq = store.save(1, &m, 10, 0xBEEF);

    // Continue appending after checkpoint.
    for i in 10u64..20 { m.append(leaf_for(&i.to_le_bytes())); }
    let final_root_live = m.root();

    // "Resume": rebuild from leaves 0..10, restore checkpoint, then
    // append 10..20.
    let mut resumed = ReceiptMerkle::new();
    for i in 0u64..10 { resumed.append(leaf_for(&i.to_le_bytes())); }
    assert!(store.restore_matches(seq, &resumed));
    for i in 10u64..20 { resumed.append(leaf_for(&i.to_le_bytes())); }

    assert_eq!(final_root_live, resumed.root());
}
