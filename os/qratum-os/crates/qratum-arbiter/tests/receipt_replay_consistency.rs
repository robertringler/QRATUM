//! P0.5 — Receipt chain replay equality.

use qratum_arbiter::execution_receipt::{ExecutionReceipt, ReceiptChain, TransitionKind};

#[test]
fn replay_yields_same_final_hash() {
    let mut c = ReceiptChain::new();
    for i in 0u64..128 {
        c.append(ExecutionReceipt {
            seq: 0, kind: (i & 3) as u8,
            canonical_key_lo: i, canonical_key_hi: i << 4,
            arbitration_link: i.wrapping_mul(7),
            migration_link: i.wrapping_mul(11),
            replay_link: 0,
        });
    }
    let original = c.chain_hash;
    let replayed = c.replay();
    assert_eq!(original, replayed);
}

#[test]
fn out_of_order_append_diverges() {
    let mut a = ReceiptChain::new();
    let mut b = ReceiptChain::new();
    let r1 = ExecutionReceipt { seq: 0, kind: 0, canonical_key_lo: 1, canonical_key_hi: 0,
        arbitration_link: 0xAA, migration_link: 0, replay_link: 0 };
    let r2 = ExecutionReceipt { seq: 0, kind: 1, canonical_key_lo: 2, canonical_key_hi: 0,
        arbitration_link: 0xBB, migration_link: 0, replay_link: 0 };
    a.append(r1); a.append(r2);
    b.append(r2); b.append(r1);
    assert_ne!(a.chain_hash, b.chain_hash);
}
