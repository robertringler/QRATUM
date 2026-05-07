//! P0.5 — Receipt chain integrity.

use qratum_arbiter::execution_receipt::{ExecutionReceipt, ReceiptChain, TransitionKind};

fn r(kind: TransitionKind, key: u64) -> ExecutionReceipt {
    ExecutionReceipt {
        seq: 0, kind: kind as u8,
        canonical_key_lo: key, canonical_key_hi: key.rotate_left(7),
        arbitration_link: key ^ 0xA1, migration_link: key ^ 0xB2, replay_link: 0,
    }
}

#[test]
fn empty_chain_holds() {
    let c = ReceiptChain::new();
    assert!(c.integrity_holds());
}

#[test]
fn append_changes_chain_hash() {
    let mut c = ReceiptChain::new();
    let h0 = c.chain_hash;
    c.append(r(TransitionKind::Submit, 0x11));
    assert_ne!(c.chain_hash, h0);
}

#[test]
fn integrity_holds_after_many_appends() {
    let mut c = ReceiptChain::new();
    for i in 0u64..256 {
        let kind = match i & 3 {
            0 => TransitionKind::Submit,
            1 => TransitionKind::Fold,
            2 => TransitionKind::Migrate,
            _ => TransitionKind::Replay,
        };
        c.append(r(kind, i.wrapping_mul(0x9E37_79B9_7F4A_7C15)));
    }
    assert!(c.integrity_holds());
}
