//! INV-X — receipt Merkle roots deterministic for identical workloads.

use qratum_arbiter::receipt_merkle::{ReceiptMerkle, leaf_for, roots_equal};

fn build(n: usize) -> ReceiptMerkle {
    let mut m = ReceiptMerkle::new();
    for i in 0..n {
        m.append(leaf_for(&(i as u64).to_le_bytes()));
    }
    m
}

#[test]
fn root_is_stable_across_runs() {
    let a = build(64);
    let b = build(64);
    assert!(roots_equal(&a, &b));
}

#[test]
fn root_changes_with_perturbation() {
    let a = build(64);
    let mut b = build(64);
    b.append(leaf_for(b"extra"));
    assert!(!roots_equal(&a, &b));
}

#[test]
fn empty_root_is_zero() {
    let m = ReceiptMerkle::new();
    assert_eq!(m.root(), [0u8; 32]);
}

#[test]
fn single_leaf_root_equals_leaf() {
    let mut m = ReceiptMerkle::new();
    m.append([7u8; 32]);
    assert_eq!(m.root(), [7u8; 32]);
}
