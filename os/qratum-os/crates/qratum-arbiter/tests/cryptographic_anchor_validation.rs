//! INV-Z — cryptographic anchor (Blake3) validation.

use qratum_arbiter::receipt_merkle::{ReceiptMerkle, leaf_for};

#[test]
fn root_is_blake3_pure_function_of_leaves() {
    let mut a = ReceiptMerkle::new();
    let mut b = ReceiptMerkle::new();
    for i in 0u32..100 {
        a.append(leaf_for(&i.to_le_bytes()));
        b.append(leaf_for(&i.to_le_bytes()));
    }
    assert_eq!(a.root(), b.root());
}

#[test]
fn one_byte_perturbation_changes_root() {
    let mut a = ReceiptMerkle::new();
    let mut b = ReceiptMerkle::new();
    for i in 0u32..100 { a.append(leaf_for(&i.to_le_bytes())); }
    for i in 0u32..100 {
        let v = if i == 50 { 50_001 } else { i };
        b.append(leaf_for(&v.to_le_bytes()));
    }
    assert_ne!(a.root(), b.root());
}
