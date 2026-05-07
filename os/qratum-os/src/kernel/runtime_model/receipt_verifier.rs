//! P0.6 — Receipt verifier. INV-X: identical workload → identical
//! Merkle root.

#![allow(dead_code)]

use super::receipt_merkle::MerkleTree;

#[inline]
pub fn roots_equal(a: &MerkleTree, b: &MerkleTree) -> bool { a.root == b.root }

pub fn verify_byte_identical(a: &MerkleTree, b: &MerkleTree) -> bool {
    if a.leaf_count != b.leaf_count { return false; }
    for i in 0..(a.leaf_count as usize) {
        if a.leaves[i] != b.leaves[i] { return false; }
    }
    a.root == b.root
}
