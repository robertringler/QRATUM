//! P0.6 — Host-side receipt Merkle tree using real Blake3.
//!
//! The kernel's `runtime_model::receipt_merkle` produces 32-byte
//! roots using the kernel's deterministic 32-byte digest. The host
//! re-hashes the same leaf bytes with **real Blake3** and exposes
//! both roots for cross-checking. INV-X requires the host root be
//! deterministic for identical workloads.

use blake3::Hasher;

#[derive(Clone, Debug, Default)]
pub struct ReceiptMerkle {
    pub leaves: Vec<[u8; 32]>,
}

impl ReceiptMerkle {
    pub fn new() -> Self { Self { leaves: Vec::new() } }

    pub fn append(&mut self, leaf: [u8; 32]) { self.leaves.push(leaf); }

    pub fn root(&self) -> [u8; 32] {
        if self.leaves.is_empty() { return [0u8; 32]; }
        let mut layer: Vec<[u8; 32]> = self.leaves.clone();
        while layer.len() > 1 {
            let mut next = Vec::with_capacity((layer.len() + 1) / 2);
            let mut i = 0;
            while i < layer.len() {
                let l = layer[i];
                let r = if i + 1 < layer.len() { layer[i + 1] } else { l };
                let mut h = Hasher::new();
                h.update(&l);
                h.update(&r);
                next.push(*h.finalize().as_bytes());
                i += 2;
            }
            layer = next;
        }
        layer[0]
    }
}

pub fn leaf_for(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

pub fn roots_equal(a: &ReceiptMerkle, b: &ReceiptMerkle) -> bool {
    a.root() == b.root()
}
