//! P0.6 — Receipt Merkle tree (kernel side, no_std).
//!
//! Append-only binary Merkle tree over receipt link-hashes. Uses the
//! kernel's Blake3-equivalent 32-byte digest from `crypto::blake3_kernel`.

#![allow(dead_code)]

use crate::kernel::crypto::blake3_kernel as h3;

pub const CAP: usize = 1024;

#[derive(Copy, Clone, Debug)]
pub struct MerkleTree {
    pub leaves:    [[u8; 32]; CAP],
    pub leaf_count:u32,
    pub root:      [u8; 32],
}

impl MerkleTree {
    pub const fn empty() -> Self {
        Self { leaves: [[0u8; 32]; CAP], leaf_count: 0, root: [0u8; 32] }
    }

    pub fn append(&mut self, leaf: [u8; 32]) -> bool {
        if (self.leaf_count as usize) >= CAP { return false; }
        self.leaves[self.leaf_count as usize] = leaf;
        self.leaf_count += 1;
        self.root = self.compute_root();
        true
    }

    fn compute_root(&self) -> [u8; 32] {
        let n = self.leaf_count as usize;
        if n == 0 { return [0u8; 32]; }
        let mut buf: [[u8; 32]; CAP] = [[0u8; 32]; CAP];
        let mut len = n;
        for i in 0..n { buf[i] = self.leaves[i]; }
        while len > 1 {
            let mut next = 0;
            let mut i = 0;
            while i < len {
                let l = buf[i];
                let r = if i + 1 < len { buf[i + 1] } else { l };
                buf[next] = h3::pair(&l, &r);
                next += 1;
                i += 2;
            }
            len = next;
        }
        buf[0]
    }
}
