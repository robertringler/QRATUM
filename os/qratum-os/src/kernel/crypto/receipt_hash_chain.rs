//! P0.6 — Receipt hash chain over 32-byte digests.

#![allow(dead_code)]

use super::blake3_kernel as h3;

#[derive(Copy, Clone, Debug)]
pub struct ReceiptHashChain {
    pub head:   [u8; 32],
    pub count:  u64,
}

impl ReceiptHashChain {
    pub const fn new() -> Self { Self { head: [0u8; 32], count: 0 } }

    pub fn append(&mut self, leaf: &[u8; 32]) -> [u8; 32] {
        self.head = h3::pair(&self.head, leaf);
        self.count += 1;
        self.head
    }

    #[inline]
    pub fn head(&self) -> [u8; 32] { self.head }
}
