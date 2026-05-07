//! P0.6 — Cryptographic receipt anchor (INV-Z, append-only).
//!
//! Anchors the receipt Merkle root into the audit chain. Append-only
//! by construction: every `anchor()` call produces a strictly larger
//! seq number and chains the previous anchor hash.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Anchor {
    pub seq:        u64,
    pub epoch:      u64,
    pub merkle_root:[u8; 32],
    pub prev_link:  [u8; 32],
}

const CAP: usize = 256;
static mut RING: [Anchor; CAP] = [Anchor {
    seq: 0, epoch: 0, merkle_root: [0u8; 32], prev_link: [0u8; 32],
}; CAP];
static HEAD: AtomicU64 = AtomicU64::new(0);
static mut LAST_LINK: [u8; 32] = [0u8; 32];

pub fn anchor(epoch: u64, merkle_root: [u8; 32]) -> u64 {
    let s = HEAD.fetch_add(1, Ordering::AcqRel);
    let prev = unsafe { LAST_LINK };
    let a = Anchor { seq: s, epoch, merkle_root, prev_link: prev };
    if (s as usize) < CAP { unsafe { RING[s as usize] = a; } }
    let next = crate::kernel::crypto::blake3_kernel::pair(&prev, &merkle_root);
    unsafe { LAST_LINK = next; }
    s
}

#[inline]
pub fn count() -> u64 { HEAD.load(Ordering::Acquire) }

#[inline]
pub fn last_link() -> [u8; 32] { unsafe { LAST_LINK } }

pub fn reset() {
    HEAD.store(0, Ordering::Release);
    unsafe { LAST_LINK = [0u8; 32]; }
}
