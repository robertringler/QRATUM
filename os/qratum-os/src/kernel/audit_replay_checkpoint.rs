//! P0.6 — Replay checkpoint. INV-U: restore reproduces identical
//! receipt root.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Checkpoint {
    pub seq:           u64,
    pub epoch:         u64,
    pub merkle_root:   [u8; 32],
    pub audit_cursor:  u64,
    pub barrier_hash:  u64,
}

const CAP: usize = 64;
static mut RING: [Checkpoint; CAP] = [Checkpoint {
    seq: 0, epoch: 0, merkle_root: [0u8; 32], audit_cursor: 0, barrier_hash: 0,
}; CAP];
static HEAD: AtomicU64 = AtomicU64::new(0);

pub fn save(cp: Checkpoint) -> u64 {
    let s = HEAD.fetch_add(1, Ordering::AcqRel);
    if (s as usize) < CAP {
        let mut c = cp; c.seq = s;
        unsafe { RING[s as usize] = c; }
    }
    s
}

pub fn load(seq: u64) -> Option<Checkpoint> {
    if (seq as usize) >= CAP { return None; }
    let head = HEAD.load(Ordering::Acquire);
    if seq >= head { return None; }
    unsafe { Some(RING[seq as usize]) }
}

#[inline]
pub fn count() -> u64 { HEAD.load(Ordering::Acquire) }

pub fn reset() { HEAD.store(0, Ordering::Release); }
