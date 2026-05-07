//! P0.5 — Receipt chain (append-only, hash-linked).

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

use super::receipt::{transition_hash, ExecutionReceipt};
use super::receipt_hash::link_hash;

const CAP: usize = 8192;
static mut RING: [ExecutionReceipt; CAP] = [ExecutionReceipt {
    seq: 0, transition_kind: 0, _pad: [0; 7],
    canonical_key_lo: 0, canonical_key_hi: 0,
    arbitration_link: 0, migration_link: 0, replay_link: 0,
}; CAP];
static HEAD: AtomicU64 = AtomicU64::new(0);
static CHAIN_HASH: AtomicU64 = AtomicU64::new(0xCBF2_9CE4_8422_2325);

pub fn append(mut r: ExecutionReceipt) -> u64 {
    let s = HEAD.fetch_add(1, Ordering::AcqRel);
    r.seq = s;
    unsafe { RING[(s as usize) % CAP] = r; }
    let prev = CHAIN_HASH.load(Ordering::Acquire);
    let next = link_hash(prev, transition_hash(&r), r.arbitration_link, r.migration_link);
    CHAIN_HASH.store(next, Ordering::Release);
    next
}

#[inline] pub fn count() -> u64 { HEAD.load(Ordering::Acquire) }
#[inline] pub fn chain_hash() -> u64 { CHAIN_HASH.load(Ordering::Acquire) }

pub fn reset() {
    HEAD.store(0, Ordering::Release);
    CHAIN_HASH.store(0xCBF2_9CE4_8422_2325, Ordering::Release);
}
