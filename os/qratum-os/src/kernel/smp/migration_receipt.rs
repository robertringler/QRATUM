//! P0.5 — Migration receipt chain (INV-Q).
//!
//! Every migration commit emits one `MigrationReceipt`. The chain
//! hash is FNV-1a-64 over the ordered receipt stream and is exposed
//! to the host through the runtime model.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct MigrationReceipt {
    pub seq:               u64,
    pub task_id:           u32,
    pub src:               u8,
    pub dst:               u8,
    pub _pad:              [u8; 2],
    pub key_hi:            u64,
    pub state_hash_before: u64,
    pub state_hash_after:  u64,
    pub arbitration_link:  u64,
}

const CAP: usize = 4096;
static mut RING: [MigrationReceipt; CAP] = [MigrationReceipt {
    seq: 0, task_id: 0, src: 0, dst: 0, _pad: [0; 2], key_hi: 0,
    state_hash_before: 0, state_hash_after: 0, arbitration_link: 0,
}; CAP];
static HEAD: AtomicU64 = AtomicU64::new(0);

pub fn append(r: MigrationReceipt) {
    let h = HEAD.fetch_add(1, Ordering::AcqRel) as usize;
    unsafe { RING[h % CAP] = r; }
}

#[inline]
pub fn count() -> u64 { HEAD.load(Ordering::Acquire) }

pub fn chain_hash() -> u64 {
    let h = (HEAD.load(Ordering::Acquire) as usize).min(CAP);
    let mut acc: u64 = 0xCBF2_9CE4_8422_2325;
    for i in 0..h {
        let r = unsafe { RING[i] };
        for b in r.seq.to_le_bytes()               { acc = mix(acc, b); }
        for b in r.task_id.to_le_bytes()           { acc = mix(acc, b); }
        acc = mix(acc, r.src); acc = mix(acc, r.dst);
        for b in r.key_hi.to_le_bytes()            { acc = mix(acc, b); }
        for b in r.state_hash_before.to_le_bytes() { acc = mix(acc, b); }
        for b in r.state_hash_after.to_le_bytes()  { acc = mix(acc, b); }
        for b in r.arbitration_link.to_le_bytes()  { acc = mix(acc, b); }
    }
    acc
}

#[inline]
fn mix(h: u64, b: u8) -> u64 { (h ^ b as u64).wrapping_mul(0x100_0000_01B3) }

pub fn reset() { HEAD.store(0, Ordering::Release); }
