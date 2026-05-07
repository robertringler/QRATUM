//! P0.5 — Replay-visible lock contention trace.
//!
//! Records every lock acquire/release ticket as a flat byte stream
//! whose hash is exposed to the host model. Capacity-bounded and
//! lock-free.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

#[derive(Copy, Clone, Default, Debug)]
pub struct ContentionEntry {
    pub seq:     u64,
    pub lock_id: u32,
    pub op:      u8, // 0=acquire 1=release
    pub _pad:    [u8; 3],
    pub ticket:  u64,
}

const CAP: usize = 4096;
static mut RING: [ContentionEntry; CAP] = [ContentionEntry {
    seq: 0, lock_id: 0, op: 0, _pad: [0; 3], ticket: 0,
}; CAP];
static SEQ: AtomicU64 = AtomicU64::new(0);

pub fn record_acquire(lock_id: u32, ticket: u64) {
    let s = SEQ.fetch_add(1, Ordering::AcqRel);
    unsafe { RING[(s as usize) % CAP] = ContentionEntry {
        seq: s, lock_id, op: 0, _pad: [0; 3], ticket,
    }; }
}

pub fn record_release(lock_id: u32, ticket: u64) {
    let s = SEQ.fetch_add(1, Ordering::AcqRel);
    unsafe { RING[(s as usize) % CAP] = ContentionEntry {
        seq: s, lock_id, op: 1, _pad: [0; 3], ticket,
    }; }
}

#[inline]
pub fn count() -> u64 { SEQ.load(Ordering::Acquire) }

pub fn chain_hash() -> u64 {
    let n = (SEQ.load(Ordering::Acquire) as usize).min(CAP);
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for i in 0..n {
        let e = unsafe { RING[i] };
        for b in e.seq.to_le_bytes()     { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
        for b in e.lock_id.to_le_bytes() { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
        h = (h ^ e.op as u64).wrapping_mul(0x100_0000_01B3);
        for b in e.ticket.to_le_bytes()  { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
    }
    h
}

pub fn reset() { SEQ.store(0, Ordering::Release); }
