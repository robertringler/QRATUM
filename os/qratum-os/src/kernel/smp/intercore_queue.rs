//! P0.6 — Inter-core message queue with deterministic ordering.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct InterMsg {
    pub seq:    u64,
    pub from:   u8,
    pub to:     u8,
    pub _pad:   [u8; 6],
    pub kind:   u32,
    pub payload:u64,
}

const CAP: usize = 1024;
static mut RING: [InterMsg; CAP] = [InterMsg {
    seq: 0, from: 0, to: 0, _pad: [0; 6], kind: 0, payload: 0,
}; CAP];
static HEAD: AtomicU64 = AtomicU64::new(0);

pub fn enqueue(from: u8, to: u8, kind: u32, payload: u64) -> u64 {
    let s = HEAD.fetch_add(1, Ordering::AcqRel);
    unsafe { RING[(s as usize) % CAP] = InterMsg {
        seq: s, from, to, _pad: [0; 6], kind, payload,
    }; }
    s
}

#[inline]
pub fn count() -> u64 { HEAD.load(Ordering::Acquire) }

pub fn chain_hash() -> u64 {
    let n = (HEAD.load(Ordering::Acquire) as usize).min(CAP);
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for i in 0..n {
        let m = unsafe { RING[i] };
        for b in m.seq.to_le_bytes()     { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
        h = (h ^ m.from as u64).wrapping_mul(0x100_0000_01B3);
        h = (h ^ m.to as u64).wrapping_mul(0x100_0000_01B3);
        for b in m.kind.to_le_bytes()    { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
        for b in m.payload.to_le_bytes() { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
    }
    h
}

pub fn reset() { HEAD.store(0, Ordering::Release); }
