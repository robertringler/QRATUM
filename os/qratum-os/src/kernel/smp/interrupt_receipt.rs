//! P0.6 — Interrupt receipt log. Every dispatched IRQ becomes a
//! deterministic record observable to the host replay.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct IrqReceipt {
    pub seq:    u64,
    pub vector: u8,
    pub cpu:    u8,
    pub _pad:   [u8; 6],
    pub tsc:    u64,
}

const CAP: usize = 4096;
static mut RING: [IrqReceipt; CAP] = [IrqReceipt {
    seq: 0, vector: 0, cpu: 0, _pad: [0; 6], tsc: 0,
}; CAP];
static HEAD: AtomicU64 = AtomicU64::new(0);

pub fn record(vector: u8, cpu: u8, tsc: u64) {
    let s = HEAD.fetch_add(1, Ordering::AcqRel);
    unsafe { RING[(s as usize) % CAP] = IrqReceipt {
        seq: s, vector, cpu, _pad: [0; 6], tsc,
    }; }
}

#[inline]
pub fn count() -> u64 { HEAD.load(Ordering::Acquire) }

pub fn chain_hash() -> u64 {
    let n = (HEAD.load(Ordering::Acquire) as usize).min(CAP);
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for i in 0..n {
        let r = unsafe { RING[i] };
        for b in r.seq.to_le_bytes() { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
        h = (h ^ r.vector as u64).wrapping_mul(0x100_0000_01B3);
        h = (h ^ r.cpu as u64).wrapping_mul(0x100_0000_01B3);
        for b in r.tsc.to_le_bytes() { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
    }
    h
}

pub fn reset() { HEAD.store(0, Ordering::Release); }
