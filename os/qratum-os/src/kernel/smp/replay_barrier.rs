//! P0.6 — Replay barrier. All CPUs must rendezvous before replay
//! resumes; the barrier records a "barrier hash" that is part of the
//! receipt root.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

use super::MAX_CPUS;

static ARRIVED: [AtomicU64; MAX_CPUS] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
];
static GENERATION: AtomicU64 = AtomicU64::new(0);

pub fn arrive(cpu_id: u8) -> u64 {
    if (cpu_id as usize) >= MAX_CPUS { return 0; }
    let g = GENERATION.load(Ordering::Acquire);
    ARRIVED[cpu_id as usize].store(g + 1, Ordering::Release);
    g + 1
}

pub fn all_arrived(expected: u8) -> bool {
    let g = GENERATION.load(Ordering::Acquire) + 1;
    let mut k = 0;
    for i in 0..MAX_CPUS {
        if ARRIVED[i].load(Ordering::Acquire) >= g { k += 1; }
    }
    k >= expected
}

pub fn release() -> u64 {
    GENERATION.fetch_add(1, Ordering::AcqRel) + 1
}

pub fn barrier_hash(generation: u64) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for b in generation.to_le_bytes() { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
    for i in 0..MAX_CPUS {
        let v = ARRIVED[i].load(Ordering::Acquire);
        for b in v.to_le_bytes() { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
    }
    h
}

pub fn reset() {
    GENERATION.store(0, Ordering::Release);
    for s in &ARRIVED { s.store(0, Ordering::Release); }
}
