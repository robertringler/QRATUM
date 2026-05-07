//! P0.5 — Replay-stable per-core run-queue receipt.
//!
//! Wraps `smp::scheduler` ops with a receipt emitter. Every enqueue,
//! dequeue, or steal produces a `RunQueueReceipt`. Receipts are
//! FNV-1a-64 chained — see `migration_receipt.rs`.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct RunQueueReceipt {
    pub seq:        u64,
    pub cpu:        u8,
    pub op:         u8, // 0=enqueue 1=dequeue 2=steal
    pub task_id:    u32,
    pub key_hi:     u64,
    pub depth_after:u32,
}

const CAP: usize = 4096;
static mut RING: [RunQueueReceipt; CAP] = [RunQueueReceipt {
    seq: 0, cpu: 0, op: 0, task_id: 0, key_hi: 0, depth_after: 0,
}; CAP];
static HEAD: AtomicU64 = AtomicU64::new(0);

pub fn record(r: RunQueueReceipt) {
    let h = HEAD.fetch_add(1, Ordering::AcqRel) as usize;
    // SAFETY: bounded ring; only a single writer holds the
    // migration barrier, so concurrent overlap is impossible.
    unsafe { RING[h % CAP] = r; }
}

#[inline]
pub fn head() -> u64 { HEAD.load(Ordering::Acquire) }

pub fn chain_hash() -> u64 {
    let h = (HEAD.load(Ordering::Acquire) as usize).min(CAP);
    let mut acc: u64 = 0xCBF2_9CE4_8422_2325;
    for i in 0..h {
        let r = unsafe { RING[i] };
        for b in r.seq.to_le_bytes()         { acc = mix(acc, b); }
        acc = mix(acc, r.cpu); acc = mix(acc, r.op);
        for b in r.task_id.to_le_bytes()     { acc = mix(acc, b); }
        for b in r.key_hi.to_le_bytes()      { acc = mix(acc, b); }
        for b in r.depth_after.to_le_bytes() { acc = mix(acc, b); }
    }
    acc
}

#[inline]
fn mix(h: u64, b: u8) -> u64 { (h ^ b as u64).wrapping_mul(0x100_0000_01B3) }

pub fn reset() { HEAD.store(0, Ordering::Release); }
