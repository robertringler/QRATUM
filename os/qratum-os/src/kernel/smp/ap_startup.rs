//! P0.6 — Real AP startup path (deterministic record + receipt).
//!
//! Real INIT/SIPI sequencing remains gated behind `cfg(target_feature
//! = "smp_real")`. This module owns the deterministic *AP startup
//! receipt* record produced for every AP whose online-bit transitions
//! from 0 → 1. Receipts are appended to `interrupt_receipt`'s shared
//! ring — see `INV-T`.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

use super::MAX_CPUS;

#[derive(Copy, Clone, Default, Debug, Eq, PartialEq)]
pub struct ApStartupReceipt {
    pub seq:          u64,
    pub cpu_id:       u8,
    pub apic_id:      u8,
    pub _pad:         [u8; 6],
    pub stack_top:    u64,
    pub tsc_boot:     u64,
    pub prev_link:    u64,
}

const CAP: usize = MAX_CPUS;
static mut RING: [ApStartupReceipt; CAP] = [ApStartupReceipt {
    seq: 0, cpu_id: 0, apic_id: 0, _pad: [0; 6],
    stack_top: 0, tsc_boot: 0, prev_link: 0,
}; CAP];
static HEAD: AtomicU64 = AtomicU64::new(0);
static CHAIN: AtomicU64 = AtomicU64::new(0xCBF2_9CE4_8422_2325);

/// Stage and chain a per-AP startup receipt. INV-T: returns the
/// receipt's chained link hash; identical inputs produce identical
/// hashes.
pub fn emit(cpu_id: u8, apic_id: u8, stack_top: u64, tsc_boot: u64) -> u64 {
    let s = HEAD.fetch_add(1, Ordering::AcqRel);
    let prev = CHAIN.load(Ordering::Acquire);
    let mut r = ApStartupReceipt {
        seq: s, cpu_id, apic_id, _pad: [0; 6],
        stack_top, tsc_boot, prev_link: prev,
    };
    let next = link(&r);
    r.prev_link = prev;
    if (s as usize) < CAP {
        unsafe { RING[s as usize] = r; }
    }
    CHAIN.store(next, Ordering::Release);
    next
}

#[inline]
pub fn count() -> u64 { HEAD.load(Ordering::Acquire) }

#[inline]
pub fn chain_hash() -> u64 { CHAIN.load(Ordering::Acquire) }

fn link(r: &ApStartupReceipt) -> u64 {
    let mut h: u64 = r.prev_link;
    for b in r.seq.to_le_bytes()       { h = step(h, b); }
    h = step(h, r.cpu_id); h = step(h, r.apic_id);
    for b in r.stack_top.to_le_bytes() { h = step(h, b); }
    for b in r.tsc_boot.to_le_bytes()  { h = step(h, b); }
    h
}

#[inline]
fn step(h: u64, b: u8) -> u64 { (h ^ b as u64).wrapping_mul(0x100_0000_01B3) }

pub fn reset() {
    HEAD.store(0, Ordering::Release);
    CHAIN.store(0xCBF2_9CE4_8422_2325, Ordering::Release);
}
