//! P0.6 — Deterministic work stealing (INV-V).
//!
//! A *steal* must be a pure function of `(snapshot of queue depths,
//! steal-counter)`. This module implements the canonical decision
//! procedure used both by the kernel and the host model.

#![allow(dead_code)]

/// Pick the "busy" core (max depth) and the "idle" core (min depth)
/// using a deterministic tie-break (lowest cpu_id wins). Returns
/// `Some((src, dst))` if a steal is legal; `None` if not.
pub fn pick_steal(depths: &[u32]) -> Option<(u8, u8)> {
    if depths.len() < 2 { return None; }
    let mut max_d = 0u32; let mut min_d = u32::MAX;
    let mut src = 0u8;    let mut dst = 0u8;
    for (i, &d) in depths.iter().enumerate() {
        if d > max_d { max_d = d; src = i as u8; }
        if d < min_d { min_d = d; dst = i as u8; }
    }
    if max_d <= min_d.saturating_add(1) { return None; }
    if src == dst { return None; }
    Some((src, dst))
}

/// Receipt for a single steal decision.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct StealReceipt {
    pub seq:        u64,
    pub src:        u8,
    pub dst:        u8,
    pub depth_src:  u32,
    pub depth_dst:  u32,
}

pub fn receipt_hash(r: &StealReceipt) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for b in r.seq.to_le_bytes()       { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
    h = (h ^ r.src as u64).wrapping_mul(0x100_0000_01B3);
    h = (h ^ r.dst as u64).wrapping_mul(0x100_0000_01B3);
    for b in r.depth_src.to_le_bytes() { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
    for b in r.depth_dst.to_le_bytes() { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
    h
}
