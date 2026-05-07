//! P0.5 — Receipt chain hashing (pure functions).

#![allow(dead_code)]

#[inline]
fn fnv_step(h: u64, b: u8) -> u64 { (h ^ b as u64).wrapping_mul(0x100_0000_01B3) }

pub fn link_hash(prev: u64, transition: u64, arbitration: u64, migration: u64) -> u64 {
    let mut h = prev;
    for b in transition.to_le_bytes()  { h = fnv_step(h, b); }
    for b in arbitration.to_le_bytes() { h = fnv_step(h, b); }
    for b in migration.to_le_bytes()   { h = fnv_step(h, b); }
    h
}
