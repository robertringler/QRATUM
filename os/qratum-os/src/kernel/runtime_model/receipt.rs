//! P0.5 — Execution receipt (per-transition).

#![allow(dead_code)]

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct ExecutionReceipt {
    pub seq:                u64,
    pub transition_kind:    u8, // 0=Submit 1=Fold 2=Migrate 3=Replay
    pub _pad:               [u8; 7],
    pub canonical_key_lo:   u64,
    pub canonical_key_hi:   u64,
    pub arbitration_link:   u64,
    pub migration_link:     u64,
    pub replay_link:        u64,
}

#[inline]
pub fn fnv64(seed: u64, bytes: &[u8]) -> u64 {
    let mut h = seed;
    for &b in bytes { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
    h
}

/// Canonical transition hash — pure function of receipt fields.
pub fn transition_hash(r: &ExecutionReceipt) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for b in r.seq.to_le_bytes()              { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
    h = (h ^ r.transition_kind as u64).wrapping_mul(0x100_0000_01B3);
    for b in r.canonical_key_lo.to_le_bytes() { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
    for b in r.canonical_key_hi.to_le_bytes() { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
    for b in r.arbitration_link.to_le_bytes() { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
    for b in r.migration_link.to_le_bytes()   { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
    for b in r.replay_link.to_le_bytes()      { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
    h
}
