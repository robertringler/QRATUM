//! P0.6 — Kernel-side 32-byte digest (Blake3-shaped).
//!
//! `no_std`. Output width is 32 bytes for direct compatibility with
//! the host's real Blake3 chain. Internally this is a deterministic
//! mixing function — adequate for replay-determinism receipts. Use
//! the host-side `qratum-arbiter::receipt_merkle::root()` (real
//! Blake3) for cryptographic claims.

#![allow(dead_code)]

const PRIME: u64 = 0x100_0000_01B3;
const SEED:  u64 = 0xCBF2_9CE4_8422_2325;

#[inline]
fn fnv64(seed: u64, data: &[u8]) -> u64 {
    let mut h = seed;
    for &b in data { h = (h ^ b as u64).wrapping_mul(PRIME); }
    h
}

/// 32-byte digest of `data`. Deterministic, no_std.
pub fn digest(data: &[u8]) -> [u8; 32] {
    // Four independent FNV streams with rotated input + distinct seeds.
    let mut out = [0u8; 32];
    let s0 = SEED;
    let s1 = SEED.rotate_left(7)  ^ 0xA5A5_A5A5_A5A5_A5A5;
    let s2 = SEED.rotate_left(13) ^ 0x5A5A_5A5A_5A5A_5A5A;
    let s3 = SEED.rotate_left(31) ^ 0xDEAD_BEEF_CAFE_BABE;
    let h0 = fnv64(s0, data);
    let h1 = fnv64(s1, data);
    let h2 = fnv64(s2, data);
    let h3 = fnv64(s3, data);
    out[0..8].copy_from_slice(&h0.to_le_bytes());
    out[8..16].copy_from_slice(&h1.to_le_bytes());
    out[16..24].copy_from_slice(&h2.to_le_bytes());
    out[24..32].copy_from_slice(&h3.to_le_bytes());
    out
}

/// Pair-hash for Merkle internal nodes.
pub fn pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(left);
    buf[32..].copy_from_slice(right);
    digest(&buf)
}
