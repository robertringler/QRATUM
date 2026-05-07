//! `CanonicalKey` — bitwise structural identity of a canonicalized intent.

/// Canonical key. Bitwise equality is the *only* equivalence relation
/// (K5). No tolerance, no normalization beyond byte hashing.
#[repr(C, align(8))]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct CanonicalKey {
    pub authority:   u8,
    pub scope:       u8,
    pub priority:    u8,
    pub _pad:        u8,
    pub name_hash:   u64,
    pub params_hash: u64,
    pub lock_key:    u64,
}

impl CanonicalKey {
    pub const ZERO: Self = Self {
        authority: 0, scope: 0, priority: 0, _pad: 0,
        name_hash: 0, params_hash: 0, lock_key: 0,
    };
}

/// FNV-1a-64. Deterministic, allocation-free, no_std-friendly.
///
/// XXX-blake3: placeholder until the kernel-side Blake3 swap (P0.3.1).
/// Host-side `qratum-arbiter::ahtc_k` already uses real Blake3 — the
/// determinism contract is *within-impl*, not cross-impl (see
/// `spec/ahtc_k.md` §2.1).
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME:  u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    let mut i = 0;
    while i < bytes.len() {
        h ^= bytes[i] as u64;
        h = h.wrapping_mul(PRIME);
        i += 1;
    }
    h
}
