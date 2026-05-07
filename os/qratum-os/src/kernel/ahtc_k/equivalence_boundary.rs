//! P0.4 — Equivalence boundary for AHTC-K canonicalization.
//!
//! INV-N: semantically distinct execution graphs MUST NOT fold
//! together. This module makes the boundary explicit by computing a
//! *secondary* discriminator (the "shadow key") for every fold
//! operation. If two intents share a primary canonical key but differ
//! in their shadow key, the merge is a false positive and trips
//! `divergence_detector::report_collision`.
//!
//! Shadow key dimensions (independent of canonical):
//!   - exact byte hash of `params`
//!   - exact authority-tier id
//!   - exact lock_key
//!   - exact priority

#![allow(dead_code)]

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct ShadowKey {
    pub params_hash: u64,
    pub authority:   u32,
    pub priority:    u32,
    pub lock_key:    u64,
}

pub fn shadow_of(params: &[u8], authority: u32, priority: u32, lock_key: u64) -> ShadowKey {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &b in params { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
    ShadowKey { params_hash: h, authority, priority, lock_key }
}

/// Confidence score in [0, 1_000_000]. 1_000_000 = full match,
/// 0 = total semantic disagreement. Returns the per-dimension match
/// vector also, for the divergence report.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct EquivalenceVerdict {
    pub confidence_q:   u64, // scaled by 1_000_000
    pub params_match:   bool,
    pub authority_match:bool,
    pub priority_match: bool,
    pub lock_match:     bool,
}

pub fn compare(a: &ShadowKey, b: &ShadowKey) -> EquivalenceVerdict {
    let p = a.params_hash == b.params_hash;
    let u = a.authority   == b.authority;
    let r = a.priority    == b.priority;
    let l = a.lock_key    == b.lock_key;
    let bits = (p as u64) + (u as u64) + (r as u64) + (l as u64);
    EquivalenceVerdict {
        confidence_q:    bits * 250_000, // 4 bits → 1_000_000 max
        params_match:    p,
        authority_match: u,
        priority_match:  r,
        lock_match:      l,
    }
}

#[inline]
pub fn is_safe_merge(v: &EquivalenceVerdict) -> bool { v.confidence_q == 1_000_000 }
