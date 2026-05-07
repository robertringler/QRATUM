//! `canonicalize` — total, pure, allocation-free.

use super::key::{fnv1a_64, CanonicalKey};

/// Inputs to canonicalization. Mirrors the directive's required fields
/// (authority, scope, priority, name, params, lock_key).
pub struct CanonInputs<'a> {
    pub authority: u8,
    pub scope:     u8,
    pub priority:  u8,
    pub name:      &'a [u8],
    pub params:    &'a [u8],
    pub lock_key:  u64,
}

/// `canonicalize` — produces a stable canonical identity. Pure.
///
/// K1: same input → same output across runs of the same binary.
/// K4: `lock_key` is taken verbatim; AHTC-K never derives or mutates it.
/// K5: hashing is over raw input bytes (no whitespace normalization).
#[inline]
pub fn canonicalize(i: &CanonInputs<'_>) -> CanonicalKey {
    CanonicalKey {
        authority:   i.authority,
        scope:       i.scope,
        priority:    i.priority,
        _pad:        0,
        name_hash:   fnv1a_64(i.name),
        params_hash: fnv1a_64(i.params),
        lock_key:    i.lock_key,
    }
}
