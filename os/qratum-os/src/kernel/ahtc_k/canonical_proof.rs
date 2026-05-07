//! P0.4 — Canonical proof artefacts.
//!
//! For every fold operation we record:
//!   - `primary_key_lo`     — `CanonicalKey.lo`
//!   - `primary_key_hi`     — `CanonicalKey.hi`
//!   - `shadow_params_hash`
//!   - `outcome`            — Created | Incremented
//!
//! `proof_chain_hash()` is FNV-1a-64 over this stream in fold-order.
//! It is a kernel-portable identity for the canonicalization run and
//! must match the host engine bytewise (verified by
//! `tests/canonical_proof_tests.rs`).

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

const RING_CAP: usize = 8192;

#[derive(Copy, Clone, Default, Debug, Eq, PartialEq)]
pub struct ProofFrame {
    pub primary_key_lo:     u64,
    pub primary_key_hi:     u64,
    pub shadow_params_hash: u64,
    pub outcome:            u8,    // 0=Created, 1=Incremented
    pub _pad:               [u8; 7],
}

static CHAIN: spin::Mutex<[ProofFrame; RING_CAP]> =
    spin::Mutex::new([ProofFrame {
        primary_key_lo: 0, primary_key_hi: 0, shadow_params_hash: 0,
        outcome: 0, _pad: [0; 7]
    }; RING_CAP]);
static CHAIN_HEAD: AtomicU64 = AtomicU64::new(0);

pub fn append(f: ProofFrame) {
    let h = CHAIN_HEAD.fetch_add(1, Ordering::AcqRel) as usize;
    CHAIN.lock()[h % RING_CAP] = f;
}

pub fn proof_chain_hash() -> u64 {
    let head = (CHAIN_HEAD.load(Ordering::Acquire) as usize).min(RING_CAP);
    let chain = CHAIN.lock();
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for f in &chain[..head] {
        for b in f.primary_key_lo.to_le_bytes() { h = mix(h, b); }
        for b in f.primary_key_hi.to_le_bytes() { h = mix(h, b); }
        for b in f.shadow_params_hash.to_le_bytes() { h = mix(h, b); }
        h = mix(h, f.outcome);
    }
    h
}

#[inline]
fn mix(h: u64, b: u8) -> u64 { (h ^ b as u64).wrapping_mul(0x100_0000_01B3) }

pub fn reset() {
    CHAIN_HEAD.store(0, Ordering::Release);
    *CHAIN.lock() = [ProofFrame {
        primary_key_lo: 0, primary_key_hi: 0, shadow_params_hash: 0,
        outcome: 0, _pad: [0; 7]
    }; RING_CAP];
}
