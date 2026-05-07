//! P0.6 — Kernel `crypto` module.
//!
//! The kernel runs `no_std`, so this module ships a deterministic
//! 32-byte digest function (`blake3_kernel`) and a hash-chain helper
//! (`receipt_hash_chain`). The chosen digest is **collision-resistant
//! enough for replay determinism** and produces a 32-byte output
//! whose host-side equivalent is real Blake3. Cryptographic claims
//! (e.g. for external verification) are made by the host-side
//! `qratum-arbiter` re-hashing the same bytes with the real Blake3
//! crate; the kernel digest provides an append-only chained anchor.

#![allow(dead_code)]

pub mod blake3_kernel;
pub mod receipt_hash_chain;
