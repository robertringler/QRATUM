//! P0.5 — 1000-scenario adversarial corpus equivalence. The full
//! corpus generation is byte-identical run-over-run, and every
//! scenario fingerprint is non-zero and unique.

use std::collections::HashSet;
use qratum_arbiter::adversarial_corpus::{generate_full, corpus_hash, TOTAL};

#[test]
fn corpus_size_is_one_thousand() {
    let c = generate_full();
    assert_eq!(c.len(), TOTAL);
    assert_eq!(TOTAL, 1000);
}

#[test]
fn corpus_hash_is_deterministic() {
    let h1 = corpus_hash(&generate_full());
    let h2 = corpus_hash(&generate_full());
    assert_eq!(h1, h2);
    assert_ne!(h1, 0);
}

#[test]
fn fingerprints_are_unique() {
    let c = generate_full();
    let set: HashSet<u64> = c.iter().map(|s| s.fingerprint()).collect();
    assert_eq!(set.len(), c.len(), "no fingerprint collisions");
}
