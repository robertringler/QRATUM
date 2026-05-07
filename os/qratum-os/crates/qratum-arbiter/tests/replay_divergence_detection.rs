//! P0.5 — Replay divergence detection. Mutating a single byte of any
//! scenario produces a different fingerprint.

use qratum_arbiter::adversarial_corpus::{Scenario, AdversarialCategory};

#[test]
fn single_byte_mutation_diverges() {
    let s1 = Scenario::synthesize(7, AdversarialCategory::MigrationCascade, 0xABCD);
    let mut s2 = s1.clone();
    s2.ops[0] ^= 0x01;
    assert_ne!(s1.fingerprint(), s2.fingerprint());
}

#[test]
fn category_change_diverges() {
    let s1 = Scenario::synthesize(1, AdversarialCategory::MulticoreRaceStorm, 1);
    let s2 = Scenario::synthesize(1, AdversarialCategory::IrqFloodCondition,  1);
    assert_ne!(s1.fingerprint(), s2.fingerprint());
}

#[test]
fn seed_change_diverges() {
    let s1 = Scenario::synthesize(2, AdversarialCategory::CacheThrashWorkload, 1);
    let s2 = Scenario::synthesize(2, AdversarialCategory::CacheThrashWorkload, 2);
    assert_ne!(s1.fingerprint(), s2.fingerprint());
}
