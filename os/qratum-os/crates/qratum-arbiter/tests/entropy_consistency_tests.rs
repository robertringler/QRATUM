//! T6.3 — Entropy consistency.
//!
//! Asserts that `performance::shannon_entropy_bits` (host) and
//! `kernel::instrumentation::metrics::shannon_entropy_bits` (kernel)
//! are bytewise-identical across a battery of histograms.
//!
//! We can't link the kernel `no_std` symbol into a host test, so we
//! re-implement the *same algorithm* in this test file and assert it
//! agrees with the host implementation. The kernel implementation is
//! reviewed against this canonical form via INV-G (the doc phrase
//! "Shannon entropy" is a registered claim phrase).

use qratum_arbiter::performance::{shannon_entropy_bits, ENTROPY_SCALE};

/// Reference implementation — character-for-character mirror of the
/// kernel form in `os/qratum-os/src/kernel/instrumentation/metrics.rs`.
fn reference(counts: &[u64]) -> u64 {
    let total: u128 = counts.iter().map(|&c| c as u128).sum();
    if total == 0 { return 0; }
    let mut acc: u128 = 0;
    for &c in counts {
        if c == 0 { continue; }
        let log_total = (total as u64).ilog2() as u128;
        let log_c     = (c as u64).max(1).ilog2() as u128;
        let term = (c as u128) * (log_total - log_c) * (ENTROPY_SCALE as u128);
        acc += term / total;
    }
    acc as u64
}

#[test]
fn empty_histogram_yields_zero_entropy() {
    assert_eq!(shannon_entropy_bits(&[]), 0);
    assert_eq!(reference(&[]), 0);
}

#[test]
fn singleton_yields_zero_entropy() {
    let h = vec![1000u64];
    assert_eq!(shannon_entropy_bits(&h), 0);
    assert_eq!(reference(&h), 0);
}

#[test]
fn host_and_reference_agree_byte_for_byte() {
    let cases: &[&[u64]] = &[
        &[10, 5, 3, 2, 1],
        &[100, 100, 100, 100],
        &[1, 1, 1, 1, 1, 1, 1, 1],
        &[64, 1],
        &[1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
        &[1, 0, 0, 0, 5],
        &[0; 8],
    ];
    for (i, c) in cases.iter().enumerate() {
        let host  = shannon_entropy_bits(c);
        let refr  = reference(c);
        assert_eq!(host, refr,
            "case {}: host={}, ref={}, hist={:?}", i, host, refr, c);
    }
}

#[test]
fn entropy_scale_is_one_million() {
    assert_eq!(ENTROPY_SCALE, 1_000_000);
}

#[test]
fn entropy_is_monotone_in_uniformity() {
    // A flat distribution has STRICTLY more entropy than a peaked
    // one of the same support and total mass.
    let flat   = shannon_entropy_bits(&[8, 8, 8, 8]);
    let peaked = shannon_entropy_bits(&[29, 1, 1, 1]);
    assert!(flat > peaked, "flat={}, peaked={}", flat, peaked);
}
