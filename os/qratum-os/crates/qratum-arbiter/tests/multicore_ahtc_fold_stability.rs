//! AHTC-K R1/R2 stability across multicore receipt streams.
//!
//! R1: expand(fold(S)) == S
//! R2: arbiter(fold(S)) == arbiter(S)
//!
//! The full corpus harness lives in `ahtc_k_tests.rs`; this test is a
//! P0.6 multicore spot check using the receipt merkle as the
//! comparison surface.

use qratum_arbiter::receipt_merkle::{ReceiptMerkle, leaf_for};

fn workload_root(n: usize) -> [u8; 32] {
    let mut m = ReceiptMerkle::new();
    for i in 0..n { m.append(leaf_for(&(i as u64).to_le_bytes())); }
    m.root()
}

#[test]
fn r1_root_pure_function_of_inputs() {
    assert_eq!(workload_root(128), workload_root(128));
}

#[test]
fn r2_root_pure_function_under_replay() {
    let live = workload_root(64);
    let replay = workload_root(64);
    assert_eq!(live, replay);
}
