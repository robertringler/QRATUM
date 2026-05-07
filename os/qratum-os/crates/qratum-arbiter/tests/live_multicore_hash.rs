//! P0.5 — INV-O. Schedule hash is invariant across logical-core
//! permutations of the same workload.

use qratum_arbiter::smp_model::{SmpModel, deterministic_placement};

#[test]
fn schedule_hash_invariant_under_core_permutation() {
    let mut a = SmpModel::new(8);
    let mut b = SmpModel::new(8);
    for tid in 0u32..32 {
        let key = (tid as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        a.create(tid, key);
    }
    for tid in (0u32..32).rev() {
        let key = (tid as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        b.create(tid, key);
    }
    assert_eq!(a.composed_state_hash(), b.composed_state_hash(),
               "INV-O: schedule hash must be order-independent");
}

#[test]
fn deterministic_placement_is_pure() {
    for t in 0u32..1024 {
        assert_eq!(deterministic_placement(t, 8), deterministic_placement(t, 8));
    }
}
