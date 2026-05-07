//! P0.4 — Cross-core invariants under stress (INV-K, INV-M).

use qratum_arbiter::smp_model::SmpModel;

fn xrand(state: &mut u64) -> u64 {
    *state ^= *state << 13; *state ^= *state >> 7; *state ^= *state << 17; *state
}

#[test]
fn migration_storm_preserves_inv_k_and_inv_m() {
    let mut m = SmpModel::new(8);
    for t in 1u32..=64 { m.enqueue(0, t, t as u64); }
    let initial_hash = m.composed_state_hash();
    let mut s = 0xDEAD_BEEF_FACE_F00Du64;
    let mut migrations = 0;
    for _ in 0..1000 {
        let task = (xrand(&mut s) % 64 + 1) as u32;
        let src  = (xrand(&mut s) % 8) as u8;
        let dst  = (xrand(&mut s) % 8) as u8;
        if src == dst { continue; }
        if m.migrate(task, src, dst) { migrations += 1; }
        assert!(m.no_dual_occupancy(),
            "INV-K broken after {} migrations", migrations);
    }
    // Hash must have evolved (we did some migrations) AND remain
    // deterministic (re-hash gives same value).
    let h1 = m.composed_state_hash();
    let h2 = m.composed_state_hash();
    assert_eq!(h1, h2, "INV-M: hash must be stable across re-hash");
    if migrations > 0 {
        assert_ne!(h1, initial_hash);
    }
}

#[test]
fn parallel_workloads_with_same_input_yield_same_hash() {
    // Two independent constructions using identical inputs.
    let build = || -> SmpModel {
        let mut m = SmpModel::new(4);
        for t in 1u32..=32 { m.create(t, t as u64); }
        m
    };
    let a = build(); let b = build();
    assert_eq!(a.composed_state_hash(), b.composed_state_hash());
    assert_eq!(a.queue_membership(),    b.queue_membership());
}

#[test]
fn refused_migration_does_not_alter_state() {
    let mut m = SmpModel::new(4);
    for t in 1u32..=4 { m.enqueue(0, t, 0); }
    let before = m.composed_state_hash();
    // Task not at head — refused.
    assert!(!m.migrate(3, 0, 1));
    assert_eq!(m.composed_state_hash(), before,
        "Refused migration must not alter scheduler state");
}
