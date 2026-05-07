//! P0.4 — Migration replay equivalence (INV-L).

use qratum_arbiter::smp_model::SmpModel;

fn build_population(m: &mut SmpModel) {
    for t in 1u32..=16 {
        m.enqueue(0, t, (t as u64).wrapping_mul(0x0111_1111_1111_1111));
    }
}

#[test]
fn identical_migration_sequences_yield_identical_replay_hash() {
    let mut a = SmpModel::new(4);
    let mut b = SmpModel::new(4);
    build_population(&mut a);
    build_population(&mut b);

    let plan: &[(u32, u8, u8)] = &[
        (1, 0, 1), (2, 0, 2), (3, 0, 3),
        (4, 0, 1), (5, 0, 2), (6, 0, 3),
    ];
    for &(t, s, d) in plan { assert!(a.migrate(t, s, d)); }
    for &(t, s, d) in plan { assert!(b.migrate(t, s, d)); }

    assert_eq!(a.migration_replay_hash(), b.migration_replay_hash(),
        "INV-L: identical migration sequences must produce identical replay hash");
    assert_eq!(a.composed_state_hash(),  b.composed_state_hash(),
        "Composed state must also match");
}

#[test]
fn divergent_migration_sequence_yields_divergent_hash() {
    let mut a = SmpModel::new(4);
    let mut b = SmpModel::new(4);
    build_population(&mut a);
    build_population(&mut b);
    a.migrate(1, 0, 1);
    b.migrate(2, 0, 1);
    assert_ne!(a.migration_replay_hash(), b.migration_replay_hash());
}

#[test]
fn migration_preserves_inv_k_no_dual_occupancy() {
    let mut m = SmpModel::new(4);
    build_population(&mut m);
    for t in 1u32..=8 {
        let dst = (t % 3 + 1) as u8;
        m.migrate(t, 0, dst);
        assert!(m.no_dual_occupancy(),
            "INV-K broken after migrating task {}", t);
    }
}

#[test]
fn replay_after_reset_then_replay_log_recovers_state() {
    let mut live = SmpModel::new(4);
    build_population(&mut live);
    let plan: &[(u32, u8, u8)] = &[(1,0,1),(2,0,2),(3,0,3),(4,0,1)];
    for &(t,s,d) in plan { live.migrate(t,s,d); }
    let live_hash = live.migration_replay_hash();

    let mut replay = SmpModel::new(4);
    build_population(&mut replay);
    for f in live.mig_log.clone() {
        assert!(replay.migrate(f.task_id, f.src, f.dst));
    }
    assert_eq!(replay.migration_replay_hash(), live_hash);
}
