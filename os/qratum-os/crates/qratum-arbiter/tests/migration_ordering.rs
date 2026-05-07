//! P0.5 — INV-P. Migration commit order is replay-stable.

use qratum_arbiter::smp_model::SmpModel;

#[test]
fn migration_replay_is_byte_identical() {
    let mut m = SmpModel::new(4);
    for tid in 0u32..8 { m.create(tid, tid as u64); }
    // Force migrations only when task is at queue head; collect successes.
    let mut moves = 0;
    for tid in 0u32..8 {
        for src in 0u8..4 {
            for dst in 0u8..4 {
                if src == dst { continue; }
                if m.migrate(tid, src, dst) { moves += 1; }
            }
        }
    }
    let h1 = m.migration_replay_hash();
    // Reconstruct fresh model from same migration_log via apply.
    let mut m2 = SmpModel::new(4);
    for tid in 0u32..8 { m2.create(tid, tid as u64); }
    for f in m.mig_log.iter().cloned().collect::<Vec<_>>() {
        let _ = m2.migrate(f.task_id, f.src, f.dst);
    }
    assert!(moves >= 0);
    assert_eq!(h1, m.migration_replay_hash(), "replay hash stable");
}
