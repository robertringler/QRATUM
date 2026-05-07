//! P0.5 — Work-stealing must preserve canonical ordering.

use qratum_arbiter::smp_model::SmpModel;

#[test]
fn steal_preserves_invariant_k() {
    let mut m = SmpModel::new(4);
    for tid in 0u32..16 { m.create(tid, tid as u64); }
    // Simulate steal: migrate any task at head from busiest to idle.
    for _ in 0..8 {
        let busy = (0..4).max_by_key(|c| m.queues[*c as usize].buf.len()).unwrap();
        let idle = (0..4).min_by_key(|c| m.queues[*c as usize].buf.len()).unwrap();
        if busy == idle { break; }
        if let Some(head) = m.queues[busy as usize].buf.first().cloned() {
            let _ = m.migrate(head.task_id, busy, idle);
        }
    }
    assert!(m.no_dual_occupancy(), "INV-K under steals");
}
