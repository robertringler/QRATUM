//! P0.6 — interrupt receipts hash chain consistently.

use qratum_arbiter::hardware_replay::{HardwareEvent, ReplaySession};

#[test]
fn irq_only_subset_hash_is_deterministic() {
    let mut a = ReplaySession::new();
    let mut b = ReplaySession::new();
    for i in 0u64..100 {
        let e = HardwareEvent { seq: i, cpu: 0, kind: 0, vector: 32, payload: i };
        a.push(e.clone()); b.push(e);
    }
    assert_eq!(a.stream_hash(), b.stream_hash());
}

#[test]
fn irq_seq_drift_breaks_hash() {
    let mut a = ReplaySession::new();
    let mut b = ReplaySession::new();
    for i in 0u64..100 {
        a.push(HardwareEvent { seq: i,     cpu: 0, kind: 0, vector: 32, payload: i });
        b.push(HardwareEvent { seq: i + 1, cpu: 0, kind: 0, vector: 32, payload: i });
    }
    assert_ne!(a.stream_hash(), b.stream_hash());
}
