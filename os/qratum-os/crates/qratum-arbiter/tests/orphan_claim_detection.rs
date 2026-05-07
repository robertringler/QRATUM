//! Orphan claim detection: a steal record that names a non-existent
//! source or destination must be flagged as orphan.

use qratum_arbiter::deterministic_steal::StealRecord;

fn is_orphan(rec: &StealRecord, n_cores: u8) -> bool {
    rec.src >= n_cores || rec.dst >= n_cores || rec.src == rec.dst
}

#[test]
fn well_formed_record_not_orphan() {
    let r = StealRecord { seq: 0, src: 0, dst: 1, depth_src: 8, depth_dst: 0 };
    assert!(!is_orphan(&r, 4));
}

#[test]
fn out_of_range_src_is_orphan() {
    let r = StealRecord { seq: 0, src: 99, dst: 1, depth_src: 8, depth_dst: 0 };
    assert!(is_orphan(&r, 4));
}

#[test]
fn self_steal_is_orphan() {
    let r = StealRecord { seq: 0, src: 1, dst: 1, depth_src: 1, depth_dst: 1 };
    assert!(is_orphan(&r, 4));
}
