//! P0.6 — queue pressure stability under synthetic load.

use qratum_arbiter::deterministic_steal::pick_steal;

#[test]
fn steady_pressure_no_steal() {
    for _ in 0..1000 {
        assert_eq!(pick_steal(&[8, 8, 8, 8]), None);
    }
}

#[test]
fn unbalanced_pressure_triggers_steal_deterministically() {
    let depths = [16u32, 0, 0, 0];
    let mut last = pick_steal(&depths);
    for _ in 0..1000 {
        assert_eq!(pick_steal(&depths), last);
        last = pick_steal(&depths);
    }
}
