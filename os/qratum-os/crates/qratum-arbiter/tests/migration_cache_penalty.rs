//! P0.5 — Migration cache penalty grows monotonically with NUMA
//! distance and working-set size.

use qratum_arbiter::numa_model::migration_penalty_cycles;

#[test]
fn penalty_monotone_in_distance() {
    let ws = 64u64;
    let p_local  = migration_penalty_cycles(0,  ws);
    let p_near   = migration_penalty_cycles(10, ws);
    let p_far    = migration_penalty_cycles(20, ws);
    assert!(p_local <= p_near);
    assert!(p_near  <  p_far);
}

#[test]
fn penalty_monotone_in_working_set() {
    let p_small = migration_penalty_cycles(20, 16);
    let p_big   = migration_penalty_cycles(20, 1024);
    assert!(p_big > p_small);
}

#[test]
fn pure_function_property() {
    assert_eq!(migration_penalty_cycles(20, 256),
               migration_penalty_cycles(20, 256));
}
