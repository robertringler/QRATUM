//! P0.5 — Cache locality. Higher reuse must produce strictly higher
//! locality_q than cold-only access.

use qratum_arbiter::numa_model::summarize;

#[test]
fn high_locality_score_dominates_cold() {
    let hot  = summarize(900, 100);
    let cold = summarize(100, 900);
    assert!(hot.locality_q > cold.locality_q);
    assert!(hot.locality_q  >= 800_000);
    assert!(cold.locality_q <= 200_000);
}

#[test]
fn empty_traffic_yields_zero() {
    assert_eq!(summarize(0, 0).locality_q, 0);
}
