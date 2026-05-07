//! System-wide determinism harness.
//!
//! Verifies the three contracts from the P0.3 directive:
//!   1. identical binary → identical audit log
//!   2. identical inputs → identical final state hash
//!   3. replay mode = live mode equivalence
//!
//! Each contract maps to one test below. The host crate stands in for
//! both "live" and "replay" modes: the formal arbiter is the single
//! source of truth, and we exercise it through two independent code
//! paths (direct call vs. JSON round-tripped scenario) to detect any
//! accidental nondeterminism in the serialization layer.

use qratum_arbiter::corpus::build_corpus;
use qratum_arbiter::replay::{run_scenario, ScenarioInput};

fn json_round_trip(s: &ScenarioInput) -> ScenarioInput {
    let bytes = serde_json::to_vec(s).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[test]
fn contract_1_identical_binary_identical_log() {
    // Two runs in the same process == "identical binary"; they must
    // produce equal audit-tail hashes per scenario.
    let a: Vec<_> = build_corpus().iter().map(run_scenario).collect();
    let b: Vec<_> = build_corpus().iter().map(run_scenario).collect();
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.audit_tail_hash, y.audit_tail_hash, "{}", x.name);
    }
}

#[test]
fn contract_2_identical_inputs_identical_final_state() {
    let a: Vec<_> = build_corpus().iter().map(run_scenario).collect();
    let b: Vec<_> = build_corpus().iter().map(run_scenario).collect();
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.lock_state_hash, y.lock_state_hash, "{}", x.name);
    }
}

#[test]
fn contract_3_live_mode_equals_replay_mode() {
    // "Live" = direct in-memory; "Replay" = serialize → deserialize → run.
    for s in build_corpus() {
        let live = run_scenario(&s);
        let replayed = run_scenario(&json_round_trip(&s));
        assert_eq!(live, replayed, "live/replay drift in {}", s.name);
    }
}

#[test]
fn full_corpus_replay_within_perf_budget() {
    // Soft benchmark — must complete in well under a second on dev hardware.
    let corpus = build_corpus();
    let started = std::time::Instant::now();
    let mut last_hash = String::new();
    for s in &corpus {
        last_hash = run_scenario(s).lock_state_hash;
    }
    let elapsed = started.elapsed();
    assert!(elapsed.as_secs() < 5, "corpus replay too slow: {:?}", elapsed);
    assert!(!last_hash.is_empty());
    eprintln!("[perf] {} scenarios replayed in {:?}", corpus.len(), elapsed);
}
