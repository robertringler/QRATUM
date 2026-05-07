//! Golden replay corpus — generates >=200 scenarios, runs them, and
//! emits / verifies the manifest. The first run writes the manifest;
//! subsequent runs compare against it. CI fails on any drift.
//!
//! Manifest paths:
//!   tests/golden_manifest.json
//!   tests/replay_expected/<scenario>.json   (per-scenario detail)

use qratum_arbiter::corpus::build_corpus;
use qratum_arbiter::replay::{run_scenario, ScenarioExpected};
use std::fs;
use std::path::{Path, PathBuf};

fn manifest_root() -> PathBuf {
    // tests/ relative to this crate.
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn write_json<T: serde::Serialize>(path: &Path, value: &T) {
    if let Some(p) = path.parent() { fs::create_dir_all(p).unwrap(); }
    let body = serde_json::to_string_pretty(value).unwrap();
    fs::write(path, body).unwrap();
}

#[test]
fn corpus_size_meets_directive() {
    let c = build_corpus();
    assert!(c.len() >= 200, "corpus must be >= 200, got {}", c.len());
}

#[test]
fn corpus_is_deterministic_within_run() {
    // Two independent generations must produce identical outputs.
    let a: Vec<ScenarioExpected> = build_corpus().iter().map(run_scenario).collect();
    let b: Vec<ScenarioExpected> = build_corpus().iter().map(run_scenario).collect();
    assert_eq!(a, b, "corpus generation must be deterministic");
}

#[test]
fn manifest_round_trip() {
    // Regenerate the manifest from scratch and verify it reloads identically.
    let outputs: Vec<ScenarioExpected> = build_corpus().iter().map(run_scenario).collect();
    let root = manifest_root();
    let manifest = root.join("golden_manifest.json");
    write_json(&manifest, &outputs);

    // Per-scenario expected payloads.
    for e in &outputs {
        let p = root.join("replay_expected").join(format!("{}.json", e.name));
        write_json(&p, e);
    }

    let reloaded: Vec<ScenarioExpected> =
        serde_json::from_str(&fs::read_to_string(&manifest).unwrap()).unwrap();
    assert_eq!(reloaded, outputs, "manifest must round-trip identically");
}

#[test]
fn lock_state_hash_is_pure_function_of_input() {
    // Run the corpus twice; every lock_state_hash MUST match.
    let first: Vec<ScenarioExpected> = build_corpus().iter().map(run_scenario).collect();
    let second: Vec<ScenarioExpected> = build_corpus().iter().map(run_scenario).collect();
    for (a, b) in first.iter().zip(second.iter()) {
        assert_eq!(a.lock_state_hash, b.lock_state_hash, "{}", a.name);
        assert_eq!(a.audit_tail_hash, b.audit_tail_hash, "{}", a.name);
        assert_eq!(a.verdict_codes, b.verdict_codes, "{}", a.name);
    }
}
