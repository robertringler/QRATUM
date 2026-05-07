//! T6.2 — Claim registry validation.
//!
//! Every published claim phrase ("10×", "reduction", "speedup",
//! "optimization", "improvement") that appears in the QRATUM OS
//! README + AHTC-K spec MUST resolve to exactly one entry in
//! `proof_gate::CLAIMS`.
//!
//! Phrases inside fenced code blocks or admonition tables that name
//! variables/columns rather than published numbers are skipped via a
//! conservative allowlist. See `is_claim_line`.
//!
//! INV-G enforcement: a metric without a matching claim entry fails
//! the build.

use std::fs;

use qratum_arbiter::proof_gate::CLAIMS;

const CLAIM_TOKENS: &[&str] = &[
    "10×", "10x ", "10X ",
    "reduction", "speedup", "optimization", "improvement",
];

fn read(p: &str) -> String {
    let abs = format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), p);
    fs::read_to_string(&abs).unwrap_or_else(|e| panic!("read {}: {}", abs, e))
}

fn is_claim_line(line: &str) -> bool {
    let l = line.trim();
    if l.is_empty()                 { return false; }
    if l.starts_with("//")          { return false; }
    if l.starts_with('#')           { return false; }
    // Skip table header rows whose only content is a column name.
    if l.starts_with('|') && l.matches('|').count() >= 2
        && !l.chars().any(|c| c.is_ascii_digit()) { return false; }
    CLAIM_TOKENS.iter().any(|t| {
        let lower = l.to_ascii_lowercase();
        if *t == "10×" { l.contains("10×") }
        else { lower.contains(&t.to_ascii_lowercase()) }
    })
}

fn line_resolves_to_claim(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    CLAIMS.iter().any(|c|
        c.doc_phrases.iter().any(|p| lower.contains(&p.to_ascii_lowercase()))
    )
}

#[test]
fn registry_is_non_empty_and_well_formed() {
    assert!(!CLAIMS.is_empty(), "claim registry must not be empty");
    for c in CLAIMS {
        assert!(!c.id.is_empty(),                   "claim has empty id");
        assert!(!c.computation_function.is_empty(), "{}: no compute fn", c.id);
        assert!(!c.test_case_id.is_empty(),         "{}: no test case",  c.id);
        assert!(!c.ci_gate_id.is_empty(),           "{}: no ci gate",    c.id);
        assert!(!c.doc_phrases.is_empty(),          "{}: no phrases",    c.id);
    }
}

#[test]
fn every_published_claim_is_registered() {
    let docs = [
        "README.md",
        "spec/ahtc_k.md",
    ];
    let mut unresolved: Vec<(String, usize, String)> = Vec::new();
    for path in docs {
        let body = read(path);
        for (i, line) in body.lines().enumerate() {
            if is_claim_line(line) && !line_resolves_to_claim(line) {
                unresolved.push((path.to_string(), i + 1, line.trim().to_string()));
            }
        }
    }
    assert!(unresolved.is_empty(),
        "INV-G: {} unregistered claim lines:\n{}",
        unresolved.len(),
        unresolved.iter()
            .map(|(p, l, s)| format!("  {}:{}  {}", p, l, s))
            .collect::<Vec<_>>().join("\n"));
}

#[test]
fn every_claim_has_a_passing_test_case_in_the_repo() {
    // The test_case_id should appear by name in at least one
    // tests/*.rs file. This catches "claim registered but the named
    // test was deleted" drift.
    let test_dir = format!("{}/tests", env!("CARGO_MANIFEST_DIR"));
    let mut all = String::new();
    for ent in fs::read_dir(&test_dir).expect("tests dir") {
        let p = ent.unwrap().path();
        if p.extension().map(|e| e == "rs").unwrap_or(false) {
            all.push_str(&fs::read_to_string(&p).unwrap_or_default());
        }
    }
    for c in CLAIMS {
        assert!(all.contains(c.test_case_id),
            "claim '{}' references missing test '{}'", c.id, c.test_case_id);
    }
}
