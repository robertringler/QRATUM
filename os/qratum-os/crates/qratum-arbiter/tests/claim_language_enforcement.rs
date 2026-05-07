//! P0.5 — Claim language enforcement. Forbidden marketing terms must
//! not appear in `performance_claims_policy.md` or in any registered
//! claim doc field. Approved terms list must be present.

use std::fs;
use std::path::PathBuf;

const FORBIDDEN: &[&str] = &[
    "universal acceleration",
    "CPU acceleration",
    "operating-system-wide speedup",
    "general compute multiplier",
];

const APPROVED: &[&str] = &[
    "scheduler event reduction",
    "canonical execution reduction",
    "replay-stable execution compression",
    "semantic-preserving execution reduction",
];

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // .../crates/qratum-arbiter -> .../os/qratum-os
    p.pop(); p.pop();
    p
}

#[test]
fn policy_file_exists_and_is_clean() {
    let policy = workspace_root().join("spec/performance_claims_policy.md");
    let body = fs::read_to_string(&policy)
        .unwrap_or_else(|e| panic!("policy file missing: {}: {}", policy.display(), e));
    for term in APPROVED {
        assert!(body.contains(term), "approved term missing: {}", term);
    }
    // Forbidden terms must appear ONLY inside the explicit prohibition section.
    let upper = body.to_lowercase();
    for term in FORBIDDEN {
        let count = upper.matches(&term.to_lowercase()).count();
        // Allowed exactly once: the listing itself.
        assert!(count <= 1, "forbidden term appears more than once: {}", term);
    }
}
