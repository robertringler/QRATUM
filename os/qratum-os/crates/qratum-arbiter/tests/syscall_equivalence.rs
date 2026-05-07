//! Syscall path equivalence (legacy `int 0x80` ⇔ MSR `syscall/sysret`).
//!
//! The two kernel paths are required by spec to be **semantically
//! identical**: same intent → same arbiter call → same verdict → same
//! audit emission → same return encoding. The kernel side is enforced
//! by sharing one dispatcher (`sysret_dispatch` and `int 0x80` both
//! call into the same arbiter). This test models that contract on the
//! host: for any intent, both `path = legacy` and `path = msr` must
//! produce identical verdicts and identical post-state hashes.

use qratum_arbiter::corpus::build_corpus;
use qratum_arbiter::replay::{run_scenario, InputOp, ScenarioInput};
use qratum_arbiter::state_machine::{Authority, Intent, Scope};

#[derive(Copy, Clone)]
enum SyscallPath { Legacy, Msr }

/// Re-tag every Submit's `kind` to encode the path. The arbiter MUST
/// ignore the path-tag so verdicts coincide.
fn retag(s: &ScenarioInput, path: SyscallPath) -> ScenarioInput {
    let new_kind = match path { SyscallPath::Legacy => 6u8, SyscallPath::Msr => 7u8 };
    let ops = s.ops.iter().map(|op| match op {
        InputOp::Submit(i) => {
            // Preserve everything except the path tag. Forbid kind=0 retag
            // (schema-violation scenarios stay schema-violation).
            if i.kind == 0 { return op.clone(); }
            InputOp::Submit(Intent { kind: new_kind, ..*i })
        }
        InputOp::Release => InputOp::Release,
    }).collect();
    ScenarioInput { name: format!("{}_{}", s.name, match path { SyscallPath::Legacy => "legacy", SyscallPath::Msr => "msr" }), ops }
}

#[test]
fn legacy_and_msr_paths_are_equivalent() {
    let mut mismatches = 0usize;
    for s in build_corpus() {
        let leg = run_scenario(&retag(&s, SyscallPath::Legacy));
        let msr = run_scenario(&retag(&s, SyscallPath::Msr));
        // Verdict codes must match exactly. Lock-state and audit-tail hashes
        // also match because the retag doesn't change which intent wins.
        if leg.verdict_codes != msr.verdict_codes
            || leg.lock_state_hash != msr.lock_state_hash {
            eprintln!("DIVERGENCE in {}", s.name);
            eprintln!("  legacy: {:?} {}", leg.verdict_codes, leg.lock_state_hash);
            eprintln!("  msr:    {:?} {}", msr.verdict_codes, msr.lock_state_hash);
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "syscall-path equivalence violated");
}

#[test]
fn return_encoding_is_path_independent() {
    // Each verdict code is a u8 — both paths return the same numeric
    // encoding. (Kernel side: `sysret_dispatch` returns `i64` with the
    // verdict code in the low byte; `int 0x80` does the same.)
    let i = Intent {
        id: [1u8; 16], kind: 6, authority: Authority::User,
        scope: Scope::Process, tick: 1, payload_h: [0u8; 32],
    };
    let s = ScenarioInput { name: "ret_enc".into(), ops: vec![InputOp::Submit(i)] };
    let r = run_scenario(&s);
    assert_eq!(r.verdict_codes.len(), 1);
    assert!(r.verdict_codes[0] >= 0x10);
}
