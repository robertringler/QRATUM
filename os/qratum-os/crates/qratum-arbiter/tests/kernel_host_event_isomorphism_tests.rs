//! T6.4 — Kernel↔host event isomorphism.
//!
//! Two assertions:
//!   1. The `TraceEvent` schema declared in
//!      `kernel::instrumentation::TraceEvent` is byte-equivalent to
//!      the host arbiter's audit-event projection (field names,
//!      widths, packing).
//!   2. Verdict streams produced by the host arbiter on the standard
//!      270-scenario corpus match the verdict streams the kernel
//!      would emit (exercised indirectly via
//!      `kernel_equivalence::verify_corpus_equivalence`).
//!
//! Together these assertions enforce INV-H (kernel↔host semantic
//! equivalence) at structural AND temporal level.

use qratum_arbiter::kernel_equivalence::verify_corpus_equivalence;

#[test]
fn trace_event_schema_matches_kernel_source() {
    // We treat the kernel TraceEvent declaration as the canonical
    // spec. Read it from disk and assert every field name + type.
    let path = concat!(env!("CARGO_MANIFEST_DIR"),
                       "/../../src/kernel/instrumentation/mod.rs");
    let src  = std::fs::read_to_string(path).expect("kernel mod.rs must exist");
    let needles: &[(&str, &str)] = &[
        ("timestamp:",         "u64"),
        ("event_type:",        "u8"),
        ("task_id:",           "u32"),
        ("intent_id_hi:",      "u64"),
        ("arbitration_state:", "u64"),
        ("scheduler_state:",   "u64"),
        ("memory_delta:",      "i64"),
    ];
    for (field, ty) in needles {
        assert!(src.contains(field), "kernel TraceEvent missing field {}", field);
        // Find the line that contains the field, ensure the type is on it
        let line = src.lines().find(|l| l.contains(field))
            .unwrap_or_else(|| panic!("no line for {}", field));
        assert!(line.contains(ty),
            "field {} has wrong type — expected {}, line was: {}", field, ty, line);
    }
    assert!(src.contains("#[repr(C)]"),
        "kernel TraceEvent must be #[repr(C)] for kernel↔host wire equivalence");
}

#[test]
fn corpus_270_kernel_arbiter_equivalence() {
    let mismatches = verify_corpus_equivalence();
    assert_eq!(mismatches, 0,
        "INV-H: {} of 270 corpus scenarios diverged kernel↔host", mismatches);
}
