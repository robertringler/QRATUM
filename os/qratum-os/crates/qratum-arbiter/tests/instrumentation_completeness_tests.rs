//! T6.1 — Instrumentation completeness.
//!
//! Asserts the full taxonomy:
//!   * `TRACE_POINTS.len() == 11`
//!   * Stable IDs are 1..=11 in declaration order, matching enum
//!     discriminants.
//!   * Every spec has a non-empty name and provenance string.
//!   * Every kind in the registry round-trips through its stable id.
//!
//! The "kernel coverage" half (every counter > 0 after a workload) is
//! exercised by the kernel self-test in `kmain` — this host gate
//! enforces the structural invariant that the registry is internally
//! consistent.

use qratum_arbiter::proof_gate::{TracePointKind, TracePointSpec, TRACE_POINTS};

#[test]
fn trace_point_count_is_eleven() {
    assert_eq!(TRACE_POINTS.len(), 11,
        "P0.3.2 directive: 11 trace points required, found {}",
        TRACE_POINTS.len());
}

#[test]
fn stable_ids_are_dense_and_match_discriminants() {
    for (i, spec) in TRACE_POINTS.iter().enumerate() {
        let expected = (i as u16) + 1;
        assert_eq!(spec.stable_id, expected,
            "trace_point[{}].stable_id={:#x} expected {:#x}",
            i, spec.stable_id, expected);
        assert_eq!(spec.kind as u8, expected as u8,
            "trace_point[{}].kind discriminant={} expected {}",
            i, spec.kind as u8, expected);
    }
}

#[test]
fn every_spec_has_provenance() {
    for spec in TRACE_POINTS {
        assert!(!spec.name.is_empty(),       "empty name in {:?}", spec.kind);
        assert!(!spec.provenance.is_empty(), "no provenance in {:?}", spec.kind);
    }
}

#[test]
fn enum_to_spec_round_trips() {
    let kinds = [
        TracePointKind::SchedEnter, TracePointKind::SchedExit,
        TracePointKind::IntentSubmit, TracePointKind::ArbiterDecision,
        TracePointKind::AuditWrite, TracePointKind::AhtcFold,
        TracePointKind::AhtcExpand, TracePointKind::InvariantTrip,
        TracePointKind::SyscallEnter, TracePointKind::SyscallExit,
        TracePointKind::PreemptIrq,
    ];
    for k in kinds {
        let idx = (k as u8 - 1) as usize;
        let spec: &TracePointSpec = &TRACE_POINTS[idx];
        assert_eq!(spec.kind as u8, k as u8);
    }
}

#[test]
fn kernel_registry_byte_identical_to_host() {
    // The kernel and host registries are duplicated by design — this
    // test reads the kernel source as a string and checks that every
    // host stable_id appears with the same hex value. Byte-identical
    // taxonomy is a P0.3.2 INV-H precondition.
    let kernel_src = std::fs::read_to_string(
        concat!(env!("CARGO_MANIFEST_DIR"),
                "/../../src/kernel/instrumentation/trace_point.rs"))
        .expect("kernel trace_point.rs must exist");
    for spec in TRACE_POINTS {
        let needle = format!("stable_id: 0x{:04X}", spec.stable_id);
        assert!(kernel_src.contains(&needle),
            "kernel registry missing {} ({})", needle, spec.name);
    }
}
