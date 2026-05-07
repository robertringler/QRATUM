//! Trace-point taxonomy. Every observable kernel hot path MUST map to
//! exactly one variant. Adding a path without adding a `TracePointKind`
//! is an INV-F violation by construction (the registry is consumed by
//! `proof_gate::claim_registry`).

#![allow(dead_code)]

#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TracePointKind {
    SchedEnter      = 1,
    SchedExit       = 2,
    IntentSubmit    = 3,
    ArbiterDecision = 4,
    AuditWrite      = 5,
    AhtcFold        = 6,
    AhtcExpand      = 7,
    InvariantTrip   = 8,
    SyscallEnter    = 9,
    SyscallExit     = 10,
    PreemptIrq      = 11,
}

#[derive(Copy, Clone, Debug)]
pub struct TracePointSpec {
    pub kind:        TracePointKind,
    pub stable_id:   u16,
    /// Human-readable name. Stable for CI logs.
    pub name:        &'static str,
    /// Short rationale for INV-G provenance — every metric that cites
    /// this trace point must reference its `stable_id`.
    pub provenance:  &'static str,
}

/// Canonical registry. Order MUST match `TracePointKind` discriminants
/// 1..=N (asserted at compile time below).
pub const TRACE_POINTS: &[TracePointSpec] = &[
    TracePointSpec { kind: TracePointKind::SchedEnter,      stable_id: 0x0001,
        name: "sched.enter",       provenance: "scheduler dispatch entry" },
    TracePointSpec { kind: TracePointKind::SchedExit,       stable_id: 0x0002,
        name: "sched.exit",        provenance: "scheduler dispatch exit" },
    TracePointSpec { kind: TracePointKind::IntentSubmit,    stable_id: 0x0003,
        name: "intent.submit",     provenance: "raw intent ingress, pre-arbiter" },
    TracePointSpec { kind: TracePointKind::ArbiterDecision, stable_id: 0x0004,
        name: "arbiter.decision",  provenance: "verdict produced (Accept|Deny*)" },
    TracePointSpec { kind: TracePointKind::AuditWrite,      stable_id: 0x0005,
        name: "audit.write",       provenance: "audit frame appended (chain advanced)" },
    TracePointSpec { kind: TracePointKind::AhtcFold,        stable_id: 0x0006,
        name: "ahtc.fold",         provenance: "AHTC-K fold-or-insert outcome" },
    TracePointSpec { kind: TracePointKind::AhtcExpand,      stable_id: 0x0007,
        name: "ahtc.expand",       provenance: "AHTC-K replay expand" },
    TracePointSpec { kind: TracePointKind::InvariantTrip,   stable_id: 0x0008,
        name: "inv.trip",          provenance: "invariant violation" },
    TracePointSpec { kind: TracePointKind::SyscallEnter,    stable_id: 0x0009,
        name: "syscall.enter",     provenance: "syscall ingress (int 0x80 or syscall msr)" },
    TracePointSpec { kind: TracePointKind::SyscallExit,     stable_id: 0x000A,
        name: "syscall.exit",      provenance: "syscall return-to-user" },
    TracePointSpec { kind: TracePointKind::PreemptIrq,      stable_id: 0x000B,
        name: "preempt.irq",       provenance: "timer-driven preemption tick" },
];

const _: () = {
    // Stable-id parity with discriminants — order is the contract.
    let mut i = 0;
    while i < TRACE_POINTS.len() {
        let s = TRACE_POINTS[i];
        assert!(s.stable_id as usize == i + 1);
        assert!(s.kind as u8 as usize == i + 1);
        i += 1;
    }
};

pub fn lookup(kind: TracePointKind) -> &'static TracePointSpec {
    &TRACE_POINTS[(kind as u8 - 1) as usize]
}
