//! Host-side mirror of `kernel::proof_gate` and
//! `kernel::instrumentation::trace_point`. Identical layout — these
//! two definitions are the canonical source of truth that CI
//! cross-validates against the kernel.

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
    pub kind:       TracePointKind,
    pub stable_id:  u16,
    pub name:       &'static str,
    pub provenance: &'static str,
}

pub const TRACE_POINTS: &[TracePointSpec] = &[
    TracePointSpec { kind: TracePointKind::SchedEnter,      stable_id: 0x0001, name: "sched.enter",      provenance: "scheduler dispatch entry" },
    TracePointSpec { kind: TracePointKind::SchedExit,       stable_id: 0x0002, name: "sched.exit",       provenance: "scheduler dispatch exit" },
    TracePointSpec { kind: TracePointKind::IntentSubmit,    stable_id: 0x0003, name: "intent.submit",    provenance: "raw intent ingress, pre-arbiter" },
    TracePointSpec { kind: TracePointKind::ArbiterDecision, stable_id: 0x0004, name: "arbiter.decision", provenance: "verdict produced (Accept|Deny*)" },
    TracePointSpec { kind: TracePointKind::AuditWrite,      stable_id: 0x0005, name: "audit.write",      provenance: "audit frame appended (chain advanced)" },
    TracePointSpec { kind: TracePointKind::AhtcFold,        stable_id: 0x0006, name: "ahtc.fold",        provenance: "AHTC-K fold-or-insert outcome" },
    TracePointSpec { kind: TracePointKind::AhtcExpand,      stable_id: 0x0007, name: "ahtc.expand",      provenance: "AHTC-K replay expand" },
    TracePointSpec { kind: TracePointKind::InvariantTrip,   stable_id: 0x0008, name: "inv.trip",         provenance: "invariant violation" },
    TracePointSpec { kind: TracePointKind::SyscallEnter,    stable_id: 0x0009, name: "syscall.enter",    provenance: "syscall ingress (int 0x80 or syscall msr)" },
    TracePointSpec { kind: TracePointKind::SyscallExit,     stable_id: 0x000A, name: "syscall.exit",     provenance: "syscall return-to-user" },
    TracePointSpec { kind: TracePointKind::PreemptIrq,      stable_id: 0x000B, name: "preempt.irq",      provenance: "timer-driven preemption tick" },
];

#[derive(Copy, Clone, Debug)]
pub struct Claim {
    pub id:                   &'static str,
    pub metric_source:        TracePointKind,
    pub computation_function: &'static str,
    pub test_case_id:         &'static str,
    pub ci_gate_id:           &'static str,
    pub doc_phrases:          &'static [&'static str],
}

pub const CLAIMS: &[Claim] = &[
    Claim {
        id: "ahtc-k.scheduler.10x",
        metric_source: TracePointKind::AhtcFold,
        computation_function: "performance::fold_metrics",
        test_case_id: "ahtc_k_meets_strict_10x_at_70pct_duplication",
        ci_gate_id:   "P0.3.1 — strict 10× validation at 70 % duplication",
        doc_phrases: &["scheduler reduction", "scheduling reduction",
                       "10× scheduler", "10× scheduling", "≥ 10× reduction",
                       "≥10× reduction", "10× hard gate",
                       "1515", "3030"],
    },
    Claim {
        id: "ahtc-k.memory.reduction",
        metric_source: TracePointKind::AhtcFold,
        computation_function: "performance::fold_metrics",
        test_case_id: "baseline_vs_ahtc_k_real_reductions",
        ci_gate_id:   "P0.3.1 — performance truth (baseline vs AHTC-K)",
        doc_phrases: &["memory reduction", "10.64"],
    },
    Claim {
        id: "ahtc-k.event.reduction",
        metric_source: TracePointKind::AhtcFold,
        computation_function: "performance::fold_metrics",
        test_case_id: "baseline_vs_ahtc_k_real_reductions",
        ci_gate_id:   "P0.3.1 — performance truth (baseline vs AHTC-K)",
        doc_phrases: &["event reduction", "all four reduction", "reduction ratios",
                       "reduction ratio is monotone"],
    },
    Claim {
        id: "ahtc-k.arbitration.parity",
        metric_source: TracePointKind::ArbiterDecision,
        computation_function: "performance::fold_metrics",
        test_case_id: "arbitration_invariance",
        ci_gate_id:   "Run AHTC-K acceptance suite",
        doc_phrases: &["arbitration reduction", "arbitration parity"],
    },
    Claim {
        id: "kernel.host.equivalence.270",
        metric_source: TracePointKind::ArbiterDecision,
        computation_function: "kernel_equivalence::verify_corpus_equivalence",
        test_case_id: "corpus_270_kernel_arbiter_equivalence",
        ci_gate_id:   "Run kernel ↔ arbiter equivalence (270-corpus)",
        doc_phrases: &["270-scenario", "270 scenarios", "kernel↔arbiter"],
    },
    Claim {
        id: "determinism.4hash.receipt",
        metric_source: TracePointKind::AuditWrite,
        computation_function: "kernel_equivalence::compute_receipt",
        test_case_id: "determinism_receipt_is_byte_stable",
        ci_gate_id:   "4-hash determinism receipt is byte-stable across runs",
        doc_phrases: &["4-hash determinism receipt", "byte-stable"],
    },
    Claim {
        id: "latency.envelope.parity",
        metric_source: TracePointKind::SchedExit,
        computation_function: "performance::LatencyEnvelope::within",
        test_case_id: "host_self_parity_within_envelope",
        ci_gate_id:   "P0.3.1 — kernel↔host timing envelope",
        doc_phrases: &["latency envelope", "self-parity", "deviation"],
    },
];
