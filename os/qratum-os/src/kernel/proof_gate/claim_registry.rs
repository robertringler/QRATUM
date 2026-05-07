//! Static registry of every performance claim in the QRATUM repo.
//!
//! Adding a published "10×" / "reduction" / "speedup" string anywhere
//! WITHOUT a corresponding entry here causes
//! `claim_registry_validation_tests::every_published_claim_is_registered`
//! to fail (build hard-fails in CI).

use super::super::instrumentation::trace_point::TracePointKind;

#[derive(Copy, Clone, Debug)]
pub struct Claim {
    /// Human-readable claim id (stable across phases).
    pub id:                  &'static str,
    /// Trace point that produces the underlying numbers.
    pub metric_source:       TracePointKind,
    /// Name of the host function that computes the published number.
    pub computation_function: &'static str,
    /// Host crate test that gates merges on this claim.
    pub test_case_id:        &'static str,
    /// CI workflow step name (in `.github/workflows/determinism.yml`).
    pub ci_gate_id:          &'static str,
    /// Phrases (lowercased, substring) that this claim covers in the
    /// public docs. Validator marks a doc match as "covered" if ANY
    /// phrase appears in the same line as the claim text.
    pub doc_phrases:         &'static [&'static str],
}

pub const CLAIMS: &[Claim] = &[
    Claim {
        id: "ahtc-k.scheduler.10x",
        metric_source:        TracePointKind::AhtcFold,
        computation_function: "performance::fold_metrics",
        test_case_id:         "ahtc_k_meets_strict_10x_at_70pct_duplication",
        ci_gate_id:           "P0.3.1 — strict 10× validation at 70 % duplication",
        doc_phrases: &[
            "scheduler reduction",
            "scheduling reduction",
            "10× scheduler",
            "10× scheduling",
            "≥ 10× reduction",
            "1515",
            "3030",
        ],
    },
    Claim {
        id: "ahtc-k.memory.reduction",
        metric_source:        TracePointKind::AhtcFold,
        computation_function: "performance::fold_metrics",
        test_case_id:         "baseline_vs_ahtc_k_real_reductions",
        ci_gate_id:           "P0.3.1 — performance truth (baseline vs AHTC-K)",
        doc_phrases: &["memory reduction", "10.64"],
    },
    Claim {
        id: "ahtc-k.event.reduction",
        metric_source:        TracePointKind::AhtcFold,
        computation_function: "performance::fold_metrics",
        test_case_id:         "baseline_vs_ahtc_k_real_reductions",
        ci_gate_id:           "P0.3.1 — performance truth (baseline vs AHTC-K)",
        doc_phrases: &["event reduction"],
    },
    Claim {
        id: "ahtc-k.arbitration.parity",
        metric_source:        TracePointKind::ArbiterDecision,
        computation_function: "performance::fold_metrics",
        test_case_id:         "arbitration_invariance",
        ci_gate_id:           "Run AHTC-K acceptance suite",
        doc_phrases: &["arbitration reduction", "arbitration parity"],
    },
    Claim {
        id: "kernel.host.equivalence.270",
        metric_source:        TracePointKind::ArbiterDecision,
        computation_function: "kernel_equivalence::verify_corpus_equivalence",
        test_case_id:         "corpus_270_kernel_arbiter_equivalence",
        ci_gate_id:           "Run kernel ↔ arbiter equivalence (270-corpus)",
        doc_phrases: &["270-scenario", "270 scenarios", "kernel↔arbiter"],
    },
    Claim {
        id: "determinism.4hash.receipt",
        metric_source:        TracePointKind::AuditWrite,
        computation_function: "kernel_equivalence::compute_receipt",
        test_case_id:         "determinism_receipt_is_byte_stable",
        ci_gate_id:           "4-hash determinism receipt is byte-stable across runs",
        doc_phrases: &["4-hash determinism receipt", "byte-stable"],
    },
    Claim {
        id: "latency.envelope.parity",
        metric_source:        TracePointKind::SchedExit,
        computation_function: "performance::LatencyEnvelope::within",
        test_case_id:         "host_self_parity_within_envelope",
        ci_gate_id:           "P0.3.1 — kernel↔host timing envelope",
        doc_phrases: &["latency envelope", "self-parity", "deviation"],
    },
];
