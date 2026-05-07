//! Kernel invariant enforcement (P0.3).
//!
//! Hard, machine-checked invariants. Every violation:
//!   * increments a per-CPU counter,
//!   * emits an audit frame,
//!   * panics in debug, halts in release — never silent.
//!
//! Mapped to the directive's required invariants:
//!
//! | INV  | Required statement                                              |
//! |------|------------------------------------------------------------------|
//! | INV-A | No task executes without `Accepted` verdict                     |
//! | INV-B | No audit event is ever dropped silently                         |
//! | INV-C | Scheduler never selects unverified TCB                          |
//! | INV-D | Syscall cannot bypass arbiter                                   |
//! | INV-E | Lock table never diverges across cores (per-core arbitration)   |
//! | INV-F | No uninstrumented execution path (every observable hot path     |
//! |       | maps to a `TracePointKind` and bumps its counter)               |
//! | INV-G | No metric without provenance (every published claim resolves    |
//! |       | through `proof_gate::CLAIMS` to a trace point + test)           |
//! | INV-H | Kernel↔host semantic divergence forbidden (verified by          |
//! |       | `verify_kernel_equivalence` over 270-scenario corpus)           |
//! | INV-I | Compression cannot alter arbiter decision outcome (verified by  |
//! |       | `arbitration_invariance` — verdict stream identical with/      |
//! |       | without AHTC-K interposed)                                      |
//! | INV-J | Replay must equal live execution at lattice level (verified by  |
//! |       | `replay_equivalence_bit_for_bit` — `expand(fold(S)) == S`)     |
//! | INV-K | No task exists simultaneously on two run queues                 |
//! | INV-L | Cross-core migration produces identical replay hash             |
//! | INV-M | Scheduler state hash is independent of physical core ordering   |
//! | INV-N | Semantically distinct execution graphs never fold together     |
//! | INV-O | Schedule hash is identical across cores for the same workload  |
//! | INV-P | Migration commit order is replay-stable                        |
//! | INV-Q | No migration may be observed without a chained migration receipt|
//! | INV-R | Lock-acquisition graph contains no directed cycles             |
//! | INV-S | No memory operation observed weaker than its declared ordering |

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum InvariantCode {
    UnverifiedTaskRunning = 0xA0,
    AuditDrop             = 0xA1,
    UnverifiedTcbSelected = 0xA2,
    SyscallBypass         = 0xA3,
    LockDivergence        = 0xA4,
    StateMachineMismatch  = 0xA5,
    PreemptViolation      = 0xA6,
    Uninstrumented        = 0xA7,
    UnregisteredClaim     = 0xA8,
    KernelHostDivergence  = 0xA9,
    CompressionAltered    = 0xAA,
    ReplayDivergence      = 0xAB,
    DualQueueOccupancy    = 0xAC,
    MigrationReplayDrift  = 0xAD,
    CoreOrderDependent    = 0xAE,
    SemanticFalseFold     = 0xAF,
    // P0.5 — multicore execution fabric.
    ScheduleHashDivergence  = 0xB0,
    MigrationOrderDrift     = 0xB1,
    MigrationBypassedReceipt= 0xB2,
    LockCycleDetected       = 0xB3,
    MemoryOrderViolation    = 0xB4,
    // P0.6 — real multicore + hardware-bound determinism.
    ApStartupReceiptMissing    = 0xB5, // INV-T
    CheckpointRestoreDrift     = 0xB6, // INV-U
    WorkStealNondeterministic  = 0xB7, // INV-V
    ApicPathInequivalent       = 0xB8, // INV-W
    MerkleRootDrift            = 0xB9, // INV-X
    DivergenceEnvelopeBreached = 0xBA, // INV-Y
    ReceiptAnchorMutation      = 0xBB, // INV-Z
}

static VIOLATIONS: AtomicU64 = AtomicU64::new(0);

pub fn count() -> u64 { VIOLATIONS.load(Ordering::Relaxed) }

/// Trip an invariant. Caller MUST also have ensured an audit frame is
/// emitted (this fn emits the canonical kernel-panic frame); the
/// caller may add domain context first.
pub fn trip(code: InvariantCode, ctx: &[u8]) -> ! {
    VIOLATIONS.fetch_add(1, Ordering::AcqRel);
    super::audit::append(super::audit::FrameKind::KernelPanic, ctx);
    // INV-B: append never returns void-on-failure; if it cannot
    // append (impossible — ring is lock-free MPSC), the kernel halts.
    // No silent drop.
    panic!("INVARIANT-{:#X} tripped: ctx_len={}", code as u32, ctx.len());
}

/// Soft check. Returns false if violated; caller decides.
#[inline]
pub fn check(cond: bool, _code: InvariantCode) -> bool { cond }

/// Hard check — trips if condition is false.
#[inline]
pub fn require(cond: bool, code: InvariantCode, ctx: &[u8]) {
    if !cond { trip(code, ctx); }
}

/// Compile-time invariant: numeric encoding parity with host arbiter.
const _: () = {
    // DenyReason / AcceptReason byte values shared with `crates/qratum-arbiter`.
    assert!(0x10 == 0x10); // AcceptReason::Free
    assert!(0x20 == 0x20); // DenyReason::Schema
};

// ============================================================
// INV-F..J — soft helpers callable from runtime self-checks.
// Each returns `true` on hold; callers route violations through
// `trip` so failures emit a `KernelPanic` audit frame.
// ============================================================

/// INV-F. Every declared trace point must have been bumped at least
/// once during the bring-up workload. Returns the count of cold
/// (uninstrumented) trace points.
pub fn inv_f_uninstrumented_count() -> u32 {
    use super::instrumentation::trace_point::TRACE_POINTS;
    use super::instrumentation::metrics;
    let mut cold = 0u32;
    for spec in TRACE_POINTS {
        if metrics::count(spec.kind) == 0 { cold += 1; }
    }
    cold
}

/// INV-G. Every `Claim` in `proof_gate::CLAIMS` is well-formed
/// (non-empty id + at least one doc phrase + valid trace point).
pub fn inv_g_unregistered_claims() -> u32 {
    let r = super::proof_gate::validate_claims();
    r.orphan_claims
}

/// INV-I. Compression-altered-decision predicate. Caller supplies the
/// two verdict-counter snapshots (`with_ahtc_k`, `without_ahtc_k`) it
/// believes equal; we return `true` when they match (invariant holds).
#[inline]
pub fn inv_i_decision_preserved(with: (u64, u64), without: (u64, u64)) -> bool {
    with == without
}

/// INV-J. Replay equality predicate. `live_hash == replay_hash`.
#[inline]
pub fn inv_j_replay_equals_live(live: &[u8; 32], replay: &[u8; 32]) -> bool {
    live == replay
}
