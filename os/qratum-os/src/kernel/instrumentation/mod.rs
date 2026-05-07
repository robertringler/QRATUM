//! P0.3.2 — Hard instrumentation enforcement layer.
//!
//! `instrumentation::*` is the single canonical taxonomy for every
//! observable kernel event. It sits *above* `profiler::` (which owns
//! the rings) and *below* every call site. Direct calls to
//! `profiler::trace_event` from outside this module are still
//! permitted today (the profiler is an interface), but the
//! `proof_gate::claim_registry` requires every published metric to
//! resolve to a `TracePoint` defined here.
//!
//! Trace-point coverage is asserted by
//! `tests/instrumentation_completeness_tests.rs` — every variant of
//! `TracePointKind` MUST be both
//!   * declared in [`TRACE_POINTS`] with a stable id, and
//!   * exercised (counter > 0) at least once during the standard
//!     bring-up workload (`harness::run_min_workload`).
//!
//! INV-F (no-uninstrumented-execution) is the kernel-side enforcer.

#![allow(dead_code)]

pub mod envelope;
pub mod metrics;
pub mod sink;
pub mod trace_point;

pub use envelope::{EpsilonBounds, LatencyEnvelope};
pub use metrics::{CompressionMetrics, shannon_entropy_bits, ENTROPY_SCALE};
pub use sink::{record, TraceContext};
pub use trace_point::{TracePointKind, TracePointSpec, TRACE_POINTS};

/// Strict structural form of every observable kernel event. Mirrored
/// in the host crate (`performance::TraceContext`) so the two halves
/// can be compared field-for-field.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct TraceEvent {
    /// RDTSC at event entry.
    pub timestamp:         u64,
    /// `TracePointKind as u8` — stable wire-format byte.
    pub event_type:        u8,
    pub _pad0:             [u8; 3],
    /// Scheduler task id (or `u32::MAX` if pre-scheduler).
    pub task_id:           u32,
    /// Intent id high half — 8 bytes is enough for replay key.
    pub intent_id_hi:      u64,
    /// Snapshot of `(holder_authority, inflight)` packed.
    pub arbitration_state: u64,
    /// Snapshot of `(current << 32 | ready_count)`.
    pub scheduler_state:   u64,
    /// Net change to resident batch memory (signed).
    pub memory_delta:      i64,
}

impl TraceEvent {
    pub const EMPTY: Self = Self {
        timestamp: 0, event_type: 0, _pad0: [0; 3],
        task_id: u32::MAX, intent_id_hi: 0,
        arbitration_state: 0, scheduler_state: 0, memory_delta: 0,
    };
}
