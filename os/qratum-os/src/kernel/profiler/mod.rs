//! P0.3.1 deterministic profiler spine.
//!
//! Three concerns, three submodules:
//!
//!   * [`cycle_counter`] — RDTSC reader, the single source of truth
//!     for kernel-side time. Inlined on every hot path.
//!   * [`arbitration_trace`] — fixed-capacity ring of arbiter decision
//!     events (`IntentSubmit`, `ArbiterDecision`, `AuditWrite`).
//!   * [`scheduler_trace`] — fixed-capacity ring of scheduler events
//!     (`SchedEnter`, `SchedExit`, `AhtcFold`).
//!
//! All trace storage is fixed-size and pre-allocated. The profiler
//! never allocates and never panics on overflow — it bumps a
//! `dropped` counter (visible in `summary()`) so the directive's
//! "audit system must not drop trace events" can be checked at exit.
//!
//! Determinism contract: timestamps are monotonic on a single CPU and
//! bytewise reproducible *across runs of the same input* once you
//! subtract the boot-time RDTSC base. The profiler therefore exposes
//! `summary()` (counters) for use by `runtime_model::execution_lattice`
//! and `delta_summary(prev) -> ProfilerSummary` for tests, both of
//! which are deterministic functions of the trace contents.

#![allow(dead_code)]

pub mod arbitration_trace;
pub mod cycle_counter;
pub mod scheduler_trace;
pub mod cache_metrics;
pub mod context_switch;
pub mod irq_latency;
pub mod tlb_metrics;
// P0.5 additions.
pub mod numa_metrics;
pub mod cache_locality;
pub mod migration_cost;
// P0.6 — hardware-bound determinism profilers.
pub mod apic_latency;
pub mod irq_jitter;
pub mod migration_heatmap;
pub mod queue_pressure;
pub mod replay_drift;

pub use cycle_counter::rdtsc;

/// Unified event kind. The two ring buffers (arbitration + scheduler)
/// share this taxonomy so the host crate can replay one merged stream.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TraceKind {
    SchedEnter      = 1,
    SchedExit       = 2,
    IntentSubmit    = 3,
    ArbiterDecision = 4,
    AuditWrite      = 5,
    AhtcFold        = 6,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct TraceEvent {
    pub kind: u8,
    pub _pad: [u8; 7],
    /// RDTSC at event entry.
    pub tsc:  u64,
    /// Event-specific payload (verdict, intent kind, batch idx, …).
    pub data: u64,
}

impl TraceEvent {
    pub const EMPTY: Self = Self { kind: 0, _pad: [0; 7], tsc: 0, data: 0 };
}

/// Aggregate counters surfaced to `runtime_model::execution_lattice`.
/// Strict closure: every `submit` MUST produce one `IntentSubmit`,
/// one `ArbiterDecision`, and at least one `AuditWrite`.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct ProfilerSummary {
    pub sched_enter:      u64,
    pub sched_exit:       u64,
    pub intent_submit:    u64,
    pub arbiter_decision: u64,
    pub audit_write:      u64,
    pub ahtc_fold:        u64,
    /// Total dropped trace events across all rings. MUST be zero in CI.
    pub dropped:          u64,
    /// Sum of per-event TSC deltas (entry → exit) over closed pairs.
    pub total_arbitration_cycles: u64,
    pub total_scheduling_cycles:  u64,
    pub total_audit_cycles:       u64,
}

/// Single source of truth for `runtime_model::execution_lattice`.
pub fn summary() -> ProfilerSummary {
    let a = arbitration_trace::summary();
    let s = scheduler_trace::summary();
    ProfilerSummary {
        sched_enter:      s.sched_enter,
        sched_exit:       s.sched_exit,
        intent_submit:    a.intent_submit,
        arbiter_decision: a.arbiter_decision,
        audit_write:      a.audit_write,
        ahtc_fold:        s.ahtc_fold,
        dropped:          a.dropped + s.dropped,
        total_arbitration_cycles: a.total_arbitration_cycles,
        total_scheduling_cycles:  s.total_scheduling_cycles,
        total_audit_cycles:       a.total_audit_cycles,
    }
}

/// Public hot-path helper. Routes the event to the correct ring and
/// stamps it with RDTSC. Returns the TSC for chaining (`SCHED_ENTER` →
/// `SCHED_EXIT` paired-cycle measurement).
#[inline]
pub fn trace_event(kind: TraceKind, data: u64) -> u64 {
    let tsc = rdtsc();
    let ev = TraceEvent { kind: kind as u8, _pad: [0; 7], tsc, data };
    match kind {
        TraceKind::SchedEnter | TraceKind::SchedExit | TraceKind::AhtcFold =>
            scheduler_trace::push(ev),
        TraceKind::IntentSubmit | TraceKind::ArbiterDecision | TraceKind::AuditWrite =>
            arbitration_trace::push(ev),
    }
    tsc
}

/// Reset both rings — test/QSH support only. Production paths must not
/// invoke this; the profiler is append-only by contract.
pub fn reset() {
    arbitration_trace::reset();
    scheduler_trace::reset();
}
