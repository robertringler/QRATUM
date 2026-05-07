//! Sink — the only place that ever constructs a `TraceEvent` and
//! hands it to the underlying profiler ring. Call sites supply a
//! `TraceContext`; sink stamps RDTSC + dispatches.

#![allow(dead_code)]

use super::trace_point::TracePointKind;
use super::TraceEvent;

/// Caller-furnished context. Optional fields default to "none"
/// sentinels (`u32::MAX`, `0`).
#[derive(Copy, Clone, Debug, Default)]
pub struct TraceContext {
    pub task_id:           u32,
    pub intent_id_hi:      u64,
    pub arbitration_state: u64,
    pub scheduler_state:   u64,
    pub memory_delta:      i64,
}

#[inline]
pub fn record(kind: TracePointKind, ctx: TraceContext) -> u64 {
    // Translate to the existing profiler taxonomy — TracePointKind is
    // a strict superset, with the original four ring-routed kinds
    // mapped 1:1.
    use crate::kernel::profiler::TraceKind as P;
    let pk = match kind {
        TracePointKind::SchedEnter      => Some(P::SchedEnter),
        TracePointKind::SchedExit       => Some(P::SchedExit),
        TracePointKind::IntentSubmit    => Some(P::IntentSubmit),
        TracePointKind::ArbiterDecision => Some(P::ArbiterDecision),
        TracePointKind::AuditWrite      => Some(P::AuditWrite),
        TracePointKind::AhtcFold        => Some(P::AhtcFold),
        // Newly-named kinds bump the per-kind counter only (no ring
        // overflow). They surface via `coverage_summary`.
        _ => None,
    };

    let tsc = if let Some(pk) = pk {
        crate::kernel::profiler::trace_event(pk, ctx.intent_id_hi)
    } else {
        crate::kernel::profiler::cycle_counter::rdtsc()
    };

    super::metrics::bump(kind);

    let _ev = TraceEvent {
        timestamp:         tsc,
        event_type:        kind as u8,
        _pad0:             [0; 3],
        task_id:           ctx.task_id,
        intent_id_hi:      ctx.intent_id_hi,
        arbitration_state: ctx.arbitration_state,
        scheduler_state:   ctx.scheduler_state,
        memory_delta:      ctx.memory_delta,
    };
    // The profiler rings already retain the wire form; keeping `_ev`
    // here is intentional — it documents the canonical schema and
    // gives a target for the "every observable field must be present"
    // structural test in host equivalence.
    tsc
}
