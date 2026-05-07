//! Arbitration trace ring — fixed-capacity, no-allocation.
//!
//! Captures `IntentSubmit` → `ArbiterDecision` → `AuditWrite` triples.
//! On overflow the ring increments `DROPPED` rather than overwriting,
//! because the directive's "audit system must not drop trace events"
//! is checked numerically by `verify_kernel_equivalence`.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};
use spin::Mutex;

use super::TraceEvent;

const RING_CAP: usize = 4096;

struct Ring {
    slots: [TraceEvent; RING_CAP],
    head:  usize,
    full:  bool,
}

impl Ring {
    const fn new() -> Self {
        Self { slots: [TraceEvent::EMPTY; RING_CAP], head: 0, full: false }
    }
}

static RING: Mutex<Ring> = Mutex::new(Ring::new());

static INTENT_SUBMIT:    AtomicU64 = AtomicU64::new(0);
static ARBITER_DECISION: AtomicU64 = AtomicU64::new(0);
static AUDIT_WRITE:      AtomicU64 = AtomicU64::new(0);
static DROPPED:          AtomicU64 = AtomicU64::new(0);
static TOTAL_ARB_CYCLES: AtomicU64 = AtomicU64::new(0);
static TOTAL_AUD_CYCLES: AtomicU64 = AtomicU64::new(0);
/// Last `IntentSubmit` TSC, for paired-cycle measurement against the
/// next `ArbiterDecision`. Single-writer (arbiter is single-CPU in P0.3).
static LAST_SUBMIT_TSC:  AtomicU64 = AtomicU64::new(0);
static LAST_DECIDE_TSC:  AtomicU64 = AtomicU64::new(0);

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct ArbitrationSummary {
    pub intent_submit:    u64,
    pub arbiter_decision: u64,
    pub audit_write:      u64,
    pub dropped:          u64,
    pub total_arbitration_cycles: u64,
    pub total_audit_cycles:       u64,
}

pub(super) fn push(ev: TraceEvent) {
    use super::TraceKind;
    // Counter bookkeeping: this is the source of truth even if the
    // ring overflows — the ring is for replay, the counters for proof.
    match ev.kind {
        x if x == TraceKind::IntentSubmit as u8 => {
            INTENT_SUBMIT.fetch_add(1, Ordering::Relaxed);
            LAST_SUBMIT_TSC.store(ev.tsc, Ordering::Relaxed);
        }
        x if x == TraceKind::ArbiterDecision as u8 => {
            ARBITER_DECISION.fetch_add(1, Ordering::Relaxed);
            let s = LAST_SUBMIT_TSC.load(Ordering::Relaxed);
            if s != 0 && ev.tsc >= s {
                TOTAL_ARB_CYCLES.fetch_add(ev.tsc - s, Ordering::Relaxed);
            }
            LAST_DECIDE_TSC.store(ev.tsc, Ordering::Relaxed);
        }
        x if x == TraceKind::AuditWrite as u8 => {
            AUDIT_WRITE.fetch_add(1, Ordering::Relaxed);
            let d = LAST_DECIDE_TSC.load(Ordering::Relaxed);
            if d != 0 && ev.tsc >= d {
                TOTAL_AUD_CYCLES.fetch_add(ev.tsc - d, Ordering::Relaxed);
            }
        }
        _ => {}
    }

    let mut r = RING.lock();
    let h = r.head;
    if h >= RING_CAP {
        DROPPED.fetch_add(1, Ordering::Relaxed);
        return;
    }
    r.slots[h] = ev;
    r.head = h + 1;
    if r.head == RING_CAP { r.full = true; }
}

pub fn summary() -> ArbitrationSummary {
    ArbitrationSummary {
        intent_submit:    INTENT_SUBMIT.load(Ordering::Relaxed),
        arbiter_decision: ARBITER_DECISION.load(Ordering::Relaxed),
        audit_write:      AUDIT_WRITE.load(Ordering::Relaxed),
        dropped:          DROPPED.load(Ordering::Relaxed),
        total_arbitration_cycles: TOTAL_ARB_CYCLES.load(Ordering::Relaxed),
        total_audit_cycles:       TOTAL_AUD_CYCLES.load(Ordering::Relaxed),
    }
}

pub(super) fn reset() {
    let mut r = RING.lock();
    r.head = 0;
    r.full = false;
    for s in r.slots.iter_mut() { *s = TraceEvent::EMPTY; }
    INTENT_SUBMIT.store(0, Ordering::Relaxed);
    ARBITER_DECISION.store(0, Ordering::Relaxed);
    AUDIT_WRITE.store(0, Ordering::Relaxed);
    DROPPED.store(0, Ordering::Relaxed);
    TOTAL_ARB_CYCLES.store(0, Ordering::Relaxed);
    TOTAL_AUD_CYCLES.store(0, Ordering::Relaxed);
    LAST_SUBMIT_TSC.store(0, Ordering::Relaxed);
    LAST_DECIDE_TSC.store(0, Ordering::Relaxed);
}
