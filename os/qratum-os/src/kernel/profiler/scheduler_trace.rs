//! Scheduler trace ring — paired `SchedEnter`/`SchedExit` events plus
//! `AhtcFold` notifications. Same fixed-capacity, no-allocation
//! discipline as `arbitration_trace`.

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

static SCHED_ENTER:        AtomicU64 = AtomicU64::new(0);
static SCHED_EXIT:         AtomicU64 = AtomicU64::new(0);
static AHTC_FOLD:          AtomicU64 = AtomicU64::new(0);
static DROPPED:            AtomicU64 = AtomicU64::new(0);
static TOTAL_SCHED_CYCLES: AtomicU64 = AtomicU64::new(0);
static LAST_ENTER_TSC:     AtomicU64 = AtomicU64::new(0);

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct SchedulerSummary {
    pub sched_enter:    u64,
    pub sched_exit:     u64,
    pub ahtc_fold:      u64,
    pub dropped:        u64,
    pub total_scheduling_cycles: u64,
}

pub(super) fn push(ev: TraceEvent) {
    use super::TraceKind;
    match ev.kind {
        x if x == TraceKind::SchedEnter as u8 => {
            SCHED_ENTER.fetch_add(1, Ordering::Relaxed);
            LAST_ENTER_TSC.store(ev.tsc, Ordering::Relaxed);
        }
        x if x == TraceKind::SchedExit as u8 => {
            SCHED_EXIT.fetch_add(1, Ordering::Relaxed);
            let s = LAST_ENTER_TSC.load(Ordering::Relaxed);
            if s != 0 && ev.tsc >= s {
                TOTAL_SCHED_CYCLES.fetch_add(ev.tsc - s, Ordering::Relaxed);
            }
        }
        x if x == TraceKind::AhtcFold as u8 => {
            AHTC_FOLD.fetch_add(1, Ordering::Relaxed);
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

pub fn summary() -> SchedulerSummary {
    SchedulerSummary {
        sched_enter: SCHED_ENTER.load(Ordering::Relaxed),
        sched_exit:  SCHED_EXIT.load(Ordering::Relaxed),
        ahtc_fold:   AHTC_FOLD.load(Ordering::Relaxed),
        dropped:     DROPPED.load(Ordering::Relaxed),
        total_scheduling_cycles: TOTAL_SCHED_CYCLES.load(Ordering::Relaxed),
    }
}

pub(super) fn reset() {
    let mut r = RING.lock();
    r.head = 0;
    r.full = false;
    for s in r.slots.iter_mut() { *s = TraceEvent::EMPTY; }
    SCHED_ENTER.store(0, Ordering::Relaxed);
    SCHED_EXIT.store(0, Ordering::Relaxed);
    AHTC_FOLD.store(0, Ordering::Relaxed);
    DROPPED.store(0, Ordering::Relaxed);
    TOTAL_SCHED_CYCLES.store(0, Ordering::Relaxed);
    LAST_ENTER_TSC.store(0, Ordering::Relaxed);
}
