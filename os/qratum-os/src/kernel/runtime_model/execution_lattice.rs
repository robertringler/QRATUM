//! `ExecutionState` — the canonical lattice. Every field is a *view*
//! onto an existing subsystem (no duplicated truth). The lattice
//! is read-only from outside `runtime_model`; mutations happen in
//! the owning subsystem and are observed here via `snapshot()`.

use core::sync::atomic::{AtomicU64, Ordering};

use crate::kernel::ahtc_k;
use crate::kernel::audit;
use crate::kernel::intent;
use crate::kernel::sched;
use crate::kernel::smp;
use crate::kernel::tick;

/// Read-only view of arbiter state for the canonical lattice.
#[derive(Copy, Clone, Debug, Default)]
pub struct ArbiterStateView {
    pub inflight:        u32,
    pub holder_authority: u8,
    pub last_loc_tick:   u64,
}

/// Read-only view of scheduler state.
#[derive(Copy, Clone, Debug, Default)]
pub struct SchedulerStateView {
    pub current:        u32,
    pub ready_count:    u32,
    pub running_count:  u32,
    pub switches:       u64,
    pub preempt_switches: u64,
}

/// Read-only view of AHTC-K state.
#[derive(Copy, Clone, Debug, Default)]
pub struct AhtcStateView {
    pub submits:      u64,
    pub fold_hits:    u64,
    pub fold_inserts: u64,
    pub enqueue_ops:  u64,
    pub queue_len:    u32,
}

/// Read-only view of SMP per-core state. Only `online_count` and the
/// BSP id surface in P0.3; AP-side counters land in P0.3.1.
#[derive(Copy, Clone, Debug, Default)]
pub struct SmpStateView {
    pub online_count: u32,
    pub current_cpu:  u8,
    pub ipi_dropped:  u64,
    pub ipi_sent:     u64,
}

/// Audit cursor — monotonic seq plus persistent flush counter.
#[derive(Copy, Clone, Debug, Default)]
pub struct AuditCursorView {
    pub live_seq:       u64,
    pub flushed_seq:    u64,
}

/// Canonical ExecutionState. Equality of two snapshots ⇒ identical
/// kernel execution position (modulo unobservable hardware state).
#[derive(Copy, Clone, Debug, Default)]
pub struct ExecutionState {
    pub arbiter:    ArbiterStateView,
    pub scheduler:  SchedulerStateView,
    pub ahtc_k:     AhtcStateView,
    pub smp:        SmpStateView,
    pub audit:      AuditCursorView,
    pub loc_tick:   u64,
}

impl ExecutionState {
    /// Take a coherent snapshot. The order of reads is fixed and
    /// documented (audit-cursor first, tick last) so two snapshots
    /// taken at the same logical position are byte-identical.
    pub fn snapshot() -> Self {
        let live_seq = audit::count();
        let flushed  = ahtc_k::audit_persist_flushed_or_zero();

        let arbiter = ArbiterStateView {
            inflight:         intent::inflight(),
            holder_authority: intent::holder(),
            last_loc_tick:    tick::loc_now(),
        };

        let (ready, running) = sched_counts();
        let scheduler = SchedulerStateView {
            current:           sched_current() as u32,
            ready_count:       ready,
            running_count:     running,
            switches:          sched::switches() as u64,
            preempt_switches:  sched::preemptive_switches() as u64,
        };

        let s = ahtc_k::stats();
        let ahtc = AhtcStateView {
            submits:      s.submits,
            fold_hits:    s.fold_hits,
            fold_inserts: s.fold_inserts,
            enqueue_ops:  s.enqueue_ops,
            queue_len:    s.queue_len,
        };

        let smp_view = SmpStateView {
            online_count: smp::online_count() as u32,
            current_cpu:  smp::current_cpu(),
            ipi_dropped:  smp::ipi_stub::dropped() as u64,
            ipi_sent:     smp::ipi_stub::sent() as u64,
        };

        Self {
            arbiter,
            scheduler,
            ahtc_k: ahtc,
            smp:    smp_view,
            audit:  AuditCursorView { live_seq, flushed_seq: flushed },
            loc_tick: tick::loc_now(),
        }
    }
}

// -- helper accessors that don't exist as direct getters today -------------

fn sched_current() -> usize {
    sched::SCHED.lock().current
}

fn sched_counts() -> (u32, u32) {
    let s = sched::SCHED.lock();
    let mut ready = 0u32;
    let mut running = 0u32;
    for t in s.tasks.iter() {
        match t.state {
            sched::TaskState::Ready   => ready += 1,
            sched::TaskState::Running => running += 1,
            _ => {}
        }
    }
    (ready, running)
}

/// Global lattice instance — last-snapshot generation counter (debug only).
pub static KSTATE: KStateGen = KStateGen { gen: AtomicU64::new(0) };

pub struct KStateGen { pub gen: AtomicU64 }
impl KStateGen {
    pub fn bump(&self) -> u64 { self.gen.fetch_add(1, Ordering::Relaxed) + 1 }
    pub fn current(&self) -> u64 { self.gen.load(Ordering::Relaxed) }
}
