//! Symmetric multi-processing scaffolding (P0.3).
//!
//! Single-core today; xAPIC backend + AP boot land in P0.3.1. The
//! types and APIs here are the **final shape** kernel callers will
//! see — so wiring written today survives the multi-core flip.
//!
//! Spec: `spec/smp_model.md`.

#![allow(dead_code)]

pub mod ipi_stub;
pub mod core_local;
pub mod affinity;
pub mod barriers;
pub mod scheduler;
pub mod migration;
// P0.5 additions — live multicore execution fabric.
pub mod ap_boot;
pub mod lapic;
pub mod xapic;
pub mod runqueue;
pub mod migration_receipt;
pub mod deterministic_ticket;
pub mod lock_graph;
pub mod memory_order;
pub mod contention_trace;
// P0.6 — real multicore + hardware-bound determinism.
pub mod ap_startup;
pub mod lapic_timer;
pub mod x2apic;
pub mod intercore_queue;
pub mod deterministic_work_steal;
pub mod core_topology;
pub mod receipt_sync;
pub mod migration_epoch;
pub mod replay_barrier;
pub mod interrupt_receipt;

pub const MAX_CPUS: usize = 8;

/// Per-CPU state. Scheduler + audit segments are referenced through
/// `Option<&'static mut _>` slots assigned at AP-up time. P0.3 only
/// populates index 0 (the BSP).
pub struct PerCpu {
    pub cpu_id: u8,
    pub online: bool,
    pub invariant_violations: u64,
}

impl PerCpu {
    pub const fn new(cpu_id: u8) -> Self {
        Self { cpu_id, online: false, invariant_violations: 0 }
    }
}

static mut PERCPU: [PerCpu; MAX_CPUS] = [
    PerCpu::new(0), PerCpu::new(1), PerCpu::new(2), PerCpu::new(3),
    PerCpu::new(4), PerCpu::new(5), PerCpu::new(6), PerCpu::new(7),
];

/// Initialize SMP scaffolding (BSP only in P0.3).
pub fn init() {
    // SAFETY: single-CPU at this point in boot.
    unsafe { PERCPU[0].online = true; }
    ipi_stub::init();
}

/// Currently executing CPU id. P0.3: always 0.
#[inline]
pub fn current_cpu() -> u8 { 0 }

/// Reference to per-CPU state for the current core.
///
/// # Safety
/// Caller must not retain the reference across IPI boundaries; in
/// multi-core this becomes a percpu-segment relative access.
pub unsafe fn per_cpu() -> &'static mut PerCpu {
    &mut PERCPU[current_cpu() as usize]
}

/// Number of online CPUs.
pub fn online_count() -> usize {
    // SAFETY: read of online flag is benign single-byte; in multi-core
    // this reads from percpu memory under an atomic counter.
    unsafe { PERCPU.iter().filter(|c| c.online).count() }
}

// ============================================================
// CoreState + deterministic reconciliation (P0.3 Task 4)
// ============================================================

/// Per-core observable state. Aggregates of these compose the global
/// canonical `ExecutionState`. P0.3 populates index 0 only; P0.3.1
/// fills indices 1..=N as APs come online.
#[derive(Copy, Clone, Debug, Default)]
pub struct CoreState {
    pub cpu_id:        u8,
    pub online:        bool,
    pub tcb_queue_len: u32,
    pub local_tick:    u64,
    pub ipi_inbox_len: u32,
    pub audit_cursor:  u64,
}

/// Snapshot every core. Fixed iteration order (0..MAX_CPUS) so the
/// resulting array is byte-identical given identical kernel state.
pub fn snapshot_cores() -> [CoreState; MAX_CPUS] {
    let mut out = [CoreState::default(); MAX_CPUS];
    for (i, slot) in out.iter_mut().enumerate() {
        // SAFETY: each PerCpu is plain-data; multi-core P0.3.1 uses
        // percpu segment relative reads.
        let p = unsafe { &PERCPU[i] };
        slot.cpu_id = p.cpu_id;
        slot.online = p.online;
        // P0.3 single-core: all queues observed via BSP only.
        if i == 0 {
            slot.local_tick    = crate::kernel::tick::loc_now();
            slot.audit_cursor  = crate::kernel::audit::count();
            slot.ipi_inbox_len = ipi_stub::inbox_len(0);
        }
    }
    out
}

/// Reconcile per-core states into one canonical aggregate. The
/// reconciliation rule is **strict commutative monoid**:
///   - `online_count`        = Σ online
///   - `total_tcb_queue_len` = Σ tcb_queue_len
///   - `max_local_tick`      = max(local_tick) — single canonical clock
///   - `total_ipi_inbox_len` = Σ ipi_inbox_len
///   - `max_audit_cursor`    = max(audit_cursor) — global audit seq
///
/// Determinism: all reductions are over a fixed-order array — no
/// hashmaps, no thread-id ordering. Identical input arrays → identical
/// output (verified by `runtime_model_tests::reconcile_is_deterministic`).
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct ReconciledSmpState {
    pub online_count:        u32,
    pub total_tcb_queue_len: u64,
    pub max_local_tick:      u64,
    pub total_ipi_inbox_len: u64,
    pub max_audit_cursor:    u64,
}

pub fn reconcile_smp_states() -> ReconciledSmpState {
    let cores = snapshot_cores();
    reconcile(&cores)
}

pub fn reconcile(cores: &[CoreState]) -> ReconciledSmpState {
    let mut r = ReconciledSmpState::default();
    for c in cores {
        if c.online { r.online_count += 1; }
        r.total_tcb_queue_len = r.total_tcb_queue_len.saturating_add(c.tcb_queue_len as u64);
        r.total_ipi_inbox_len = r.total_ipi_inbox_len.saturating_add(c.ipi_inbox_len as u64);
        if c.local_tick   > r.max_local_tick   { r.max_local_tick   = c.local_tick; }
        if c.audit_cursor > r.max_audit_cursor { r.max_audit_cursor = c.audit_cursor; }
    }
    r
}
