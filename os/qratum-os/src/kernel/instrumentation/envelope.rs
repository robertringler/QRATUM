//! Latency envelope re-exports — host-side authoritative definitions
//! live in `crates/qratum-arbiter/src/performance.rs`. The kernel
//! mirror is a constants-only view so the two cannot drift.

#![allow(dead_code)]

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct EpsilonBounds {
    pub arbitration_pct:  u32,
    pub scheduling_pct:   u32,
    pub syscall_pct:      u32,
    pub ahtc_fold_pct:    u32,
}

impl EpsilonBounds {
    /// Directive-mandated bounds. P0.3.2 tightens AHTC-K fold to 2 %.
    pub const STRICT: Self = Self {
        arbitration_pct: 5,
        scheduling_pct:  10,
        syscall_pct:     3,
        ahtc_fold_pct:   2,
    };
}

#[derive(Copy, Clone, Debug, Default)]
pub struct LatencyEnvelope {
    pub kernel_arbitration_cycles: u64,
    pub host_arbitration_ns:       u64,
    pub kernel_scheduling_cycles:  u64,
    pub host_scheduling_ns:        u64,
    pub kernel_syscall_cycles:     u64,
    pub host_syscall_ns:           u64,
    pub kernel_ahtc_fold_cycles:   u64,
    pub host_ahtc_fold_ns:         u64,
}
