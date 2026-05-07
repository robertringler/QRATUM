//! P0.6 — Divergence envelope (INV-Y).
//!
//! Bounds on tolerated hardware-jitter without flagging divergence.

#![allow(dead_code)]

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Envelope {
    pub irq_jitter_cyc:    u64,
    pub apic_latency_cyc:  u64,
    pub replay_drift_cyc:  u64,
    pub queue_pressure:    u64,
}

impl Envelope {
    pub const fn default_bounds() -> Self {
        Self {
            irq_jitter_cyc:   10_000,
            apic_latency_cyc: 50_000,
            replay_drift_cyc: 1_000,
            queue_pressure:   64,
        }
    }
}

pub fn within(observed: Envelope, bounds: Envelope) -> bool {
    observed.irq_jitter_cyc   <= bounds.irq_jitter_cyc &&
    observed.apic_latency_cyc <= bounds.apic_latency_cyc &&
    observed.replay_drift_cyc <= bounds.replay_drift_cyc &&
    observed.queue_pressure   <= bounds.queue_pressure
}

/// True if observed exceeds any bound (real divergence).
pub fn is_divergent(observed: Envelope, bounds: Envelope) -> bool {
    !within(observed, bounds)
}
