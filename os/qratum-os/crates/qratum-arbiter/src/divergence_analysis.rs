//! P0.6 — Divergence envelope analysis (host mirror of kernel's
//! `runtime_model::divergence_envelope`).

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Envelope {
    pub irq_jitter_cyc:    u64,
    pub apic_latency_cyc:  u64,
    pub replay_drift_cyc:  u64,
    pub queue_pressure:    u64,
}

impl Envelope {
    pub const fn defaults() -> Self {
        Self { irq_jitter_cyc: 10_000, apic_latency_cyc: 50_000,
               replay_drift_cyc: 1_000, queue_pressure: 64 }
    }
}

pub fn within(observed: Envelope, bounds: Envelope) -> bool {
    observed.irq_jitter_cyc   <= bounds.irq_jitter_cyc &&
    observed.apic_latency_cyc <= bounds.apic_latency_cyc &&
    observed.replay_drift_cyc <= bounds.replay_drift_cyc &&
    observed.queue_pressure   <= bounds.queue_pressure
}

pub fn is_divergent(observed: Envelope, bounds: Envelope) -> bool {
    !within(observed, bounds)
}

/// Returns the dimensions that breached `bounds` — useful for reports.
pub fn breaches(observed: Envelope, bounds: Envelope) -> Vec<&'static str> {
    let mut v = Vec::new();
    if observed.irq_jitter_cyc   > bounds.irq_jitter_cyc   { v.push("irq_jitter_cyc"); }
    if observed.apic_latency_cyc > bounds.apic_latency_cyc { v.push("apic_latency_cyc"); }
    if observed.replay_drift_cyc > bounds.replay_drift_cyc { v.push("replay_drift_cyc"); }
    if observed.queue_pressure   > bounds.queue_pressure   { v.push("queue_pressure"); }
    v
}
