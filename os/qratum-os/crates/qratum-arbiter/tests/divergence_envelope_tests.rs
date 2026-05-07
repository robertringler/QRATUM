//! INV-Y — divergence envelope rejects schedule mutations.

use qratum_arbiter::divergence_analysis::{Envelope, within, is_divergent, breaches};

#[test]
fn within_bounds_holds() {
    let bounds = Envelope::defaults();
    let observed = Envelope { irq_jitter_cyc: 100, apic_latency_cyc: 200,
                              replay_drift_cyc: 10, queue_pressure: 4 };
    assert!(within(observed, bounds));
    assert!(!is_divergent(observed, bounds));
    assert!(breaches(observed, bounds).is_empty());
}

#[test]
fn jitter_breach_detected() {
    let bounds = Envelope::defaults();
    let observed = Envelope { irq_jitter_cyc: 999_999, ..Envelope::defaults() };
    assert!(is_divergent(observed, bounds));
    assert!(breaches(observed, bounds).contains(&"irq_jitter_cyc"));
}

#[test]
fn multiple_breaches_listed() {
    let bounds = Envelope::defaults();
    let observed = Envelope {
        irq_jitter_cyc: 999_999, apic_latency_cyc: 999_999,
        replay_drift_cyc: 1, queue_pressure: 1,
    };
    let b = breaches(observed, bounds);
    assert!(b.contains(&"irq_jitter_cyc"));
    assert!(b.contains(&"apic_latency_cyc"));
}
