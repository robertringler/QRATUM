//! INV-Y — hardware jitter must remain inside declared bounds.

use qratum_arbiter::divergence_analysis::{Envelope, within};

#[test]
fn declared_bounds_pass_self_test() {
    let b = Envelope::defaults();
    let zero = Envelope::default();
    assert!(within(zero, b));
}

#[test]
fn boundary_value_within() {
    let b = Envelope::defaults();
    let edge = Envelope { irq_jitter_cyc: b.irq_jitter_cyc, ..Envelope::default() };
    assert!(within(edge, b));
}

#[test]
fn one_past_boundary_fails() {
    let b = Envelope::defaults();
    let over = Envelope { irq_jitter_cyc: b.irq_jitter_cyc + 1, ..Envelope::default() };
    assert!(!within(over, b));
}
