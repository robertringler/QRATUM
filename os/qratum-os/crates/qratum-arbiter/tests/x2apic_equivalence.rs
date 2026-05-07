//! INV-W — xAPIC and x2APIC paths are execution-equivalent.

use qratum_arbiter::apic_model::{paths_equivalent, canonical_ipi, ApicMode};

#[test]
fn equivalence_holds_across_full_vector_range() {
    for v in 0u8..=255 {
        for cpu in 0u8..8 {
            assert!(paths_equivalent(v, cpu));
            let a = canonical_ipi(ApicMode::XApic, v, cpu);
            let b = canonical_ipi(ApicMode::X2Apic, v, cpu);
            assert_eq!(a, b);
        }
    }
}

#[test]
fn canonical_form_is_stable() {
    assert_eq!(canonical_ipi(ApicMode::XApic, 32, 0), (32, 0));
    assert_eq!(canonical_ipi(ApicMode::X2Apic, 32, 0), (32, 0));
}
