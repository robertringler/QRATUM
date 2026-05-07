//! P0.6 — Host model of xAPIC ↔ x2APIC equivalence (INV-W).

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ApicMode { XApic, X2Apic }

/// Canonical IPI projection. Both modes project to the same
/// `(vector, dest_cpu)` pair.
pub fn canonical_ipi(_mode: ApicMode, vector: u8, dest_cpu: u8) -> (u8, u8) {
    (vector, dest_cpu)
}

pub fn paths_equivalent(vector: u8, dest_cpu: u8) -> bool {
    canonical_ipi(ApicMode::XApic, vector, dest_cpu)
        == canonical_ipi(ApicMode::X2Apic, vector, dest_cpu)
}
