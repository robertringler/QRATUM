//! P0.7 — Unified APIC backend selector.
//!
//! On `init` we read CPUID + IA32_APIC_BASE and pick:
//!   * x2APIC MSR backend if available, else
//!   * xAPIC MMIO backend if LAPIC is present, else
//!   * `Backend::None` (BSP keeps the legacy PIC path).

#![allow(dead_code)]

use super::cpuid::Features;
use super::msr::{rdmsr, wrmsr,
    IA32_APIC_BASE, APIC_BASE_GLOBAL_EN, APIC_BASE_X2APIC_EN};
use super::{lapic_mmio, x2apic_msr, acpi};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Backend { None, XApic, X2Apic }

static mut ACTIVE: Backend = Backend::None;

/// Bring up the best available LAPIC backend.
///
/// # Safety
/// Must be called once with IRQs disabled and the legacy 8259 already
/// fully masked by the caller (`pic::mask_all()`).
pub unsafe fn init(feats: &Features) -> Backend {
    if !feats.apic { ACTIVE = Backend::None; return Backend::None; }

    // Read APIC_BASE; preserve BSP bit, set GLOBAL_EN, optionally set X2APIC_EN.
    let mut base = rdmsr(IA32_APIC_BASE);
    base |= APIC_BASE_GLOBAL_EN;

    if feats.x2apic {
        base |= APIC_BASE_X2APIC_EN;
        wrmsr(IA32_APIC_BASE, base);
        x2apic_msr::enable();
        ACTIVE = Backend::X2Apic;
        return Backend::X2Apic;
    }

    wrmsr(IA32_APIC_BASE, base);
    let mmio = acpi::lapic_base().unwrap_or(base & 0xFFFF_F000);
    lapic_mmio::set_base(mmio);
    lapic_mmio::enable();
    ACTIVE = Backend::XApic;
    Backend::XApic
}

#[inline]
pub fn active() -> Backend { unsafe { ACTIVE } }

/// End-of-interrupt — dispatched to the active backend.
///
/// # Safety
/// Must be called from within an interrupt handler.
#[inline]
pub unsafe fn eoi() {
    match ACTIVE {
        Backend::X2Apic => x2apic_msr::eoi(),
        Backend::XApic  => lapic_mmio::eoi(),
        Backend::None   => {}
    }
}

/// Send INIT IPI to `apic_id`.
///
/// # Safety
/// IRQs off; apic_id from MADT.
pub unsafe fn send_init(apic_id: u8) {
    match ACTIVE {
        Backend::X2Apic => x2apic_msr::send_init(apic_id as u32),
        Backend::XApic  => lapic_mmio::send_init(apic_id),
        Backend::None   => {}
    }
}

/// Send STARTUP IPI with real-mode page vector.
///
/// # Safety
/// See [`send_init`].
pub unsafe fn send_startup(apic_id: u8, page: u8) {
    match ACTIVE {
        Backend::X2Apic => x2apic_msr::send_startup(apic_id as u32, page),
        Backend::XApic  => lapic_mmio::send_startup(apic_id, page),
        Backend::None   => {}
    }
}
