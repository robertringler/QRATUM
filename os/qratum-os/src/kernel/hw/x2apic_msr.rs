//! P0.7 — x2APIC MSR backend.

#![allow(dead_code)]

use super::msr::{
    rdmsr, wrmsr,
    IA32_X2APIC_APICID, IA32_X2APIC_EOI, IA32_X2APIC_ESR,
    IA32_X2APIC_ICR, IA32_X2APIC_SIVR, IA32_X2APIC_TPR,
};

const SIVR_ENABLE: u64 = 1 << 8;

/// Software-enable x2APIC. Caller must have already set
/// `IA32_APIC_BASE.X2APIC_EN` and `.GLOBAL_EN`.
///
/// # Safety
/// Running CPU must support x2APIC and have it enabled in APIC_BASE.
pub unsafe fn enable() {
    wrmsr(IA32_X2APIC_TPR, 0);
    wrmsr(IA32_X2APIC_SIVR, SIVR_ENABLE | 0xFF);
}

#[inline]
pub fn id() -> u32 { unsafe { rdmsr(IA32_X2APIC_APICID) as u32 } }

/// # Safety
/// Must be called from within an interrupt handler.
#[inline]
pub unsafe fn eoi() { wrmsr(IA32_X2APIC_EOI, 0); }

/// Send INIT IPI to `apic_id` (assert + deassert).
///
/// # Safety
/// Caller must hold IRQs off and have validated apic_id.
pub unsafe fn send_init(apic_id: u32) {
    wrmsr(IA32_X2APIC_ESR, 0);
    let dest = (apic_id as u64) << 32;
    // delivery=INIT(101), level=assert, trigger=level
    wrmsr(IA32_X2APIC_ICR, dest | (0b101 << 8) | (1 << 14) | (1 << 15));
    wait();
    // deassert
    wrmsr(IA32_X2APIC_ICR, dest | (0b101 << 8) | (1 << 15));
    wait();
}

/// Send STARTUP IPI with `page` (real-mode entry = page << 12).
///
/// # Safety
/// See [`send_init`]; `page` must point to a valid real-mode stub.
pub unsafe fn send_startup(apic_id: u32, page: u8) {
    wrmsr(IA32_X2APIC_ESR, 0);
    let dest = (apic_id as u64) << 32;
    // delivery=STARTUP(110), level=assert
    wrmsr(IA32_X2APIC_ICR, dest | (0b110 << 8) | (page as u64) | (1 << 14));
    wait();
}

unsafe fn wait() {
    // x2APIC ICR write is fire-and-forget; a small spin keeps the
    // sequence stable on virtual CPUs that need a visibility delay.
    for _ in 0..1000 { core::hint::spin_loop(); }
}
