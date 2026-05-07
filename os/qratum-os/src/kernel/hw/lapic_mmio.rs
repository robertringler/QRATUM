//! P0.7 — xAPIC MMIO backend.
//!
//! Operates on a 4 KiB MMIO window mapped at the LAPIC physical
//! base (advertised in `IA32_APIC_BASE` MSR / ACPI MADT). Register
//! offsets are documented in the Intel SDM Vol 3.

#![allow(dead_code)]

use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{AtomicU64, Ordering};

pub const REG_ID:           usize = 0x020;
pub const REG_VERSION:      usize = 0x030;
pub const REG_TPR:          usize = 0x080;
pub const REG_EOI:          usize = 0x0B0;
pub const REG_LDR:          usize = 0x0D0;
pub const REG_DFR:          usize = 0x0E0;
pub const REG_SIVR:         usize = 0x0F0;
pub const REG_ESR:          usize = 0x280;
pub const REG_ICR_LOW:      usize = 0x300;
pub const REG_ICR_HIGH:     usize = 0x310;
pub const REG_LVT_TIMER:    usize = 0x320;
pub const REG_LVT_LINT0:    usize = 0x350;
pub const REG_LVT_LINT1:    usize = 0x360;
pub const REG_LVT_ERROR:    usize = 0x370;
pub const REG_INIT_COUNT:   usize = 0x380;
pub const REG_CUR_COUNT:    usize = 0x390;
pub const REG_DCR:          usize = 0x3E0;

pub const SIVR_ENABLE:      u32 = 1 << 8;
pub const LVT_MASKED:       u32 = 1 << 16;
pub const LVT_PERIODIC:     u32 = 1 << 17;

pub const ICR_DELIVERY_INIT:    u32 = 0b101 << 8;
pub const ICR_DELIVERY_STARTUP: u32 = 0b110 << 8;
pub const ICR_LEVEL_ASSERT:     u32 = 1 << 14;
pub const ICR_TRIGGER_LEVEL:    u32 = 1 << 15;

static BASE: AtomicU64 = AtomicU64::new(0);

/// Set the MMIO base. Must be called before any other op.
pub fn set_base(addr: u64) { BASE.store(addr, Ordering::Release); }

#[inline]
pub fn base() -> u64 { BASE.load(Ordering::Acquire) }

#[inline]
fn reg(off: usize) -> *mut u32 { (base() + off as u64) as *mut u32 }

/// # Safety
/// Caller must have set the MMIO base and the page must be mapped UC.
#[inline]
pub unsafe fn read(off: usize) -> u32 { read_volatile(reg(off)) }

/// # Safety
/// See [`read`].
#[inline]
pub unsafe fn write(off: usize, val: u32) { write_volatile(reg(off), val); }

/// Software-enable the LAPIC and set spurious vector to 0xFF.
///
/// # Safety
/// MMIO base must be set; CPU must support xAPIC.
pub unsafe fn enable() {
    write(REG_TPR, 0);
    write(REG_SIVR, SIVR_ENABLE | 0xFF);
    // Mask LINT0/LINT1 to prevent legacy NMI/EXTINT delivery.
    write(REG_LVT_LINT0, LVT_MASKED);
    write(REG_LVT_LINT1, LVT_MASKED);
    write(REG_LVT_ERROR, LVT_MASKED);
}

/// End-of-interrupt.
///
/// # Safety
/// Must be called from within an interrupt handler.
#[inline]
pub unsafe fn eoi() { write(REG_EOI, 0); }

/// Send INIT IPI (assert + deassert) to a target APIC ID.
///
/// # Safety
/// Caller must hold interrupts disabled and have validated that
/// `apic_id` is a real LAPIC ID from the MADT.
pub unsafe fn send_init(apic_id: u8) {
    write(REG_ESR, 0);
    write(REG_ICR_HIGH, (apic_id as u32) << 24);
    write(REG_ICR_LOW, ICR_DELIVERY_INIT | ICR_LEVEL_ASSERT | ICR_TRIGGER_LEVEL);
    wait_send();
    write(REG_ICR_HIGH, (apic_id as u32) << 24);
    write(REG_ICR_LOW, ICR_DELIVERY_INIT | ICR_TRIGGER_LEVEL);
    wait_send();
}

/// Send STARTUP IPI with vector `page` (real-mode start = page << 12).
///
/// # Safety
/// `page` must be a 4 KiB page number below 1 MiB containing a valid
/// real-mode entry stub.
pub unsafe fn send_startup(apic_id: u8, page: u8) {
    write(REG_ESR, 0);
    write(REG_ICR_HIGH, (apic_id as u32) << 24);
    write(REG_ICR_LOW, ICR_DELIVERY_STARTUP | (page as u32) | ICR_LEVEL_ASSERT);
    wait_send();
}

unsafe fn wait_send() {
    // ICR delivery-status bit 12 clears when the IPI is accepted.
    for _ in 0..1_000_000 {
        if (read(REG_ICR_LOW) & (1 << 12)) == 0 { return; }
        core::hint::spin_loop();
    }
}
