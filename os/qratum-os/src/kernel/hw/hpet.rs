//! P0.7 — HPET driver (high-precision event timer).
//!
//! Provides a monotone 64-bit nanosecond clock when present. Used as
//! a calibration reference for the LAPIC timer and as a fallback
//! tick source.

#![allow(dead_code)]

use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{AtomicU64, Ordering};

use super::acpi;

const REG_CAPS:    usize = 0x000;
const REG_CONFIG:  usize = 0x010;
const REG_COUNTER: usize = 0x0F0;

const CONFIG_ENABLE: u64 = 1 << 0;

static BASE: AtomicU64 = AtomicU64::new(0);
static FEMTO_PER_TICK: AtomicU64 = AtomicU64::new(0);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HpetError { Absent, BadCaps }

/// Initialize the HPET. Reads address from ACPI HPET table.
///
/// # Safety
/// Caller must have completed ACPI parsing.
pub unsafe fn init() -> Result<(), HpetError> {
    let addr = acpi::hpet_address().ok_or(HpetError::Absent)?;
    BASE.store(addr, Ordering::Release);

    let caps = read_volatile((addr + REG_CAPS as u64) as *const u64);
    let femto = (caps >> 32) & 0xFFFF_FFFF;
    if femto == 0 || femto > 0x05F5_E100 { return Err(HpetError::BadCaps); }
    FEMTO_PER_TICK.store(femto, Ordering::Release);

    // Enable main counter.
    let cfg = read_volatile((addr + REG_CONFIG as u64) as *mut u64);
    write_volatile((addr + REG_CONFIG as u64) as *mut u64, cfg | CONFIG_ENABLE);
    Ok(())
}

/// 64-bit monotone counter (raw ticks).
#[inline]
pub fn counter() -> u64 {
    let b = BASE.load(Ordering::Acquire);
    if b == 0 { return 0; }
    unsafe { read_volatile((b + REG_COUNTER as u64) as *const u64) }
}

/// Convert raw ticks to nanoseconds.
#[inline]
pub fn nanos(ticks: u64) -> u64 {
    let f = FEMTO_PER_TICK.load(Ordering::Acquire);
    if f == 0 { return 0; }
    ticks.saturating_mul(f) / 1_000_000
}

/// Busy-wait for `ns` nanoseconds. Fine for short waits (< 1 ms);
/// caller should use the scheduler for longer sleeps.
pub fn spin_ns(ns: u64) {
    let f = FEMTO_PER_TICK.load(Ordering::Acquire);
    if f == 0 { return; }
    let target_ticks = ns.saturating_mul(1_000_000) / f;
    let start = counter();
    while counter().wrapping_sub(start) < target_ticks {
        core::hint::spin_loop();
    }
}
