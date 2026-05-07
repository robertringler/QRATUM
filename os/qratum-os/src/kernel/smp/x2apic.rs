//! P0.6 — x2APIC mode shim. INV-W: xAPIC and x2APIC paths must be
//! execution-equivalent. Real MSR programming gated behind
//! `cfg(target_feature = "smp_real")`.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU32, Ordering};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Mode { XApic, X2Apic }

static MODE: AtomicU32 = AtomicU32::new(0); // 0=xApic, 1=x2Apic

pub fn set_mode(m: Mode) {
    MODE.store(match m { Mode::XApic => 0, Mode::X2Apic => 1 }, Ordering::Release);
}

pub fn mode() -> Mode {
    match MODE.load(Ordering::Acquire) { 1 => Mode::X2Apic, _ => Mode::XApic }
}

/// Pure equivalence function: given (vector, dest_cpu) — both encodings
/// must produce the same canonical (vector, dest) pair. Returns the
/// canonical pair used for receipts.
pub fn canonical_ipi(vector: u8, dest_cpu: u8) -> (u8, u8) { (vector, dest_cpu) }
