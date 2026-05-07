//! P0.5 — LAPIC (local APIC) deterministic shim.
//!
//! Real MSR/MMIO programming is hardware-gated. This module exposes
//! the deterministic *register file* used by the kernel's IPI
//! delivery logic and the host model. Reads and writes are
//! reproducible across runs.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU32, Ordering};

use super::MAX_CPUS;

#[repr(u32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DeliveryMode {
    Fixed   = 0,
    Init    = 5,
    Startup = 6,
}

#[derive(Copy, Clone, Debug)]
pub struct Icr {
    pub vector:   u8,
    pub mode:     DeliveryMode,
    pub dest_cpu: u8,
}

static SENT: [AtomicU32; MAX_CPUS] = [
    AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0),
    AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0),
];

/// Send an IPI. Real impl writes to the ICR; here it bumps the
/// per-target counter so traces are observable.
pub fn send(icr: Icr) -> bool {
    if (icr.dest_cpu as usize) >= MAX_CPUS { return false; }
    SENT[icr.dest_cpu as usize].fetch_add(1, Ordering::AcqRel);
    true
}

#[inline]
pub fn ipi_count(cpu: u8) -> u32 {
    if (cpu as usize) >= MAX_CPUS { 0 }
    else { SENT[cpu as usize].load(Ordering::Acquire) }
}

pub fn reset_for_test() {
    for s in &SENT { s.store(0, Ordering::Release); }
}
