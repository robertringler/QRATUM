//! P0.7 — **Hardware Reality Layer**.
//!
//! This module is the *real* hardware backend that replaces the
//! P0.2 PIC-shaped abstractions when the running CPU has the
//! relevant features. Order of bring-up is fixed:
//!
//!   1. [`cpuid`]     — feature detection (LAPIC, x2APIC, TSC, HPET).
//!   2. [`acpi`]      — locate RSDP, walk MADT/HPET tables.
//!   3. [`apic_real`] — pick xAPIC-MMIO or x2APIC-MSR backend, mask 8259.
//!   4. [`hpet`]      — if present, take over from PIT for time keeping.
//!   5. [`frame_alloc`] — bitmap allocator over the UEFI memory map.
//!   6. [`kernel_paging`] — switch CR3 to a kernel-owned PML4.
//!   7. [`init_sipi`] — INIT + SIPI sequence to bring up APs.
//!
//! `init()` runs the whole sequence; each step degrades gracefully
//! (returns `Status::Skipped`) if its feature is unavailable so the
//! BSP keeps running.
//!
//! **Determinism boundary**: hardware backends are inherently
//! non-deterministic (real timing). The deterministic core
//! (`runtime_model`, audit, ahtc_k) keeps using the canonical
//! event stream — this layer feeds *into* it but is not bounded by
//! P0.6 envelopes.

#![allow(dead_code)]

pub mod cpuid;
pub mod msr;
pub mod acpi;
pub mod lapic_mmio;
pub mod x2apic_msr;
pub mod apic_real;
pub mod hpet;
pub mod frame_alloc;
pub mod kernel_paging;
pub mod init_sipi;
pub mod ap_trampoline;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Status {
    /// Subsystem brought up successfully.
    Ok,
    /// Hardware feature absent — fell back to legacy.
    Skipped,
    /// Hardware feature present but bring-up failed.
    Failed,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct InitReport {
    pub cpuid:        bool,
    pub acpi:         bool,
    pub x2apic:       bool,
    pub xapic_mmio:   bool,
    pub hpet:         bool,
    pub frame_alloc:  bool,
    pub kernel_pml4:  bool,
    pub aps_started:  u8,
}

static mut LAST_REPORT: InitReport = InitReport {
    cpuid: false, acpi: false, x2apic: false, xapic_mmio: false,
    hpet: false, frame_alloc: false, kernel_pml4: false, aps_started: 0,
};

/// Run the full hardware bring-up.
///
/// # Safety
/// Must be called once, post-`ExitBootServices`, with interrupts
/// disabled. The function masks the legacy 8259 controller and
/// re-programmes paging — caller must not be running with interrupt
/// frames pending.
pub unsafe fn init(rsdp: Option<u64>, ap_stacks: &[u64]) -> InitReport {
    let mut r = InitReport::default();

    let feats = cpuid::detect();
    r.cpuid = true;

    // ACPI gate — without RSDP we cannot locate the LAPIC base, the MADT,
    // or the per-CPU APIC IDs, so the entire APIC + SIPI chain is skipped.
    let Some(addr) = rsdp else {
        LAST_REPORT = r;
        return r;
    };
    if acpi::parse(addr).is_err() {
        LAST_REPORT = r;
        return r;
    }
    r.acpi = true;

    let backend = apic_real::init(&feats);
    match backend {
        apic_real::Backend::X2Apic => r.x2apic = true,
        apic_real::Backend::XApic  => r.xapic_mmio = true,
        apic_real::Backend::None   => {} // PIC fallback already in apic.rs
    }

    if feats.hpet && acpi::hpet_address().is_some() {
        if hpet::init().is_ok() { r.hpet = true; }
    }

    if frame_alloc::init().is_ok() { r.frame_alloc = true; }
    if kernel_paging::take_ownership().is_ok() { r.kernel_pml4 = true; }

    r.aps_started = init_sipi::start_aps(ap_stacks);

    LAST_REPORT = r;
    r
}

#[inline]
pub fn last_report() -> InitReport { unsafe { LAST_REPORT } }
