//! P0.7 — Real INIT/SIPI sequence driver.
//!
//! Procedure (Intel SDM Vol 3, §8.4.4 *Multiple-Processor Initialization*):
//!
//!   1. Allocate a 4 KiB frame `< 1 MiB`, copy the AP trampoline blob.
//!   2. For each AP advertised in MADT (apic_id ≠ BSP):
//!        a. Send INIT IPI; wait 10 ms.
//!        b. Send STARTUP IPI with vector = (frame_pa >> 12); wait 200 µs.
//!        c. Send STARTUP IPI again (per SDM); wait 200 µs.
//!        d. Poll for the AP's online flag in `smp::ap_boot::AP_ONLINE`.
//!
//! The full version waits using HPET. Without HPET we busy-spin a
//! conservative cycle count.
//!
//! **Phase 3 status**: the trampoline blob in `ap_trampoline.rs` is
//! currently a real-mode CLI+HLT stub. APs accept SIPI without
//! triple-faulting but never enter long mode / Rust. A full
//! 16→32→64 mode-switch trampoline is gated on QEMU host availability
//! (development without execution validation = guaranteed triple-fault).
//! `start_aps` therefore reports SIPI delivery but the polling loop
//! will time out and emit `[SMP] AP BOOT FAILED` per Phase 4 spec.

#![allow(dead_code)]

use crate::kprintln;
use crate::kernel::smp::{ap_boot, ap_startup};
use super::{acpi, ap_trampoline, apic_real, frame_alloc, hpet};

/// Bring up every AP listed in the MADT. Returns the count of APs
/// for which an INIT/SIPI sequence was issued (not necessarily the
/// count that actually came online — caller polls `ap_boot`).
///
/// `ap_stacks[i]` is the kernel stack top each AP should switch to
/// once it reaches long mode (consumed by the trampoline once the
/// long-mode bridge is added).
pub fn start_aps(ap_stacks: &[u64]) -> u8 {
    let topo = match acpi::topology() {
        Some(t) => t,
        None => return 0,
    };
    let entries = acpi::lapics();

    // Allocate the trampoline frame (< 1 MiB).
    let frame_pa = match frame_alloc::alloc_below(0x10_0000) {
        Ok(pa) => pa,
        Err(_) => {
            kprintln!("[SMP] AP BOOT FAILED: no <1MiB frame");
            return 0;
        }
    };
    unsafe { ap_trampoline::install(frame_pa); }
    let page = (frame_pa >> 12) as u8;

    let bsp_id = bsp_apic_id();
    let mut sipi_sent: u8 = 0;
    let target_count = topo.n_total.saturating_sub(1);

    for (i, e) in entries.iter().enumerate() {
        if !e.enabled || e.apic_id == bsp_id { continue; }
        let stack_top = ap_stacks.get(i).copied().unwrap_or(0);
        ap_boot::stage(e.processor_uid, e.apic_id, stack_top);

        kprintln!("[SMP] INIT SENT id=0x{:02X}", e.apic_id);

        unsafe {
            apic_real::send_init(e.apic_id);
            wait_us(10_000);
            apic_real::send_startup(e.apic_id, page);
            wait_us(200);
            apic_real::send_startup(e.apic_id, page);
            wait_us(200);

            // Synthetic startup receipt is intentionally NOT emitted here
            // until the trampoline can prove AP entry. Phase 4 polling
            // below is the canonical liveness probe.
            let _ = ap_startup::emit;
        }
        sipi_sent = sipi_sent.saturating_add(1);
    }

    // Phase 4 — poll AP_COUNT for up to ~1 second for an AP to
    // increment the online counter from inside ap_entry.
    let start_online = ap_boot::online_count();
    for _ in 0..1000 {
        wait_us(1000);
        let now = ap_boot::online_count();
        if now > start_online { break; }
    }
    let final_online = ap_boot::online_count();
    let new_aps = final_online.saturating_sub(start_online);

    if sipi_sent > 0 && new_aps == 0 {
        kprintln!("[SMP] AP BOOT FAILED (sipi_sent={}, online_delta=0) -- trampoline does not enter long mode", sipi_sent);
    } else if new_aps > 0 {
        kprintln!("[SMP] AP BOOTED count={}", new_aps);
    }

    let _ = target_count;
    sipi_sent
}

fn bsp_apic_id() -> u8 {
    use super::msr::{rdmsr, IA32_APIC_BASE, APIC_BASE_BSP};
    let base = unsafe { rdmsr(IA32_APIC_BASE) };
    if base & APIC_BASE_BSP == 0 { return 0; }
    match apic_real::active() {
        apic_real::Backend::X2Apic => super::x2apic_msr::id() as u8,
        apic_real::Backend::XApic  => unsafe {
            (super::lapic_mmio::read(super::lapic_mmio::REG_ID) >> 24) as u8
        },
        apic_real::Backend::None => 0,
    }
}

fn wait_us(us: u64) {
    if super::acpi::hpet_address().is_some() {
        hpet::spin_ns(us.saturating_mul(1000));
    } else {
        // ~1 cycle per loop, conservative on a 3 GHz host: 3000 cycles ≈ 1 µs.
        for _ in 0..(us * 3000) { core::hint::spin_loop(); }
    }
}
