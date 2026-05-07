//! Kernel main — runs after ExitBootServices (Agent 7, Agent 12).
//!
//! Boot-up order is fixed and audit-logged:
//!   1. serial   — console first, so panics are visible
//!   2. gdt      — owned segments incl. user CS/DS for ring-3 readiness
//!   3. idt      — exceptions + IRQs + int 0x80 syscall gate (DPL=3)
//!   4. pic      — remap to 0x20/0x28, mask all
//!   5. pit      — 100 Hz scheduler tick
//!   6. heap     — kernel-only manual allocator (1 MiB arena)
//!   7. sched    — task[0] = current thread (qsh)
//!   8. spawn    — heartbeat task (proves criterion #3)
//!   9. unmask + sti — interrupts go live
//!  10. qsh      — interactive shell

use crate::kprintln;
use crate::shared::BootInfo;

use super::audit::{self, FrameKind};
use super::{apic, gdt, heap, heartbeat, hw, idt, paging, pit, qsh, sched, serial, smp, sysc};

pub fn kernel_main(boot: &BootInfo) -> ! {
    serial::init();
    audit::append(FrameKind::BootBegin, b"qratum kernel_main");

    print_banner(boot);

    kprintln!("[KERN] init gdt+tss");
    gdt::init();
    kprintln!("[KERN] init syscall MSRs (LSTAR/STAR/SFMASK + EFER.SCE)");
    sysc::init();
    kprintln!("[KERN] init idt");
    idt::init();
    kprintln!("[KERN] init irq controller (apic abstraction, pic backend)");
    apic::init();
    kprintln!("[KERN] init smp scaffolding (single-core, BSP only)");
    smp::init();

    // P0.7 — hardware reality bring-up. Runs after the PIC has been remapped
    // and masked, before the scheduler comes online. Each step degrades to
    // Skipped if its feature is unavailable so the BSP keeps running.
    let rsdp_opt = if boot.rsdp_phys != 0 { Some(boot.rsdp_phys) } else { None };
    if rsdp_opt.is_none() {
        kprintln!("[HW] ACPI DISABLED - falling back BSP-only");
    } else {
        kprintln!("[HW] ACPI DETECTED rsdp=0x{:016X}", boot.rsdp_phys);
    }
    let ap_stacks: [u64; 0] = [];
    let report = unsafe { hw::init(rsdp_opt, &ap_stacks) };
    kprintln!(
        "[KERN] hw::init cpuid={} acpi={} x2apic={} xapic={} hpet={} fralloc={} pml4={} aps={}",
        report.cpuid, report.acpi, report.x2apic, report.xapic_mmio,
        report.hpet, report.frame_alloc, report.kernel_pml4, report.aps_started,
    );

    // P0.7 — Phase 5 validation hook. We verify but do not panic: the
    // current trampoline is the real-mode HLT stub, so APs entering Rust
    // is intentionally not yet possible (Phase 3 trampoline replacement
    // is gated on QEMU host availability for safe development).
    let online = crate::kernel::smp::ap_boot::online_count();
    if online >= 2 {
        kprintln!("[P0.7] MULTI-CORE ACTIVE: {} cores online", online);
    } else {
        kprintln!("[P0.7] SINGLE CORE MODE ONLY (online={}, sipi_sent={})",
                  online, report.aps_started);
    }

    kprintln!("[KERN] init pit @ {} Hz (apic timer_set)", pit::HZ);
    apic::timer_set(pit::HZ);

    heap::init();
    kprintln!("[KERN] kheap online: {} KiB free", heap::free_bytes() / 1024);

    let probe_va = x86_64::VirtAddr::new(boot as *const _ as u64);
    kprintln!("[KERN] paging: kernel page USER bit = {}", paging::is_user(probe_va));

    sched::init();
    // Arbiter inline: every task must carry an accepted verdict.
    let hb_intent = {
        let mut e = super::intent::make_console_intent("task.heartbeat");
        e.authority = 4; // System spawns kernel tasks
        e
    };
    let hb_verdict = super::intent::submit(&hb_intent);
    match sched::spawn_with_verdict("heartbeat", 1, 1, heartbeat::entry, &hb_verdict) {
        Some(i) => kprintln!("[KERN] spawned heartbeat task in slot {} (verdict reason={})",
                             i, hb_verdict.reason),
        None    => kprintln!("[KERN] WARN: heartbeat spawn denied (verdict reason={})",
                             hb_verdict.reason),
    }
    // Release the spawn-time intent so subsequent Console intents are not
    // permanently dominated by this transient System verdict.
    super::intent::release_one();

    apic::unmask_timer();
    unsafe { core::arch::asm!("sti", options(nomem, nostack, preserves_flags)); }
    kprintln!("[KERN] interrupts enabled");

    audit::append(FrameKind::BootReady, b"P0.2 subsystems online");
    kprintln!("[KERN] audit ring online ({} frames)", audit::count());
    kprintln!("[KERN] entering qsh");

    qsh::run();
}

fn print_banner(boot: &BootInfo) {
    kprintln!();
    kprintln!("============================================================");
    kprintln!(" QRATUM CORE OS  --  P0.2 KERNEL HARDENING");
    kprintln!(" preemptive ready, mpsc audit, syscall/sysret, arbiter inline");
    kprintln!("============================================================");
    kprintln!(" magic       : 0x{:016X}", boot.magic);
    kprintln!(" usable RAM  : {} MiB ({} regions, largest {} MiB)",
              boot.usable_ram_bytes / (1024 * 1024),
              boot.usable_regions,
              boot.largest_region_bytes / (1024 * 1024));
    kprintln!(" rng seed[0..4] : {:02X} {:02X} {:02X} {:02X}",
              boot.rng_seed[0], boot.rng_seed[1], boot.rng_seed[2], boot.rng_seed[3]);
    let cl_end = boot.command_line.iter().position(|&b| b == 0).unwrap_or(boot.command_line.len());
    let cl = core::str::from_utf8(&boot.command_line[..cl_end]).unwrap_or("<bad>");
    kprintln!(" cmdline     : {}", cl);
    kprintln!("============================================================");
    kprintln!();
}
