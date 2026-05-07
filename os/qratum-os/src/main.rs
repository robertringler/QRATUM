//! QRATUM CORE OS — P0.0 BOOT STRIKE
//!
//! Single-binary UEFI image. Contains both the Stage-1 bootloader logic
//! (acquire memory map, ExitBootServices) and the Stage-2 kernel (serial
//! console, audit ring, intent stub, qsh shell).
//!
//! In P0.0.1 the kernel will move to a separate ELF that this binary
//! loads from \EFI\QRATUM\kernel.elf — the public API of `kernel::kmain`
//! is shaped to make that split mechanical.

#![no_std]
#![no_main]
#![allow(internal_features)]
#![feature(abi_x86_interrupt)]
#![feature(naked_functions)]

extern crate alloc;

mod boot;
mod kernel;
mod shared;

use core::fmt::Write;
use uefi::prelude::*;

/// UEFI entry point. The `uefi::entry` attribute installs the global
/// allocator and panic handler exposed by the `uefi` crate features.
#[entry]
fn efi_main(_image: Handle, mut st: SystemTable<Boot>) -> Status {
    let _ = st.stdout().clear();
    let _ = writeln!(
        st.stdout(),
        "QRATUM CORE OS  v0.0.1  (P0.0 BOOT STRIKE)\r\n\
         ============================================\r\n"
    );

    // ---- Stage 1: bootloader ------------------------------------------------
    let boot_info = match boot::stage1(&mut st) {
        Ok(bi) => bi,
        Err(e) => {
            let _ = writeln!(st.stdout(), "[BOOT] stage1 failed: {:?}", e);
            return e;
        }
    };

    let _ = writeln!(
        st.stdout(),
        "[BOOT] usable RAM = {} MiB across {} regions\r\n\
         [BOOT] handing off to kernel...\r\n",
        boot_info.usable_ram_bytes / (1024 * 1024),
        boot_info.usable_regions
    );

    // ---- ExitBootServices ---------------------------------------------------
    // After this call, `st`'s boot services are gone. We can no longer use
    // the UEFI text console; from here on only direct hardware (serial UART,
    // raw memory) is available.
    let (_runtime, _mmap) = unsafe {
        st.exit_boot_services(uefi::table::boot::MemoryType::LOADER_DATA)
    };

    // ---- Stage 2: kernel ----------------------------------------------------
    kernel::kmain::kernel_main(&boot_info);
}
