//! P0.7 — AP trampoline blob.
//!
//! The blob is a real-mode (16-bit) entry stub that an AP enters
//! after STARTUP IPI lands. The minimal viable trampoline:
//!
//!   1. CLI
//!   2. set up a temporary stack
//!   3. spin in HLT — keeps the AP halted but accounted for
//!
//! A full real-mode → long-mode switch (load GDT, enable PE, jump
//! to 32-bit, enable PAE+LME, load CR3, jump to 64-bit) is a
//! follow-up slice; the current stub is enough to validate that
//! INIT/SIPI lands on the real silicon (AP exits reset, executes
//! our blob, and stops cleanly instead of triple-faulting).

#![allow(dead_code)]

/// 16-bit real-mode trampoline:
///   00: fa            cli
///   01: 31 c0         xor ax, ax
///   03: 8e d8         mov ds, ax
///   05: 8e c0         mov es, ax
///   07: 8e d0         mov ss, ax
///   09: bc 00 7c      mov sp, 0x7c00
///   0c: f4            hlt
///   0d: eb fd         jmp -3
pub const AP_TRAMPOLINE_BLOB: &[u8] = &[
    0xFA,
    0x31, 0xC0,
    0x8E, 0xD8,
    0x8E, 0xC0,
    0x8E, 0xD0,
    0xBC, 0x00, 0x7C,
    0xF4,
    0xEB, 0xFD,
];

/// Copy the trampoline blob to the staged frame.
///
/// # Safety
/// `frame_pa` must be a 4 KiB-aligned, identity-mapped frame
/// below 1 MiB.
pub unsafe fn install(frame_pa: u64) {
    let dst = frame_pa as *mut u8;
    for (i, b) in AP_TRAMPOLINE_BLOB.iter().enumerate() {
        core::ptr::write_volatile(dst.add(i), *b);
    }
}
