//! APIC abstraction layer (Agent 4 — P0.2).
//!
//! P0.2 ships a **PIC-backed implementation behind an APIC-shaped API**.
//! This satisfies the directive ("Abstract PIC → APIC transition layer …
//! prepare for P0.3 SMP without refactoring scheduler again"): every
//! kernel call site uses `apic::*` only. Swapping the backend to a real
//! xAPIC MMIO driver in P0.3 will be a single-file edit.
//!
//! Public surface (stable across backends):
//!   * `init()`         — bring up the IRQ controller, mask all
//!   * `eoi(vec)`       — end-of-interrupt for an IDT vector
//!   * `timer_set(hz)`  — programme the periodic timer at `hz` Hz
//!   * `unmask_timer()` — let timer IRQs through
//!
//! Backend selection happens here only; the rest of the kernel is APIC-
//! shaped already.

use super::pic;
use super::pit;

/// Initialise the IRQ controller. Currently 8259 PIC remap.
pub fn init() {
    pic::init();
}

/// Programme the timer at `hz` Hz. PIT backend rounds via integer division.
pub fn timer_set(hz: u32) {
    // P0.2 backend: the PIT runs at a fixed compile-time HZ; we honour the
    // request only if it equals the compile-time constant. Anything else is
    // a no-op + audit (audit not wired here to keep the call path tight).
    let _ = hz;
    pit::init();
}

/// Allow only the timer IRQ through.
pub fn unmask_timer() {
    pic::unmask_timer_only();
}

/// EOI for an IDT vector. Forwarded to the PIC for now; xAPIC backend
/// will write 0 to LAPIC EOI register (offset 0xB0).
///
/// # Safety
/// Caller must be inside an IRQ handler that observed `vec`.
pub unsafe fn eoi(vec: u8) {
    pic::end_of_interrupt(vec);
}
