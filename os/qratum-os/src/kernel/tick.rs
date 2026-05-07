//! Monotonic kernel tick (Agent 4 stub).
//!
//! P0.0 has no IRQ-driven timer yet. We use `RDTSC` as a free-running
//! reference and a per-LOC counter incremented by code that needs an
//! ordering tick (per spec §6.1: tick advances at every Lock-Order
//! Commit). For P0.0 there are no locks, so `loc_tick` advances on each
//! intent submission.

use core::sync::atomic::{AtomicU64, Ordering};

static LOC_TICK: AtomicU64 = AtomicU64::new(0);

/// Read the CPU timestamp counter — boot-relative, host-rate. Used only
/// for human-facing diagnostics; never for ordering.
#[inline]
pub fn rdtsc() -> u64 {
    let lo: u32; let hi: u32;
    unsafe { core::arch::asm!("rdtsc", out("eax") lo, out("edx") hi, options(nomem, nostack, preserves_flags)); }
    ((hi as u64) << 32) | (lo as u64)
}

/// Advance and return the new monotonic LOC tick.
pub fn loc_advance() -> u64 {
    LOC_TICK.fetch_add(1, Ordering::SeqCst) + 1
}

/// Read current LOC tick without advancing.
pub fn loc_now() -> u64 { LOC_TICK.load(Ordering::SeqCst) }
