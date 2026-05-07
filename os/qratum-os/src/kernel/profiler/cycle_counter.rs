//! Kernel cycle counter — thin wrapper around `RDTSC`.
//!
//! `RDTSC` is the only timestamp source the kernel uses for profiling
//! hot paths. No serialising fence is emitted (we want the cheapest
//! possible read on `intent::submit`); call sites that need ordering
//! against memory ops should issue their own `mfence`.

#![allow(dead_code)]

#[inline(always)]
pub fn rdtsc() -> u64 {
    // SAFETY: `_rdtsc` reads the time-stamp counter. Available on
    // every x86_64 CPU since the Pentium; UEFI guarantees this for
    // QRATUM's target.
    unsafe { core::arch::x86_64::_rdtsc() }
}

/// Difference of two `rdtsc()` samples, saturating to zero on
/// non-monotonic reads (which can occur across a CPU migration once
/// the SMP layer goes live — see `spec/smp_model.md`). Single-CPU
/// callers always see a strictly increasing TSC.
#[inline]
pub fn delta(start: u64, end: u64) -> u64 {
    end.saturating_sub(start)
}
