//! P0.7 — RDMSR / WRMSR + control-register helpers.
//!
//! The `x86_64` crate exposes typed MSR wrappers, but several MSRs we
//! need (IA32_APIC_BASE, x2APIC ICR, TSC-DEADLINE) are best driven
//! through raw u32 numbers to keep the surface tight.

#![allow(dead_code)]

use core::arch::asm;

pub const IA32_APIC_BASE:        u32 = 0x1B;
pub const IA32_X2APIC_APICID:    u32 = 0x802;
pub const IA32_X2APIC_VERSION:   u32 = 0x803;
pub const IA32_X2APIC_TPR:       u32 = 0x808;
pub const IA32_X2APIC_EOI:       u32 = 0x80B;
pub const IA32_X2APIC_LDR:       u32 = 0x80D;
pub const IA32_X2APIC_SIVR:      u32 = 0x80F;
pub const IA32_X2APIC_ESR:       u32 = 0x828;
pub const IA32_X2APIC_ICR:       u32 = 0x830;
pub const IA32_X2APIC_LVT_TIMER: u32 = 0x832;
pub const IA32_X2APIC_INIT_COUNT:u32 = 0x838;
pub const IA32_X2APIC_CUR_COUNT: u32 = 0x839;
pub const IA32_X2APIC_DCR:       u32 = 0x83E;
pub const IA32_TSC_DEADLINE:     u32 = 0x6E0;

/// IA32_APIC_BASE bits.
pub const APIC_BASE_BSP:        u64 = 1 << 8;
pub const APIC_BASE_X2APIC_EN:  u64 = 1 << 10;
pub const APIC_BASE_GLOBAL_EN:  u64 = 1 << 11;

/// # Safety
/// Caller asserts the MSR is implemented on the running CPU.
#[inline]
pub unsafe fn rdmsr(msr: u32) -> u64 {
    let lo: u32; let hi: u32;
    asm!("rdmsr", in("ecx") msr, out("eax") lo, out("edx") hi, options(nomem, nostack, preserves_flags));
    ((hi as u64) << 32) | (lo as u64)
}

/// # Safety
/// Caller asserts the MSR is writable on the running CPU and the
/// value is well-formed for the target MSR.
#[inline]
pub unsafe fn wrmsr(msr: u32, val: u64) {
    let lo = val as u32;
    let hi = (val >> 32) as u32;
    asm!("wrmsr", in("ecx") msr, in("eax") lo, in("edx") hi, options(nomem, nostack, preserves_flags));
}
