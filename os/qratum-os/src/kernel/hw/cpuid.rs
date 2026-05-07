//! P0.7 — CPUID feature detection.

#![allow(dead_code)]

use core::arch::x86_64::__cpuid;

#[derive(Copy, Clone, Debug, Default)]
pub struct Features {
    pub max_basic:    u32,
    pub vendor:       [u8; 12],
    pub apic:         bool,
    pub x2apic:       bool,
    pub tsc:          bool,
    pub tsc_deadline: bool,
    pub hpet:         bool, // best-effort; real signal is ACPI HPET table
    pub fxsr:         bool,
    pub sse2:         bool,
}

pub fn detect() -> Features {
    let mut f = Features::default();
    unsafe {
        let r0 = __cpuid(0);
        f.max_basic = r0.eax;
        let mut v = [0u8; 12];
        v[0..4].copy_from_slice(&r0.ebx.to_le_bytes());
        v[4..8].copy_from_slice(&r0.edx.to_le_bytes());
        v[8..12].copy_from_slice(&r0.ecx.to_le_bytes());
        f.vendor = v;

        if r0.eax >= 1 {
            let r1 = __cpuid(1);
            f.apic         = (r1.edx & (1 << 9))  != 0;
            f.tsc          = (r1.edx & (1 << 4))  != 0;
            f.fxsr         = (r1.edx & (1 << 24)) != 0;
            f.sse2         = (r1.edx & (1 << 26)) != 0;
            f.x2apic       = (r1.ecx & (1 << 21)) != 0;
            f.tsc_deadline = (r1.ecx & (1 << 24)) != 0;
        }
    }
    f
}
