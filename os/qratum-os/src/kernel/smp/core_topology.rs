//! P0.6 — Logical core topology descriptor.

#![allow(dead_code)]

use super::MAX_CPUS;

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct CoreInfo {
    pub cpu_id:     u8,
    pub apic_id:    u8,
    pub package:    u8,
    pub numa_node:  u8,
    pub smt_thread: u8,
}

static mut CORES: [CoreInfo; MAX_CPUS] = [CoreInfo {
    cpu_id: 0, apic_id: 0, package: 0, numa_node: 0, smt_thread: 0,
}; MAX_CPUS];

pub fn set(info: CoreInfo) {
    if (info.cpu_id as usize) < MAX_CPUS {
        unsafe { CORES[info.cpu_id as usize] = info; }
    }
}

pub fn get(cpu_id: u8) -> Option<CoreInfo> {
    if (cpu_id as usize) >= MAX_CPUS { None }
    else { unsafe { Some(CORES[cpu_id as usize]) } }
}

pub fn topology_hash() -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for i in 0..MAX_CPUS {
        let c = unsafe { CORES[i] };
        h = (h ^ c.cpu_id as u64).wrapping_mul(0x100_0000_01B3);
        h = (h ^ c.apic_id as u64).wrapping_mul(0x100_0000_01B3);
        h = (h ^ c.package as u64).wrapping_mul(0x100_0000_01B3);
        h = (h ^ c.numa_node as u64).wrapping_mul(0x100_0000_01B3);
        h = (h ^ c.smt_thread as u64).wrapping_mul(0x100_0000_01B3);
    }
    h
}
