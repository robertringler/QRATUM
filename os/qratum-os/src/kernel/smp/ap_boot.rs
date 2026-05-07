//! P0.5 — AP (Application Processor) boot path.
//!
//! Real INIT/SIPI sequence is hardware-gated and lands behind a
//! `cfg(target_feature = "smp_real")` flag in P0.6. This module
//! provides the deterministic *boot record* — the data structure that
//! the BSP fills in for each AP and the AP would consume on real
//! hardware. The boot record is replay-stable: a kernel run with the
//! same logical-CPU map produces the same record contents.

#![allow(dead_code)]

use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use super::MAX_CPUS;

#[derive(Copy, Clone, Debug, Default)]
pub struct ApBootRecord {
    pub cpu_id:     u8,
    pub apic_id:    u8,
    pub stack_top:  u64,
    pub started:    bool,
}

const EMPTY: ApBootRecord = ApBootRecord {
    cpu_id: 0, apic_id: 0, stack_top: 0, started: false,
};

static mut RECORDS: [ApBootRecord; MAX_CPUS] = [EMPTY; MAX_CPUS];
static AP_ONLINE: [AtomicBool; MAX_CPUS] = [
    AtomicBool::new(false), AtomicBool::new(false),
    AtomicBool::new(false), AtomicBool::new(false),
    AtomicBool::new(false), AtomicBool::new(false),
    AtomicBool::new(false), AtomicBool::new(false),
];
static AP_COUNT: AtomicU8 = AtomicU8::new(1); // BSP

/// Stage an AP record. Returns false if cpu_id is out of range.
pub fn stage(cpu_id: u8, apic_id: u8, stack_top: u64) -> bool {
    if (cpu_id as usize) >= MAX_CPUS { return false; }
    unsafe {
        RECORDS[cpu_id as usize] = ApBootRecord {
            cpu_id, apic_id, stack_top, started: false,
        };
    }
    true
}

/// Mark BSP online. Idempotent.
pub fn mark_bsp_online() {
    AP_ONLINE[0].store(true, Ordering::Release);
}

/// Mark an AP as online. Returns previous online status.
pub fn mark_online(cpu_id: u8) -> bool {
    if (cpu_id as usize) >= MAX_CPUS { return false; }
    let prev = AP_ONLINE[cpu_id as usize].swap(true, Ordering::AcqRel);
    if !prev { AP_COUNT.fetch_add(1, Ordering::AcqRel); }
    prev
}

#[inline]
pub fn online_count() -> u8 { AP_COUNT.load(Ordering::Acquire) }

#[inline]
pub fn is_online(cpu_id: u8) -> bool {
    (cpu_id as usize) < MAX_CPUS && AP_ONLINE[cpu_id as usize].load(Ordering::Acquire)
}

pub fn record(cpu_id: u8) -> Option<ApBootRecord> {
    if (cpu_id as usize) >= MAX_CPUS { return None; }
    unsafe { Some(RECORDS[cpu_id as usize]) }
}

pub fn reset_for_test() {
    for a in &AP_ONLINE { a.store(false, Ordering::Release); }
    AP_COUNT.store(1, Ordering::Release);
    unsafe { RECORDS = [EMPTY; MAX_CPUS]; }
}
