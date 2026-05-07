//! P0.4 — Affinity table.
//!
//! Maps `task_id -> home_cpu`. Read-mostly. Writes happen only at
//! task creation or successful migration commit and are serialized
//! by `barriers::with_migration_barrier`.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU8, Ordering};

const MAX_TASKS: usize = 1024;

/// One byte per task; 0xFF means "unassigned".
static AFFINITY: [AtomicU8; MAX_TASKS] = {
    const Z: AtomicU8 = AtomicU8::new(0xFF);
    [Z; MAX_TASKS]
};

#[inline]
pub fn home_cpu(task_id: u32) -> Option<u8> {
    let v = AFFINITY[(task_id as usize) % MAX_TASKS].load(Ordering::Acquire);
    if v == 0xFF { None } else { Some(v) }
}

/// Set affinity. Caller MUST hold the migration barrier or be in
/// task-create context — this is checked by `barriers::is_locked`
/// in debug builds.
pub fn set_home_cpu(task_id: u32, cpu: u8) {
    debug_assert!(super::barriers::is_locked(),
        "affinity write outside migration barrier — INV-K risk");
    AFFINITY[(task_id as usize) % MAX_TASKS].store(cpu, Ordering::Release);
}

/// Deterministic placement for newly-created tasks. Hash on task_id
/// so identical task ids map to identical cores across runs (replay
/// invariance).
#[inline]
pub fn deterministic_placement(task_id: u32, online_cpus: u8) -> u8 {
    if online_cpus == 0 { return 0; }
    // Stable hash — DO NOT replace with std::hash::Hasher (non-deterministic).
    let mut x = task_id as u64 ^ 0x9E37_79B9_7F4A_7C15;
    x ^= x >> 30; x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27; x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x % online_cpus as u64) as u8
}
