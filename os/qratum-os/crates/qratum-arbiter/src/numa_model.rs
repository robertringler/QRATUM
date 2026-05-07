//! P0.5 — Host NUMA + cache-locality + migration-cost model.
//!
//! Pure deterministic mirror of `kernel::profiler::numa_metrics` and
//! `cache_locality`. Used by host tests to verify scheduler reduction
//! ratios and locality preservation across multicore replay.

use serde::{Deserialize, Serialize};

pub const MAX_NODES: usize = 4;
pub const SCALE:     u64   = 1_000_000;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NumaTopology {
    pub nodes:    u8,
    pub distance: Vec<Vec<u8>>,
    /// Map cpu_id → numa node.
    pub cpu_node: Vec<u8>,
}

impl NumaTopology {
    pub fn flat(cpus: u8) -> Self {
        Self {
            nodes: 1,
            distance: vec![vec![0]],
            cpu_node: vec![0; cpus as usize],
        }
    }

    pub fn dual(cpus: u8) -> Self {
        let half = cpus / 2;
        let mut cpu_node = vec![0u8; cpus as usize];
        for i in 0..cpus { cpu_node[i as usize] = if i < half { 0 } else { 1 }; }
        Self {
            nodes: 2,
            distance: vec![vec![0, 20], vec![20, 0]],
            cpu_node,
        }
    }

    #[inline]
    pub fn distance(&self, a: u8, b: u8) -> u8 {
        self.distance[a as usize][b as usize]
    }

    #[inline]
    pub fn node_of(&self, cpu: u8) -> u8 { self.cpu_node[cpu as usize] }
}

#[derive(Copy, Clone, Default, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LocalitySummary {
    pub local:      u64,
    pub remote:     u64,
    pub locality_q: u64, // [0, SCALE]
}

pub fn summarize(local: u64, remote: u64) -> LocalitySummary {
    let q = if local + remote == 0 { 0 } else { local * SCALE / (local + remote) };
    LocalitySummary { local, remote, locality_q: q }
}

/// Migration cache penalty model. Pure function — identical inputs
/// produce identical penalty cycles.
pub fn migration_penalty_cycles(distance: u8, working_set_kib: u64) -> u64 {
    // Simple deterministic affine model: base + per-KB × distance.
    100u64.saturating_add(working_set_kib.saturating_mul(distance as u64))
}

/// Aggregate scheduler-event reduction measurement. Caller supplies
/// (events_baseline, events_canonical). Returns reduction ratio in
/// fixed-point [0, SCALE]. Reduction = (baseline - canonical) / baseline.
pub fn reduction_q(baseline: u64, canonical: u64) -> u64 {
    if baseline == 0 || canonical >= baseline { return 0; }
    (baseline - canonical) * SCALE / baseline
}
