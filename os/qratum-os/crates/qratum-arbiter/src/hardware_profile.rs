//! P0.4 — Hardware profile (host).
//!
//! Twin of `kernel::profiler::{cache_metrics, irq_latency,
//! context_switch, tlb_metrics}`. The host computes the same metrics
//! over a host-emulated workload so CI can assert kernel↔host parity
//! once the kernel target lands on real hardware.
//!
//! Time source is `std::time::Instant` (monotonic, ns resolution);
//! kernel uses RDTSC. The two are compared via the existing latency
//! envelope (`performance::EpsilonBounds`).

use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub irq_samples:        u64,
    pub irq_total_ns:       u64,
    pub irq_p50_ns:         u64,
    pub irq_p99_ns:         u64,
    pub ctx_switches:       u64,
    pub ctx_total_ns:       u64,
    pub ctx_min_ns:         u64,
    pub ctx_max_ns:         u64,
    pub cache_reuse_hits:   u64,
    pub cache_cold_inserts: u64,
    pub cache_locality_q:   u64,
    pub tlb_as_switches:    u64,
    pub tlb_pt_walks:       u64,
}

pub const LOCALITY_SCALE: u64 = 1_000_000;

#[derive(Default)]
pub struct HardwareProfiler {
    irq_buckets: Vec<u64>,
    ctx_samples: Vec<u64>,
    cache_reuse: u64,
    cache_cold:  u64,
    tlb_as:      u64,
    tlb_walks:   u64,
}

impl HardwareProfiler {
    pub fn new() -> Self { Self::default() }

    pub fn time_irq<F: FnOnce()>(&mut self, f: F) {
        let t0 = Instant::now(); f();
        self.irq_buckets.push(t0.elapsed().as_nanos() as u64);
    }

    pub fn time_context_switch<F: FnOnce()>(&mut self, f: F) {
        let t0 = Instant::now(); f();
        self.ctx_samples.push(t0.elapsed().as_nanos() as u64);
    }

    pub fn note_reuse(&mut self) { self.cache_reuse += 1; }
    pub fn note_cold(&mut self)  { self.cache_cold  += 1; }
    pub fn note_as_switch(&mut self) { self.tlb_as += 1; }
    pub fn note_pt_walk(&mut self)   { self.tlb_walks += 1; }

    pub fn finalize(&self) -> HardwareProfile {
        let irq_total: u64 = self.irq_buckets.iter().sum();
        let mut sorted = self.irq_buckets.clone(); sorted.sort();
        let p50 = if sorted.is_empty() { 0 } else { sorted[sorted.len()/2] };
        let p99 = if sorted.is_empty() { 0 }
                  else { sorted[((sorted.len() * 99) / 100).min(sorted.len()-1)] };
        let ctx_total: u64 = self.ctx_samples.iter().sum();
        let ctx_min   = self.ctx_samples.iter().copied().min().unwrap_or(0);
        let ctx_max   = self.ctx_samples.iter().copied().max().unwrap_or(0);
        let denom = self.cache_reuse + self.cache_cold;
        let locality_q = if denom == 0 { 0 } else { self.cache_reuse * LOCALITY_SCALE / denom };
        HardwareProfile {
            irq_samples: self.irq_buckets.len() as u64,
            irq_total_ns: irq_total,
            irq_p50_ns: p50,
            irq_p99_ns: p99,
            ctx_switches: self.ctx_samples.len() as u64,
            ctx_total_ns: ctx_total,
            ctx_min_ns: ctx_min,
            ctx_max_ns: ctx_max,
            cache_reuse_hits:   self.cache_reuse,
            cache_cold_inserts: self.cache_cold,
            cache_locality_q:   locality_q,
            tlb_as_switches:    self.tlb_as,
            tlb_pt_walks:       self.tlb_walks,
        }
    }
}

/// Comparison helper: returns `true` if AHTC-K materially improved
/// locality over baseline (locality_q grew by ≥ `min_delta_q`).
pub fn locality_improved(
    baseline: &HardwareProfile,
    ahtc:     &HardwareProfile,
    min_delta_q: u64,
) -> bool {
    ahtc.cache_locality_q.saturating_sub(baseline.cache_locality_q) >= min_delta_q
}
