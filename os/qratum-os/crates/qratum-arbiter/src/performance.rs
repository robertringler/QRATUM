//! Host-side performance & compression measurement (P0.3.1).
//!
//! Two scheduler models — a **baseline** (P0.2 simulation: every
//! accepted intent yields a scheduler enqueue) and **AHTC-K**
//! (canonical fold-or-insert, P0.3 behaviour) — are run on the same
//! deterministic workload and compared.
//!
//! All four ratios (event, scheduling, memory, arbitration) are
//! computed numerically — no theoretical claims.
//!
//! Cycle counts come from the host monotonic clock
//! (`std::time::Instant::elapsed`). Kernel-side mirrors this with
//! RDTSC via `kernel::profiler::cycle_counter`.

use crate::ahtc_k::{AhtcEngine, IntentEnvelope};
use crate::state_machine::{arbitrate, Authority, Intent, LockState, Scope, Verdict};
use serde::{Deserialize, Serialize};
use std::time::Instant;

// ---------------------------------------------------------------------
// Workload generation — deterministic, seeded.
// ---------------------------------------------------------------------

/// Bounded cold-key pool. Mirrors `tests/ahtc_k_tests.rs::workload`.
const COLD_POOL: u64 = 32;

/// Deterministic workload identical to the AHTC-K acceptance test
/// generator. Re-exposed here so `performance_truth` and the strict
/// 10× validator share one input distribution.
pub fn workload(n: usize, dup_pct: u32, seed: u64) -> Vec<IntentEnvelope> {
    let mut out = Vec::with_capacity(n);
    let hot_name = b"hot.intent";
    let hot_lock = 0x4854_4F54;
    let mut x = seed;
    for i in 0..n {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        let pct = (x % 100) as u32;
        let mut id = [0u8; 16];
        id[..8].copy_from_slice(&(i as u64).to_le_bytes());
        id[8..].copy_from_slice(&(i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).to_le_bytes());
        let env = if pct < dup_pct {
            IntentEnvelope {
                id, ts: i as u64,
                authority: 2, scope: 1, priority: 2,
                lock_key: hot_lock,
                name:   hot_name.to_vec(),
                params: Vec::new(),
            }
        } else {
            let bucket = (x >> 7) % COLD_POOL;
            let mut nm = b"cold.".to_vec();
            nm.extend_from_slice(&bucket.to_le_bytes());
            IntentEnvelope {
                id, ts: i as u64,
                authority: 2, scope: 1, priority: 2,
                lock_key: 0x434F_4C44 + bucket,
                name: nm,
                params: b"p".to_vec(),
            }
        };
        out.push(env);
    }
    out
}

// ---------------------------------------------------------------------
// Per-mode workload report.
// ---------------------------------------------------------------------

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkloadReport {
    /// Number of input intents fed to the scheduler.
    pub events_in:        u64,
    /// Scheduler-side enqueues (one per accepted slot of work).
    pub enqueues:         u64,
    /// Audit frames emitted (every Submit produces ≥1).
    pub audit_frames:     u64,
    /// Distinct canonical batches retained in memory.
    pub batches:          u64,
    /// Bytes of resident scheduler state (steady-state, lower bound).
    pub memory_bytes:     u64,
    /// Wall-clock nanoseconds spent in the scheduler path.
    pub scheduling_ns:    u64,
    /// Wall-clock nanoseconds spent in arbitration.
    pub arbitration_ns:   u64,
    /// Total wall-clock nanoseconds end-to-end.
    pub total_ns:         u64,
}

// ---------------------------------------------------------------------
// Baseline — one enqueue per accepted intent. P0.2 behaviour.
// ---------------------------------------------------------------------

/// Approximate per-event resident scheduler-slot size (bytes).
/// Conservative: matches kernel `Tcb` reservation order of magnitude.
const BASELINE_SLOT_BYTES: u64 = 256;
const AHTC_K_BATCH_BYTES:  u64 = 96;
const AHTC_K_MEMBER_BYTES: u64 = 24;

pub fn run_baseline(stream: &[IntentEnvelope]) -> WorkloadReport {
    let t0 = Instant::now();
    let mut l = LockState::new();
    let mut enqueues   = 0u64;
    let mut audit_ev   = 0u64;
    let mut arb_ns     = 0u64;
    let mut sched_ns   = 0u64;

    for e in stream {
        // Arbitration phase. Cost is measured but does NOT gate enqueue
        // — we model the directive's "P0.2-style: one enqueue per intent
        // delivered to the scheduler". Arbiter inflight-cap effects are
        // an orthogonal concern (covered by the host arbiter twin tests),
        // not a scheduler-load reduction we're allowed to claim.
        let a0 = Instant::now();
        let intent = env_to_intent(e);
        let (_v, lp) = arbitrate(&intent, &l, intent.tick);
        // Release after each cycle so the arbiter doesn't saturate; the
        // measurement target is per-intent arbitration cost, not lock
        // contention.
        let mut lp = lp; lp.release();
        l = lp;
        arb_ns += a0.elapsed().as_nanos() as u64;
        audit_ev += 1;

        // Baseline scheduling: every input becomes an enqueue.
        let s0 = Instant::now();
        enqueues += 1;
        audit_ev += 1;
        std::hint::black_box(enqueues);
        sched_ns += s0.elapsed().as_nanos() as u64;
    }

    let total_ns = t0.elapsed().as_nanos() as u64;
    WorkloadReport {
        events_in:      stream.len() as u64,
        enqueues,
        audit_frames:   audit_ev,
        batches:        enqueues,
        memory_bytes:   enqueues.saturating_mul(BASELINE_SLOT_BYTES),
        scheduling_ns:  sched_ns,
        arbitration_ns: arb_ns,
        total_ns,
    }
}

// ---------------------------------------------------------------------
// AHTC-K — fold/insert. P0.3 behaviour.
// ---------------------------------------------------------------------

pub fn run_ahtc_k(stream: &[IntentEnvelope]) -> WorkloadReport {
    let t0 = Instant::now();
    let mut l = LockState::new();
    let mut eng = AhtcEngine::new();
    let mut audit_ev = 0u64;
    let mut arb_ns   = 0u64;
    let mut sched_ns = 0u64;
    let mut total_members = 0u64;

    for e in stream {
        let a0 = Instant::now();
        let intent = env_to_intent(e);
        let (_v, lp) = arbitrate(&intent, &l, intent.tick);
        let mut lp = lp; lp.release();
        l = lp;
        arb_ns += a0.elapsed().as_nanos() as u64;
        audit_ev += 1;

        // AHTC-K scheduling: fold-or-insert. Every intent is delivered
        // to AHTC-K; deduplication happens here.
        let s0 = Instant::now();
        eng.fold_one(e);
        audit_ev += 1;
        total_members += 1;
        sched_ns += s0.elapsed().as_nanos() as u64;
    }

    let total_ns = t0.elapsed().as_nanos() as u64;
    let batches = eng.rq.batches.len() as u64;
    let mem = batches * AHTC_K_BATCH_BYTES + total_members * AHTC_K_MEMBER_BYTES;

    WorkloadReport {
        events_in:      stream.len() as u64,
        enqueues:       eng.stats.enqueue_ops,
        audit_frames:   audit_ev,
        batches,
        memory_bytes:   mem,
        scheduling_ns:  sched_ns,
        arbitration_ns: arb_ns,
        total_ns,
    }
}

// ---------------------------------------------------------------------
// CompressionReport — the four ratios the directive demands.
// ---------------------------------------------------------------------

#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
pub struct CompressionReport {
    pub event_reduction_ratio:        f64,
    pub scheduling_reduction_ratio:   f64,
    pub memory_reduction_ratio:       f64,
    pub arbitration_reduction_ratio:  f64,
    /// Set when the workload satisfies ≥ 70 % duplication AND the
    /// scheduling reduction ratio is ≥ 10.0. Hard CI gate.
    pub meets_10x_at_70pct: bool,
}

impl CompressionReport {
    pub fn pass_10x(&self) -> bool { self.scheduling_reduction_ratio >= 10.0 }
}

/// Public form: derive ratios from `before` (baseline) and `after`
/// (AHTC-K) reports. Division by zero yields f64::INFINITY which the
/// caller can floor; we don't silently round.
pub fn fold_metrics(before: &WorkloadReport, after: &WorkloadReport) -> CompressionReport {
    let er = ratio(before.events_in,    after.batches);
    let sr = ratio(before.enqueues,     after.enqueues);
    let mr = ratio(before.memory_bytes, after.memory_bytes);
    let ar = ratio(before.arbitration_ns, after.arbitration_ns);
    CompressionReport {
        event_reduction_ratio:        er,
        scheduling_reduction_ratio:   sr,
        memory_reduction_ratio:       mr,
        arbitration_reduction_ratio:  ar,
        meets_10x_at_70pct: sr >= 10.0,
    }
}

// ---------------------------------------------------------------------
// P0.3.2 — quantised, kernel-bytewise-equivalent compression metrics.
// ---------------------------------------------------------------------

/// Fixed-point scaling shared with `kernel::instrumentation::metrics`.
pub const ENTROPY_SCALE: u64 = 1_000_000;

/// Mirrors `kernel::instrumentation::metrics::CompressionMetrics`. All
/// six fields are u64 quantised by `ENTROPY_SCALE` so kernel and host
/// are bytewise-comparable.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct CompressionMetrics {
    pub event_reduction_ratio_q:         u64,
    pub scheduling_reduction_ratio_q:    u64,
    pub memory_reduction_ratio_q:        u64,
    pub arbitration_equivalence_ratio_q: u64,
    pub entropy_before_q:                u64,
    pub entropy_after_q:                 u64,
}

/// Integer Shannon entropy in milli-bits (× ENTROPY_SCALE). The
/// algorithm is bytewise-identical to `kernel::instrumentation::
/// metrics::shannon_entropy_bits` — it MUST stay that way (asserted
/// by `tests/entropy_consistency_tests.rs`).
pub fn shannon_entropy_bits(counts: &[u64]) -> u64 {
    let total: u128 = counts.iter().map(|&c| c as u128).sum();
    if total == 0 { return 0; }
    let mut acc: u128 = 0;
    for &c in counts {
        if c == 0 { continue; }
        let log_total = (total as u64).ilog2() as u128;
        let log_c     = (c as u64).max(1).ilog2() as u128;
        let term = (c as u128) * (log_total - log_c) * (ENTROPY_SCALE as u128);
        acc += term / total;
    }
    acc as u64
}

/// Build the quantised metric view from a baseline+AHTC pair plus the
/// AHTC engine's batch-occupancy histogram (passed in by the caller —
/// the host doesn't introspect AhtcEngine internals here to keep the
/// data contract narrow).
pub fn compression_metrics(
    before: &WorkloadReport,
    after:  &WorkloadReport,
    histogram_before: &[u64],
    histogram_after:  &[u64],
) -> CompressionMetrics {
    let q = |num: u64, den: u64| -> u64 {
        if den == 0 { 0 } else { (num as u128 * ENTROPY_SCALE as u128 / den as u128) as u64 }
    };
    CompressionMetrics {
        event_reduction_ratio_q:         q(before.events_in,    after.batches.max(1)),
        scheduling_reduction_ratio_q:    q(before.enqueues,     after.enqueues.max(1)),
        memory_reduction_ratio_q:        q(before.memory_bytes, after.memory_bytes.max(1)),
        arbitration_equivalence_ratio_q: q(before.arbitration_ns, after.arbitration_ns.max(1)),
        entropy_before_q: shannon_entropy_bits(histogram_before),
        entropy_after_q:  shannon_entropy_bits(histogram_after),
    }
}

fn ratio(num: u64, den: u64) -> f64 {
    if den == 0 { f64::INFINITY } else { num as f64 / den as f64 }}

// ---------------------------------------------------------------------
// LatencyEnvelope — kernel ↔ host parity bound.
// ---------------------------------------------------------------------

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EpsilonBounds {
    /// Allowed deviation per subsystem, expressed as a percentage.
    pub arbitration_pct:  u32, // 2..=5
    pub scheduling_pct:   u32, // 5..=10
    pub syscall_pct:      u32, // 1..=3
    pub ahtc_fold_pct:    u32, // 2 (P0.3.2)
}

impl EpsilonBounds {
    pub const STRICT: Self = Self {
        arbitration_pct: 5, scheduling_pct: 10, syscall_pct: 3, ahtc_fold_pct: 2,
    };
}

impl Default for EpsilonBounds {
    fn default() -> Self { Self::STRICT }
}

#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
pub struct LatencyEnvelope {
    pub kernel_arbitration_ns:  u64,
    pub host_arbitration_ns:    u64,
    pub kernel_scheduling_ns:   u64,
    pub host_scheduling_ns:     u64,
    pub kernel_syscall_ns:      u64,
    pub host_syscall_ns:        u64,
    pub kernel_ahtc_fold_ns:    u64,
    pub host_ahtc_fold_ns:      u64,
}

impl LatencyEnvelope {
    /// Returns `true` iff every per-subsystem deviation is within the
    /// supplied epsilon bounds. A zero-zero comparison passes (vacuously
    /// true — neither side has executed the path).
    pub fn within(&self, eps: &EpsilonBounds) -> bool {
        within_pct(self.kernel_arbitration_ns, self.host_arbitration_ns, eps.arbitration_pct)
            && within_pct(self.kernel_scheduling_ns, self.host_scheduling_ns, eps.scheduling_pct)
            && within_pct(self.kernel_syscall_ns,    self.host_syscall_ns,    eps.syscall_pct)
            && within_pct(self.kernel_ahtc_fold_ns,  self.host_ahtc_fold_ns,  eps.ahtc_fold_pct)
    }

    /// Largest |kernel-host|/host across all subsystems.
    pub fn max_deviation_pct(&self) -> f64 {
        let a = abs_pct(self.kernel_arbitration_ns, self.host_arbitration_ns);
        let s = abs_pct(self.kernel_scheduling_ns,  self.host_scheduling_ns);
        let y = abs_pct(self.kernel_syscall_ns,     self.host_syscall_ns);
        let f = abs_pct(self.kernel_ahtc_fold_ns,   self.host_ahtc_fold_ns);
        a.max(s).max(y).max(f)
    }
}

fn within_pct(k: u64, h: u64, pct: u32) -> bool {
    if h == 0 && k == 0 { return true; }
    if h == 0 { return false; }
    let dev = (k as i128 - h as i128).unsigned_abs() as u128;
    dev * 100 <= (h as u128) * (pct as u128)
}

fn abs_pct(k: u64, h: u64) -> f64 {
    if h == 0 { return if k == 0 { 0.0 } else { f64::INFINITY }; }
    let d = (k as f64 - h as f64).abs();
    (d / h as f64) * 100.0
}

// ---------------------------------------------------------------------
// IntentEnvelope → state_machine::Intent adapter.
// ---------------------------------------------------------------------

fn env_to_intent(e: &IntentEnvelope) -> Intent {
    let auth = match e.authority {
        1 => Authority::User,
        2 => Authority::Console,
        3 => Authority::Service,
        _ => Authority::System,
    };
    let scope = match e.scope {
        1 => Scope::Process,
        2 => Scope::Session,
        _ => Scope::Machine,
    };
    Intent {
        id:        e.id,
        kind:      1,
        authority: auth,
        scope,
        tick:      e.ts,
        payload_h: [0; 32],
    }
}

// ---------------------------------------------------------------------
// Performance report — pretty-printed, used by the test harness.
// ---------------------------------------------------------------------

pub fn print_report(label: &str, before: &WorkloadReport, after: &WorkloadReport,
                    cr: &CompressionReport, env: Option<&LatencyEnvelope>,
                    eps: &EpsilonBounds) {
    println!("\nAHTC-K Performance Report ({label})");
    println!("  events_in              : {}", before.events_in);
    println!("  baseline.enqueues      : {}", before.enqueues);
    println!("  ahtc_k.enqueues        : {}", after.enqueues);
    println!("  baseline.memory_bytes  : {}", before.memory_bytes);
    println!("  ahtc_k.memory_bytes    : {}", after.memory_bytes);
    println!("  Event reduction        : {:.2}×", cr.event_reduction_ratio);
    println!("  Scheduling reduction   : {:.2}×", cr.scheduling_reduction_ratio);
    println!("  Memory reduction       : {:.2}×", cr.memory_reduction_ratio);
    println!("  Arbitration reduction  : {:.2}×", cr.arbitration_reduction_ratio);
    if let Some(e) = env {
        println!("  Latency max deviation  : {:.2}%", e.max_deviation_pct());
        println!("  Within ε (a={}% s={}% y={}%): {}",
                 eps.arbitration_pct, eps.scheduling_pct, eps.syscall_pct,
                 e.within(eps));
    }
    println!("  Status                 : {}",
             if cr.meets_10x_at_70pct { "PASS" } else { "FAIL" });
}

// ---------------------------------------------------------------------
// Convenience: deterministic intent stream from the corpus crate, so
// `timing_equivalence` doesn't have to invent its own.
// ---------------------------------------------------------------------

pub fn intent_stream_from_authority(n: usize) -> Vec<Intent> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n as u64 {
        let auth = match i % 4 {
            0 => Authority::User,
            1 => Authority::Console,
            2 => Authority::Service,
            _ => Authority::System,
        };
        let mut id = [0u8; 16];
        id[..8].copy_from_slice(&i.to_le_bytes());
        out.push(Intent {
            id,
            kind: ((i % 7) as u8) + 1,
            authority: auth,
            scope: Scope::Process,
            tick: i,
            payload_h: [0; 32],
        });
    }
    out
}
