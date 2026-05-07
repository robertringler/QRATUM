//! P0.4 — Performance Truth Framework.
//!
//! Every public performance number must be expressed as a
//! `PerformanceClaim`. The struct enforces that the claim carries:
//!   - workload class
//!   - duplication ratio
//!   - event count
//!   - hardware profile fingerprint
//!   - cache state (cold/warm)
//!   - arbitration mode
//!   - replay mode
//!   - timing source
//!
//! `validate(text)` rejects any string that uses the banned headline
//! terms (universal compute acceleration / general CPU speedup /
//! OS-wide acceleration). `tests/performance_claim_integrity.rs` runs
//! `validate` over the entire docs tree.

use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum WorkloadClass {
    HighDuplication70Pct,
    LowDuplication0Pct,
    BurstThenIdle,
    InterruptStorm,
    MigrationStorm,
    Adversarial,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CacheState   { Cold, Warm }

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ArbitrationMode { Strict, Permissive }

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ReplayMode   { Live, Replay }

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum TimingSource { Rdtsc, MonotonicNs, Synthetic }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PerformanceClaim {
    pub headline:           String,
    pub workload_class:     WorkloadClass,
    pub duplication_pct:    u32,
    pub event_count:        u64,
    pub hardware_fingerprint: String,
    pub cache_state:        CacheState,
    pub arbitration_mode:   ArbitrationMode,
    pub replay_mode:        ReplayMode,
    pub timing_source:      TimingSource,
    pub measured_value:     f64,
    pub units:              String,
}

pub const BANNED_TERMS: &[&str] = &[
    "universal compute acceleration",
    "general cpu speedup",
    "general-purpose cpu speedup",
    "os-wide acceleration",
    "os-wide speedup",
    "blanket speedup",
];

pub const REQUIRED_VOCABULARY: &[&str] = &[
    "scheduler event reduction",
    "canonical execution reduction",
    "semantic-preserving execution compression",
];

#[derive(Clone, Debug)]
pub struct ValidationResult {
    pub banned_hits:    Vec<(String, String)>, // (path, term)
    pub vocab_present:  bool,
    pub claims_checked: usize,
}

impl ValidationResult {
    pub fn is_pass(&self) -> bool { self.banned_hits.is_empty() }
}

/// Scan a text body for banned terms. Returns the list of hits.
pub fn validate_text(path: &str, body: &str) -> Vec<(String, String)> {
    let lower = body.to_ascii_lowercase();
    let mut hits = Vec::new();
    for &t in BANNED_TERMS {
        if lower.contains(t) {
            hits.push((path.to_string(), t.to_string()));
        }
    }
    hits
}

/// Validate a `PerformanceClaim` is well-formed.
pub fn validate_claim(c: &PerformanceClaim) -> Result<(), String> {
    if c.headline.is_empty() { return Err("empty headline".into()); }
    let lower = c.headline.to_ascii_lowercase();
    for &t in BANNED_TERMS {
        if lower.contains(t) {
            return Err(format!("banned term '{}' in headline", t));
        }
    }
    if c.event_count == 0 { return Err("event_count must be > 0".into()); }
    if c.units.is_empty()  { return Err("units required".into()); }
    if c.hardware_fingerprint.is_empty() {
        return Err("hardware_fingerprint required".into());
    }
    Ok(())
}
