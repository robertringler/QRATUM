//! P0.5 — 1000-scenario adversarial corpus generator.
//!
//! Ten verbatim categories from the directive, 100 scenarios each.
//! Every scenario is fully described by `(category, seed)`; replay is
//! deterministic.

use serde::{Deserialize, Serialize};

#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum AdversarialCategory {
    MulticoreRaceStorm        = 0,
    MigrationCascade          = 1,
    LockInversionAttempt      = 2,
    CacheThrashWorkload       = 3,
    IrqFloodCondition         = 4,
    AffinityOscillation       = 5,
    PartialOrderAdversarial   = 6,
    CanonicalNearCollision    = 7,
    SchedulerStarvation       = 8,
    ReplayDivergenceInjection = 9,
}

pub const CATEGORIES: usize = 10;
pub const PER_CATEGORY: usize = 100;
pub const TOTAL: usize = CATEGORIES * PER_CATEGORY;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Scenario {
    pub id:       u32,
    pub category: AdversarialCategory,
    pub seed:     u64,
    /// Deterministic op stream — bytes interpreted by replay harness.
    pub ops:      Vec<u8>,
}

impl Scenario {
    pub fn synthesize(id: u32, category: AdversarialCategory, seed: u64) -> Self {
        let mut ops = Vec::with_capacity(64);
        let mut x = seed ^ ((category as u64) << 32);
        for _ in 0..64 {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            ops.push((x & 0xFF) as u8);
        }
        Self { id, category, seed, ops }
    }

    /// Fixed-point hash of the scenario contents.
    pub fn fingerprint(&self) -> u64 {
        let mut h: u64 = 0xCBF2_9CE4_8422_2325;
        for b in self.id.to_le_bytes() { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
        h = (h ^ self.category as u64).wrapping_mul(0x100_0000_01B3);
        for b in self.seed.to_le_bytes() { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
        for &b in &self.ops { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
        h
    }
}

const ORDERED: [AdversarialCategory; CATEGORIES] = [
    AdversarialCategory::MulticoreRaceStorm,
    AdversarialCategory::MigrationCascade,
    AdversarialCategory::LockInversionAttempt,
    AdversarialCategory::CacheThrashWorkload,
    AdversarialCategory::IrqFloodCondition,
    AdversarialCategory::AffinityOscillation,
    AdversarialCategory::PartialOrderAdversarial,
    AdversarialCategory::CanonicalNearCollision,
    AdversarialCategory::SchedulerStarvation,
    AdversarialCategory::ReplayDivergenceInjection,
];

/// Generate the full 1000-scenario corpus deterministically.
pub fn generate_full() -> Vec<Scenario> {
    let mut out = Vec::with_capacity(TOTAL);
    let mut id: u32 = 0;
    for &cat in &ORDERED {
        for k in 0..PER_CATEGORY as u64 {
            let seed = ((cat as u64) << 48) ^ k.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            out.push(Scenario::synthesize(id, cat, seed));
            id += 1;
        }
    }
    out
}

/// Hash of the entire corpus — must be reproducible run-over-run.
pub fn corpus_hash(corpus: &[Scenario]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for s in corpus {
        let f = s.fingerprint();
        for b in f.to_le_bytes() { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
    }
    h
}
