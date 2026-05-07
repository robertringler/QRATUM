//! P0.5 — Host execution-receipt chain mirror.

use serde::{Deserialize, Serialize};

#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum TransitionKind {
    Submit  = 0,
    Fold    = 1,
    Migrate = 2,
    Replay  = 3,
}

#[derive(Copy, Clone, Default, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExecutionReceipt {
    pub seq:               u64,
    pub kind:              u8,
    pub canonical_key_lo:  u64,
    pub canonical_key_hi:  u64,
    pub arbitration_link:  u64,
    pub migration_link:    u64,
    pub replay_link:       u64,
}

#[inline]
fn fnv_step(h: u64, b: u8) -> u64 { (h ^ b as u64).wrapping_mul(0x100_0000_01B3) }

pub fn transition_hash(r: &ExecutionReceipt) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for b in r.seq.to_le_bytes()              { h = fnv_step(h, b); }
    h = fnv_step(h, r.kind);
    for b in r.canonical_key_lo.to_le_bytes() { h = fnv_step(h, b); }
    for b in r.canonical_key_hi.to_le_bytes() { h = fnv_step(h, b); }
    for b in r.arbitration_link.to_le_bytes() { h = fnv_step(h, b); }
    for b in r.migration_link.to_le_bytes()   { h = fnv_step(h, b); }
    for b in r.replay_link.to_le_bytes()      { h = fnv_step(h, b); }
    h
}

pub fn link_hash(prev: u64, transition: u64, arbitration: u64, migration: u64) -> u64 {
    let mut h = prev;
    for b in transition.to_le_bytes()  { h = fnv_step(h, b); }
    for b in arbitration.to_le_bytes() { h = fnv_step(h, b); }
    for b in migration.to_le_bytes()   { h = fnv_step(h, b); }
    h
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReceiptChain {
    pub entries:    Vec<ExecutionReceipt>,
    pub chain_hash: u64,
}

impl ReceiptChain {
    pub fn new() -> Self { Self { entries: Vec::new(), chain_hash: 0xCBF2_9CE4_8422_2325 } }

    pub fn append(&mut self, mut r: ExecutionReceipt) -> u64 {
        r.seq = self.entries.len() as u64;
        let th = transition_hash(&r);
        self.chain_hash = link_hash(self.chain_hash, th, r.arbitration_link, r.migration_link);
        self.entries.push(r);
        self.chain_hash
    }

    /// Replay an existing chain into a fresh chain — must reproduce
    /// the same final hash exactly.
    pub fn replay(&self) -> u64 {
        let mut copy = ReceiptChain::new();
        for r in &self.entries { copy.append(*r); }
        copy.chain_hash
    }

    pub fn integrity_holds(&self) -> bool { self.replay() == self.chain_hash }
}
