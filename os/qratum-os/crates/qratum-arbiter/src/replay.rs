//! Replay harness — produces deterministic event streams + hashes.
//!
//! `ReplayEvent` is the canonical event-log entry. The audit-tail
//! hash is `Blake3(prev_tail || serialize(event))`. This is the same
//! chaining rule the kernel's persistent audit (`audit_persist.rs`)
//! commits to disk — so a host replay can verify any kernel log
//! byte-for-byte.

use serde::{Deserialize, Serialize};

use crate::state_machine::{arbitrate, Intent, LockState, Verdict};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Hash)]
pub enum ReplayEvent {
    Submit { intent: Intent, verdict: Verdict, lock_after: LockState },
    Release { lock_after: LockState },
}

#[derive(Clone, Debug, Default)]
pub struct ReplayLog {
    pub events: Vec<ReplayEvent>,
    pub tail_hash: [u8; 32],
}

impl ReplayLog {
    pub fn new() -> Self { Self::default() }

    pub fn push(&mut self, ev: ReplayEvent) {
        let bytes = serde_json::to_vec(&ev).expect("canonical json");
        let mut h = blake3::Hasher::new();
        h.update(&self.tail_hash);
        h.update(&bytes);
        self.tail_hash = *h.finalize().as_bytes();
        self.events.push(ev);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum InputOp {
    Submit(Intent),
    Release,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScenarioInput {
    pub name: String,
    pub ops: Vec<InputOp>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct ScenarioExpected {
    pub name: String,
    pub verdict_codes: Vec<u8>,
    pub lock_state_hash: String,
    pub audit_tail_hash: String,
}

/// Apply the input deterministically. Returns the expected manifest entry.
pub fn run_scenario(s: &ScenarioInput) -> ScenarioExpected {
    let mut l = LockState::new();
    let mut log = ReplayLog::new();
    let mut codes: Vec<u8> = Vec::with_capacity(s.ops.len());
    for op in &s.ops {
        match op {
            InputOp::Submit(i) => {
                let (v, lp) = arbitrate(i, &l, i.tick);
                l = lp;
                codes.push(v.code());
                log.push(ReplayEvent::Submit { intent: *i, verdict: v, lock_after: l });
            }
            InputOp::Release => {
                l.release();
                log.push(ReplayEvent::Release { lock_after: l });
            }
        }
    }
    let lock_bytes = serde_json::to_vec(&l).unwrap();
    let lock_hash = blake3::hash(&lock_bytes);
    ScenarioExpected {
        name: s.name.clone(),
        verdict_codes: codes,
        lock_state_hash: lock_hash.to_hex().to_string(),
        audit_tail_hash: blake3::Hash::from_bytes(log.tail_hash).to_hex().to_string(),
    }
}
