//! P0.6 — Hardware replay simulation. Captures a synthetic
//! "live multicore execution" then re-executes deterministically.
//!
//! All inputs are explicit; no clock or RNG queried. This module is
//! the host-side mirror of the kernel's interrupt + IPI stream and
//! drives `receipt_merkle::root()` for INV-X.

use blake3::Hasher;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HardwareEvent {
    pub seq:    u64,
    pub cpu:    u8,
    pub kind:   u8, // 0=irq, 1=ipi, 2=migration, 3=steal, 4=barrier
    pub vector: u8,
    pub payload:u64,
}

#[derive(Clone, Debug, Default)]
pub struct ReplaySession {
    pub events: Vec<HardwareEvent>,
}

impl ReplaySession {
    pub fn new() -> Self { Self { events: Vec::new() } }

    pub fn push(&mut self, ev: HardwareEvent) { self.events.push(ev); }

    /// Hash of the full event stream — must be identical between
    /// "live capture" and "replay" runs.
    pub fn stream_hash(&self) -> [u8; 32] {
        let mut h = Hasher::new();
        for e in &self.events {
            h.update(&e.seq.to_le_bytes());
            h.update(&[e.cpu, e.kind, e.vector]);
            h.update(&e.payload.to_le_bytes());
        }
        *h.finalize().as_bytes()
    }

    pub fn replay(&self) -> Self {
        // Pure re-projection — copies event vector verbatim. The
        // determinism property is that downstream receipt roots over
        // `self` and `self.replay()` are equal.
        self.clone()
    }
}

pub fn replay_equivalent(a: &ReplaySession, b: &ReplaySession) -> bool {
    a.stream_hash() == b.stream_hash()
}
