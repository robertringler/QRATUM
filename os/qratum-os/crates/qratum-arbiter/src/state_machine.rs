//! Host-side reference implementation of the QRATUM arbiter
//! state machine. **Authoritative** for replay tests; the kernel's
//! `intent.rs` MUST match it bit-for-bit on identical inputs.
//!
//! Spec: `spec/arbiter_state_machine.md`.

use serde::{Deserialize, Serialize};

pub const MAX_INFLIGHT: u32 = 64;
pub const DOMINANCE_WINDOW: u64 = 100;
pub const OVERRIDE_LIMIT: u32 = 4;
pub const OVERRIDE_WINDOW: u64 = 50;

#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize, Hash)]
pub enum Authority {
    User = 1,
    Console = 2,
    Service = 3,
    System = 4,
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Hash)]
pub enum Scope {
    Process = 1,
    Session = 2,
    Machine = 3,
    Fleet = 4,
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Hash)]
pub enum AcceptReason {
    Free = 0x10,
    Refresh = 0x11,
    Authority = 0x12,
    SystemOverride = 0x13,
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Hash)]
pub enum DenyReason {
    Schema = 0x20,
    QueueFull = 0x21,
    ConsoleDominance = 0x22,
    AutonomyBlock = 0x23,
    OverrideThrash = 0x24,
    AuthorityRegression = 0x25,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Hash)]
pub enum Verdict {
    Accept(AcceptReason),
    Deny(DenyReason),
}

impl Verdict {
    pub fn code(&self) -> u8 {
        match self {
            Verdict::Accept(r) => *r as u8,
            Verdict::Deny(r) => *r as u8,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Hash)]
pub struct Intent {
    pub id: [u8; 16],
    pub kind: u8,
    pub authority: Authority,
    pub scope: Scope,
    pub tick: u64,
    pub payload_h: [u8; 32],
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Default, Serialize, Deserialize, Hash)]
pub struct LockState {
    pub inflight: u32,
    pub holder_authority: Option<Authority>,
    pub last_console_tick: u64,
    pub override_count: u32,
    pub last_override_tick: u64,
}

impl LockState {
    pub fn new() -> Self { Self::default() }

    /// Mirror of `intent::release_one()` in the kernel.
    pub fn release(&mut self) {
        if self.inflight > 0 {
            self.inflight -= 1;
        }
        if self.inflight == 0 {
            self.holder_authority = None;
        }
    }
}

/// Total transition function `V : (I × L × T) → (Verdict × L')`.
///
/// Pure. No allocation. Deterministic.
pub fn arbitrate(i: &Intent, l: &LockState, t: u64) -> (Verdict, LockState) {
    let mut next = *l;

    // Schema is "kind == 0 forbidden" — no zero-kind intents.
    if i.kind == 0 {
        return (Verdict::Deny(DenyReason::Schema), *l);
    }
    if next.inflight >= MAX_INFLIGHT {
        return (Verdict::Deny(DenyReason::QueueFull), *l);
    }

    // R3 — Console dominance.
    if i.authority == Authority::Console {
        if let Some(h) = next.holder_authority {
            if h > Authority::Console
                && t.saturating_sub(next.last_console_tick) > DOMINANCE_WINDOW
            {
                return (Verdict::Deny(DenyReason::ConsoleDominance), *l);
            }
        }
        next.last_console_tick = t;
    }

    // R4 — autonomy block: System holds, lower authority arrives.
    if next.holder_authority == Some(Authority::System) && i.authority < Authority::System {
        return (Verdict::Deny(DenyReason::AutonomyBlock), *l);
    }

    // R5 — override thrash.
    if i.authority == Authority::System {
        if let Some(h) = next.holder_authority {
            if h != Authority::System
                && next.override_count >= OVERRIDE_LIMIT
                && t.saturating_sub(next.last_override_tick) < OVERRIDE_WINDOW
            {
                return (Verdict::Deny(DenyReason::OverrideThrash), *l);
            }
        }
    }

    // ----- Accept paths -----
    next.inflight += 1;
    let verdict = match next.holder_authority {
        None => {
            next.holder_authority = Some(i.authority);
            Verdict::Accept(AcceptReason::Free)
        }
        Some(h) if h == i.authority => Verdict::Accept(AcceptReason::Refresh),
        Some(h) if i.authority > h => {
            next.holder_authority = Some(i.authority);
            if i.authority == Authority::System {
                next.override_count = next.override_count.saturating_add(1);
                next.last_override_tick = t;
                Verdict::Accept(AcceptReason::SystemOverride)
            } else {
                Verdict::Accept(AcceptReason::Authority)
            }
        }
        Some(_) => {
            // i.authority < holder — pre-filtered by R4 unless holder is non-System
            // and lower-priority — defensive denial.
            return (Verdict::Deny(DenyReason::AuthorityRegression), *l);
        }
    };

    (verdict, next)
}

/// Per-tick batch: enforce total ordering by `(tick, id)` then apply.
pub fn arbitrate_batch(intents: &[Intent], l: &mut LockState) -> Vec<Verdict> {
    let mut sorted: Vec<&Intent> = intents.iter().collect();
    sorted.sort_by(|a, b| (a.tick, a.id).cmp(&(b.tick, b.id)));
    let mut out = Vec::with_capacity(sorted.len());
    for i in sorted {
        let (v, lp) = arbitrate(i, l, i.tick);
        *l = lp;
        out.push(v);
    }
    out
}
