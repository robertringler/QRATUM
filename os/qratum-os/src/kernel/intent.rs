//! Kernel arbitration v0 (Agent 3, Agent 8).
//!
//! Lifts the *deny logic* from `qratum_launcher/arbitration.py` into the
//! kernel so that intent verdicts are produced inside the trusted base.
//! P0.1 mandates a kernel-level deny path — there is no "accept-all"
//! fallback (CONSTRAINT in the directive).
//!
//! Decision order (deterministic, audit-ordered):
//!   1. **Schema** — bad envelope → `DenySchema` (errno -22 EINVAL).
//!   2. **Queue full** — `MAX_INFLIGHT` exceeded → `DenyQueueFull`.
//!   3. **Authority dominance** — Console (3) cannot pre-empt System (4).
//!   4. **Autonomy block** — Autonomy (1) denied while a higher-class
//!      holder is active.
//!   5. **Override thrash** — more than `MAX_OVERRIDES_PER_TICK` Override
//!      priority intents in one LOC tick.
//!   6. Otherwise → accept with class-derived `Reason`.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};

use crate::shared::{
    IntentEnvelope, ResultEnvelope, Reason, Verdict, INTENT_MAGIC, INTENT_VERSION,
};

use super::audit::{self, FrameKind};
use super::tick;

const MAX_INFLIGHT:           u32 = 32;
const MAX_OVERRIDES_PER_TICK: u32 = 1;

static HOLDER_AUTHORITY: AtomicU8  = AtomicU8::new(0);
static INFLIGHT:         AtomicU32 = AtomicU32::new(0);
static OVERRIDE_TICK:    AtomicU64 = AtomicU64::new(0);
static OVERRIDE_COUNT:   AtomicU32 = AtomicU32::new(0);

#[derive(Debug)]
enum SubmitError {
    BadMagic, BadVersion, BadAuthority, BadScope, BadPriority, BadName,
}

impl SubmitError {
    fn errno(&self) -> i32 { -22 } // EINVAL
}

fn validate(e: &IntentEnvelope) -> Result<(), SubmitError> {
    if e.magic != INTENT_MAGIC          { return Err(SubmitError::BadMagic); }
    if e.version != INTENT_VERSION      { return Err(SubmitError::BadVersion); }
    if !(1..=4).contains(&e.authority)  { return Err(SubmitError::BadAuthority); }
    if !(1..=3).contains(&e.scope)      { return Err(SubmitError::BadScope); }
    if !(1..=4).contains(&e.priority)   { return Err(SubmitError::BadPriority); }
    if e.name_len == 0 || e.name_len as usize > e.name.len() {
        return Err(SubmitError::BadName);
    }
    Ok(())
}

fn audit_submit(env: &IntentEnvelope) {
    let mut p = [0u8; 62];
    let nm = env.name_str().as_bytes();
    let n = nm.len().min(p.len());
    p[..n].copy_from_slice(&nm[..n]);
    audit::append(FrameKind::IntentSubmit, &p);
}

fn deny(env: &IntentEnvelope, reason: Reason, errno: i32, holder: u8) -> ResultEnvelope {
    let r = ResultEnvelope {
        id: env.id,
        verdict: Verdict::Denied as u8,
        reason: reason as u8,
        holder_authority: holder,
        _pad: 0,
        error_code: errno,
        completed_ts: tick::loc_advance(),
    };
    audit::append(FrameKind::IntentVerdict, &[r.verdict, r.reason, r.holder_authority]);
    r
}

fn accept(env: &IntentEnvelope, reason: Reason) -> ResultEnvelope {
    HOLDER_AUTHORITY.store(env.authority, Ordering::SeqCst);
    INFLIGHT.fetch_add(1, Ordering::SeqCst);
    let r = ResultEnvelope {
        id: env.id,
        verdict: Verdict::Accepted as u8,
        reason: reason as u8,
        holder_authority: env.authority,
        _pad: 0,
        error_code: 0,
        completed_ts: tick::loc_advance(),
    };
    audit::append(FrameKind::IntentVerdict, &[r.verdict, r.reason, r.holder_authority]);
    r
}

/// Kernel-internal entry. Hardened: no accept-all fallback.
pub fn submit(env: &IntentEnvelope) -> ResultEnvelope {
    if let Err(e) = validate(env) {
        audit_submit(env);
        return deny(env, Reason::DenySchema, e.errno(), 0);
    }
    audit_submit(env);

    if INFLIGHT.load(Ordering::SeqCst) >= MAX_INFLIGHT {
        return deny(env, Reason::DenyQueueFull, -11,
                    HOLDER_AUTHORITY.load(Ordering::SeqCst));
    }

    let holder = HOLDER_AUTHORITY.load(Ordering::SeqCst);
    if holder == 4 && env.authority == 3 {
        return deny(env, Reason::DenyConsoleDomin, -1, holder);
    }
    if env.authority == 1 && holder > 1 {
        return deny(env, Reason::DenyAutonomyBlock, -1, holder);
    }

    let now_tick = tick::loc_now();
    let ot = OVERRIDE_TICK.load(Ordering::SeqCst);
    if env.priority == 4 {
        if ot != now_tick {
            OVERRIDE_TICK.store(now_tick, Ordering::SeqCst);
            OVERRIDE_COUNT.store(1, Ordering::SeqCst);
        } else if OVERRIDE_COUNT.fetch_add(1, Ordering::SeqCst) + 1 > MAX_OVERRIDES_PER_TICK {
            return deny(env, Reason::DenyOverrideThrash, -1, holder);
        }
    }

    let reason = match (env.authority, env.priority) {
        (4, _) => Reason::AcceptSystemOvr,
        (_, 4) => Reason::AcceptAuthority,
        _ if holder == env.authority => Reason::AcceptRefresh,
        _ => Reason::AcceptFree,
    };
    accept(env, reason)
}

pub fn make_console_intent(name: &str) -> IntentEnvelope {
    let mut env = IntentEnvelope {
        magic: INTENT_MAGIC,
        version: INTENT_VERSION,
        flags: 0,
        id: [0; 16],
        ts: tick::loc_now(),
        authority: 3,
        scope: 1,
        priority: 2,
        _pad0: 0,
        name_len: 0,
        name: [0; 64],
    };
    let bytes = name.as_bytes();
    let n = bytes.len().min(env.name.len());
    env.name[..n].copy_from_slice(&bytes[..n]);
    env.name_len = n as u16;
    let t = tick::loc_now().to_le_bytes();
    env.id[..8].copy_from_slice(&t);
    env
}

pub fn make_bad_envelope() -> IntentEnvelope {
    let mut env = make_console_intent("force.deny");
    env.magic = 0xDEADBEEF;
    env
}

pub fn inflight() -> u32 { INFLIGHT.load(Ordering::Relaxed) }
pub fn holder()   -> u8  { HOLDER_AUTHORITY.load(Ordering::Relaxed) }

/// Cooperative completion — frees one inflight slot. Clears the holder
/// authority when the queue drains to zero.
pub fn release_one() {
    let prev = INFLIGHT.fetch_update(Ordering::SeqCst, Ordering::SeqCst,
        |v| if v == 0 { None } else { Some(v - 1) });
    if let Ok(1) = prev {
        // Was 1, now 0 — clear holder.
        HOLDER_AUTHORITY.store(0, Ordering::SeqCst);
    }
}
