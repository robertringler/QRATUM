//! Deterministic golden-corpus generator.
//!
//! Produces ≥200 named scenarios covering the categories required by the
//! P0.3 directive. Generation is **seeded** — same seed → byte-identical
//! manifest. The seed is a compile-time constant; bumping it is a
//! deliberate corpus refresh and MUST be reviewed.

use crate::replay::{InputOp, ScenarioInput};
use crate::state_machine::{Authority, Intent, Scope};

const SEED: u64 = 0x4752_4154_554d_5033; // "GRATUMP3"

/// Tiny xorshift64* PRNG — no external deps, deterministic.
#[derive(Clone, Copy)]
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed.max(1)) }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn u8(&mut self) -> u8 { self.next() as u8 }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T { xs[(self.next() as usize) % xs.len()] }
}

fn id_from(seed: u64, idx: u32) -> [u8; 16] {
    let mut out = [0u8; 16];
    let a = seed ^ (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let b = a.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    out[..8].copy_from_slice(&a.to_le_bytes());
    out[8..].copy_from_slice(&b.to_le_bytes());
    out
}

fn payload(seed: u64) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(&seed.to_le_bytes());
    *h.finalize().as_bytes()
}

fn mk_intent(rng: &mut Rng, idx: u32, tick: u64, auth: Authority, scope: Scope, kind: u8) -> Intent {
    Intent {
        id: id_from(rng.next(), idx),
        kind,
        authority: auth,
        scope,
        tick,
        payload_h: payload(rng.next()),
    }
}

fn submit(i: Intent) -> InputOp { InputOp::Submit(i) }
fn release() -> InputOp { InputOp::Release }

/// Category 1 — concurrent conflicting intents at the same tick.
fn cat_concurrent(out: &mut Vec<ScenarioInput>) {
    let mut rng = Rng::new(SEED ^ 0xC0_C0);
    for n in 0u32..50 {
        let mut ops = Vec::new();
        let tick = 100u64 + n as u64;
        let auths = [Authority::User, Authority::Console, Authority::Service, Authority::System];
        for k in 0u32..4 {
            ops.push(submit(mk_intent(&mut rng, n * 4 + k, tick, auths[k as usize], Scope::Process, 1)));
        }
        ops.push(release());
        out.push(ScenarioInput { name: format!("concurrent_{:03}", n), ops });
    }
}

/// Category 2 — identical-tick tie cases (same authority, varying ids).
fn cat_tie(out: &mut Vec<ScenarioInput>) {
    let mut rng = Rng::new(SEED ^ 0x7E_7E);
    for n in 0u32..30 {
        let tick = 200u64 + n as u64;
        let mut ops = Vec::new();
        for k in 0u32..3 {
            ops.push(submit(mk_intent(&mut rng, n * 10 + k, tick, Authority::Service, Scope::Session, 2)));
        }
        out.push(ScenarioInput { name: format!("tie_{:03}", n), ops });
    }
}

/// Category 3 — authority escalation attempts (User → Console → Service → System).
fn cat_escalation(out: &mut Vec<ScenarioInput>) {
    let mut rng = Rng::new(SEED ^ 0xE5_E5);
    let chain = [Authority::User, Authority::Console, Authority::Service, Authority::System];
    for n in 0u32..40 {
        let mut ops = Vec::new();
        for (k, a) in chain.iter().enumerate() {
            ops.push(submit(mk_intent(&mut rng, n * 4 + k as u32, 300u64 + n as u64 + k as u64, *a, Scope::Machine, 3)));
        }
        out.push(ScenarioInput { name: format!("escalate_{:03}", n), ops });
    }
}

/// Category 4 — override thrash: System repeatedly preempts.
fn cat_thrash(out: &mut Vec<ScenarioInput>) {
    let mut rng = Rng::new(SEED ^ 0x77_77);
    for n in 0u32..30 {
        let mut ops = Vec::new();
        ops.push(submit(mk_intent(&mut rng, n * 10, 400u64 + n as u64, Authority::Service, Scope::Process, 4)));
        for k in 0u32..6 {
            ops.push(release());
            ops.push(submit(mk_intent(&mut rng, n * 10 + 100 + k, 400u64 + n as u64 + k as u64, Authority::Service, Scope::Process, 4)));
            ops.push(submit(mk_intent(&mut rng, n * 10 + 200 + k, 400u64 + n as u64 + k as u64, Authority::System, Scope::Process, 4)));
        }
        out.push(ScenarioInput { name: format!("thrash_{:03}", n), ops });
    }
}

/// Category 5 — scheduler + interrupt interactions modelled as
/// interleaved tick-batches with releases between them.
fn cat_sched_irq(out: &mut Vec<ScenarioInput>) {
    let mut rng = Rng::new(SEED ^ 0x12_34);
    for n in 0u32..30 {
        let mut ops = Vec::new();
        let mut tick: u64 = 500 + n as u64 * 10;
        for k in 0u32..8 {
            let a = if k % 2 == 0 { Authority::Service } else { Authority::System };
            ops.push(submit(mk_intent(&mut rng, n * 10 + k, tick, a, Scope::Process, 5)));
            tick += 1;
            if k % 3 == 2 { ops.push(release()); }
        }
        out.push(ScenarioInput { name: format!("sched_irq_{:03}", n), ops });
    }
}

/// Category 6 — syscall/int-0x80 parity envelopes (kind=6 marks legacy
/// path, kind=7 marks MSR path; both must produce the same verdict).
fn cat_syscall_parity(out: &mut Vec<ScenarioInput>) {
    let mut rng = Rng::new(SEED ^ 0x80_C0);
    for n in 0u32..30 {
        for &kind in &[6u8, 7u8] {
            let id_seed = rng.next();
            let id = id_from(id_seed, n);
            let i = Intent {
                id,
                kind,
                authority: Authority::User,
                scope: Scope::Process,
                tick: 600 + n as u64,
                payload_h: payload(id_seed),
            };
            out.push(ScenarioInput {
                name: format!("syscall_parity_kind{}_{:03}", kind, n),
                ops: vec![submit(i)],
            });
        }
    }
}

/// Category 7 — queue-full backpressure (forces `MAX_INFLIGHT` denials).
fn cat_queue_full(out: &mut Vec<ScenarioInput>) {
    let mut rng = Rng::new(SEED ^ 0x6F_6F);
    for n in 0u32..15 {
        let mut ops = Vec::new();
        for k in 0u32..70 {
            ops.push(submit(mk_intent(&mut rng, n * 100 + k, 700u64 + n as u64, Authority::Service, Scope::Fleet, 7)));
        }
        out.push(ScenarioInput { name: format!("queue_full_{:03}", n), ops });
    }
}

/// Category 8 — schema-violation envelopes.
fn cat_schema(out: &mut Vec<ScenarioInput>) {
    let mut rng = Rng::new(SEED ^ 0x05_05);
    for n in 0u32..15 {
        let mut i = mk_intent(&mut rng, n, 800u64 + n as u64, Authority::User, Scope::Process, 0);
        i.kind = 0; // forbidden
        out.push(ScenarioInput { name: format!("schema_{:03}", n), ops: vec![submit(i)] });
    }
}

pub fn build_corpus() -> Vec<ScenarioInput> {
    let mut all = Vec::with_capacity(256);
    cat_concurrent(&mut all);
    cat_tie(&mut all);
    cat_escalation(&mut all);
    cat_thrash(&mut all);
    cat_sched_irq(&mut all);
    cat_syscall_parity(&mut all);
    cat_queue_full(&mut all);
    cat_schema(&mut all);
    all
}

#[cfg(test)]
mod sanity {
    use super::*;
    #[test]
    fn corpus_at_least_200() {
        let c = build_corpus();
        assert!(c.len() >= 200, "corpus must be >= 200 scenarios, got {}", c.len());
    }
}
