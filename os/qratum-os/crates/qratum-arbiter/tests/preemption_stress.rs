//! Preemption stress — host-side model of the kernel's preemptive
//! scheduler. We model `N` tasks each periodically submitting intents,
//! interleaved with simulated IRQ-driven preemptions. The contracts
//! the kernel must honour map onto two host invariants:
//!
//! * **No audit loss under interrupt storm**: every Submit op produces
//!   exactly one verdict (no drops, no duplicates).
//! * **Deterministic switch ordering**: re-running the same simulated
//!   workload produces the same verdict sequence.
//!
//! The kernel's preemption module (`preempt.rs`) implements the same
//! contract by gating switches on `sched_class == 1` and chaining
//! audit frames before iretq.

use qratum_arbiter::replay::{run_scenario, InputOp, ScenarioInput};
use qratum_arbiter::state_machine::{Authority, Intent, Scope};

fn workload(seed: u64, n_tasks: u32, n_ticks: u64) -> ScenarioInput {
    let mut ops = Vec::new();
    let mut x = seed.max(1);
    let mut next = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
    for tick in 0..n_ticks {
        for tid in 0..n_tasks {
            let r = next();
            let auth = match r % 4 {
                0 => Authority::User,
                1 => Authority::Console,
                2 => Authority::Service,
                _ => Authority::System,
            };
            let mut id = [0u8; 16];
            id[..8].copy_from_slice(&(tid as u64).to_le_bytes());
            id[8..].copy_from_slice(&tick.to_le_bytes());
            ops.push(InputOp::Submit(Intent {
                id, kind: ((r >> 8) & 0x7F) as u8 | 1,
                authority: auth, scope: Scope::Process,
                tick, payload_h: [tid as u8; 32],
            }));
            // Simulated preemption boundary every few ops.
            if r % 7 == 0 { ops.push(InputOp::Release); }
        }
    }
    ScenarioInput { name: format!("preempt_stress_{}_{}", n_tasks, n_ticks), ops }
}

#[test]
fn no_audit_loss_under_storm() {
    let w = workload(0xDEAD_BEEF, 8, 200);
    let submits = w.ops.iter().filter(|op| matches!(op, InputOp::Submit(_))).count();
    let r = run_scenario(&w);
    assert_eq!(r.verdict_codes.len(), submits, "every submit must yield exactly one verdict");
}

#[test]
fn deterministic_switch_ordering() {
    let w = workload(0x1234_5678, 12, 100);
    let a = run_scenario(&w);
    let b = run_scenario(&w);
    assert_eq!(a, b);
}

#[test]
fn varying_load_does_not_panic() {
    for &(t, n) in &[(2u32, 50u64), (4, 100), (16, 50), (32, 25)] {
        let w = workload(0xC0DE * t as u64, t, n);
        let _ = run_scenario(&w);
    }
}
