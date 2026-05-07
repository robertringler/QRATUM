//! `qratum-arbiter` — host-side reference arbiter, replay harness,
//! golden-corpus generator. The kernel-side `intent.rs` is the
//! authoritative target; this crate is its formal twin and the
//! source of CI determinism guarantees.

pub mod state_machine;
pub mod replay;
pub mod corpus;
pub mod ahtc_k;
pub mod kernel_equivalence;
pub mod performance;
pub mod proof_gate;
pub mod smp_model;
pub mod hardware_profile;
pub mod performance_truth;
// P0.5 additions.
pub mod numa_model;
pub mod adversarial_corpus;
pub mod execution_receipt;
// P0.6 — real multicore + hardware-bound determinism.
pub mod hardware_replay;
pub mod receipt_merkle;
pub mod deterministic_steal;
pub mod divergence_analysis;
pub mod replay_checkpoint;
pub mod topology_model;
pub mod apic_model;
