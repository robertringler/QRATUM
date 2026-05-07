//! Unified runtime model — the **single canonical truth object** for
//! kernel execution paths (P0.3 directive Task 1).
//!
//! No subsystem may invent ad-hoc state. Every read or mutation that
//! affects determinism flows through `ExecutionState` and must be
//! reproducible from `(input_stream, tick_clock)` alone.
//!
//! See `spec/runtime_model.md` (added in this phase) for the formal
//! commutative diagram between kernel and host arbiter.

#![allow(dead_code)]

pub mod execution_lattice;
pub mod state_projection;
// P0.5 additions.
pub mod receipt;
pub mod receipt_chain;
pub mod receipt_hash;
// P0.6 — cryptographic Merkle anchoring + divergence envelope.
pub mod receipt_merkle;
pub mod receipt_verifier;
pub mod execution_epoch;
pub mod divergence_envelope;

pub use execution_lattice::{ExecutionState, ArbiterStateView, SchedulerStateView,
    AhtcStateView, SmpStateView, AuditCursorView, KSTATE};
pub use state_projection::{project_to_arbiter, projection_hash, KERNEL_STATE_HASH_LEN};
pub use receipt::{ExecutionReceipt, transition_hash};
pub use receipt_hash::link_hash;
