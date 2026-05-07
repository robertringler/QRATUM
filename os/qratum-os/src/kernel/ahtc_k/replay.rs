//! Replay-side helpers (kernel-resident form). The authoritative
//! replay test harness lives in `crates/qratum-arbiter::ahtc_k`; this
//! module exposes the in-kernel data the harness consumes via the
//! audit ring.
//!
//! K2: every accepted intent contributes exactly one
//! `BatchCreate`/`BatchIncrement` frame (carrying the original
//! `intent_id` + `ts`). Walking the audit log in `seq` order and
//! emitting `(intent_id, ts)` per frame reconstructs the input
//! sequence bit-for-bit.
//!
//! `expand_synthetic` is a no-op stub kept only so the `BatchExpand`
//! frame kind has a kernel emit-site if a future qsh command wants
//! one. P0.3 leaves replay reconstruction to the host arbiter crate.

use crate::kernel::audit::{self, FrameKind};
use super::batch::IntentBatch;

/// Emit a synthetic `BatchExpand` audit frame for diagnostics. Not used
/// by the determinism path — present so the audit kind has a single
/// kernel call-site (failure mode "BATCH_EXPAND missing emitter").
pub fn emit_expand_marker(batch_idx: u32, b: &IntentBatch) {
    let mut p = [0u8; 62];
    p[..4].copy_from_slice(&batch_idx.to_le_bytes());
    p[4..8].copy_from_slice(&b.count.to_le_bytes());
    audit::append(FrameKind::BatchExpand, &p[..8]);
}
