//! P0.4 — Cross-core migration protocol (deterministic).
//!
//! Migration is the *only* operation that mutates two cores' state.
//! The protocol below is the canonical form; deviations trip INV-K.
//!
//! Steps (must execute IN THIS ORDER under the migration barrier):
//!
//!   1. Acquire affinity table (Acquire).
//!   2. Lock source RQ.
//!   3. Lock destination RQ (lower CPU id first if both held — but
//!      migration only ever holds two distinct queues; we order by
//!      `min(src,dst), max(src,dst)`).
//!   4. Pop task from source RQ.
//!   5. Push to destination RQ.
//!   6. Update affinity[task_id] = dst.
//!   7. Append `MigrationFrame { seq, task_id, src, dst, key_hi }` to
//!      `MIGRATION_LOG`.
//!   8. Emit one audit `Migrated` frame.
//!
//! Replay equivalence (INV-L): given the same migration log, the
//! reconciled scheduler hash is byte-identical regardless of when
//! the migrations actually executed (deterministic dispatch).

#![allow(dead_code)]

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use super::affinity;
use super::barriers::with_migration_barrier;
use super::scheduler::{dequeue_local, enqueue_local};

pub const MAX_MIGRATION_LOG: usize = 1024;

#[derive(Copy, Clone, Default, Debug, Eq, PartialEq)]
pub struct MigrationFrame {
    pub seq:     u64,
    pub task_id: u32,
    pub src:     u8,
    pub dst:     u8,
    pub _pad:    [u8; 2],
    pub key_hi:  u64,
}

#[derive(Copy, Clone, Default, Debug, Eq, PartialEq)]
pub struct MigrationReceipt {
    pub seq:        u64,
    pub task_id:    u32,
    pub src:        u8,
    pub dst:        u8,
    pub _pad:       [u8; 2],
    pub state_hash_before: u64,
    pub state_hash_after:  u64,
}

static SEQ:        AtomicU64 = AtomicU64::new(1);
static LOG_HEAD:   AtomicU32 = AtomicU32::new(0);
static LOG:        spin::Mutex<[MigrationFrame; MAX_MIGRATION_LOG]> =
    spin::Mutex::new([MigrationFrame {
        seq: 0, task_id: 0, src: 0, dst: 0, _pad: [0; 2], key_hi: 0
    }; MAX_MIGRATION_LOG]);

/// Commit one migration. Returns `None` if the source queue is empty
/// or destination is full. The caller MUST be on `src` core, or be
/// the BSP performing recovery (P0.4 only schedules from BSP).
pub fn commit(task_id: u32, src: u8, dst: u8, key_hi: u64) -> Option<MigrationReceipt> {
    if src == dst { return None; } // INV-K: no self-migration

    with_migration_barrier(|| {
        let before = super::scheduler::composed_state_hash();

        // Source pop — naive search: we only allow migration of the
        // head entry per call (the deterministic dispatcher chooses).
        let popped = dequeue_local(src)?;
        if popped.task_id != task_id {
            // Put it back and refuse — caller asked for a non-head.
            // Determinism requires: migration commits exactly the
            // task the dispatcher selected.
            enqueue_local(src, popped.task_id, popped.key_hi);
            return None;
        }
        if !enqueue_local(dst, task_id, key_hi) {
            // Destination full — restore source.
            enqueue_local(src, task_id, key_hi);
            return None;
        }

        affinity::set_home_cpu(task_id, dst);

        let seq = SEQ.fetch_add(1, Ordering::AcqRel);
        let frame = MigrationFrame {
            seq, task_id, src, dst, _pad: [0; 2], key_hi
        };
        let head = LOG_HEAD.fetch_add(1, Ordering::AcqRel) as usize % MAX_MIGRATION_LOG;
        LOG.lock()[head] = frame;

        let after = super::scheduler::composed_state_hash();

        Some(MigrationReceipt {
            seq, task_id, src, dst, _pad: [0; 2],
            state_hash_before: before,
            state_hash_after:  after,
        })
    })
}

/// Snapshot the migration log in order. Wraps once `LOG_HEAD` exceeds
/// `MAX_MIGRATION_LOG` — replay must reset state before exceeding.
pub fn log_snapshot() -> ([MigrationFrame; MAX_MIGRATION_LOG], usize) {
    let head = LOG_HEAD.load(Ordering::Acquire) as usize;
    let lim  = head.min(MAX_MIGRATION_LOG);
    (*LOG.lock(), lim)
}

/// Replay-equivalence hash: FNV-1a-64 over the *log frames* only.
/// INV-L: identical migration sequence → identical hash, regardless
/// of underlying core scheduler timing.
pub fn replay_hash() -> u64 {
    let (log, lim) = log_snapshot();
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for f in &log[..lim] {
        for b in f.seq.to_le_bytes() { h = fnv(h, b); }
        for b in f.task_id.to_le_bytes() { h = fnv(h, b); }
        h = fnv(h, f.src); h = fnv(h, f.dst);
        for b in f.key_hi.to_le_bytes() { h = fnv(h, b); }
    }
    h
}

#[inline]
fn fnv(h: u64, b: u8) -> u64 { (h ^ b as u64).wrapping_mul(0x100_0000_01B3) }

/// Reset (test-only). Production paths must not call.
pub fn reset() {
    SEQ.store(1, Ordering::Release);
    LOG_HEAD.store(0, Ordering::Release);
    *LOG.lock() = [MigrationFrame {
        seq: 0, task_id: 0, src: 0, dst: 0, _pad: [0; 2], key_hi: 0
    }; MAX_MIGRATION_LOG];
}
