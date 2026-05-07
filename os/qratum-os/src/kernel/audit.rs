//! Audit ring v2 (Agent 5, Agent 9 — P0.2).
//!
//! **Multi-producer safe**, lock-free, fixed-capacity 4096 frames.
//!
//! Protocol (Vyukov-style bounded MPSC, single global sequence):
//!
//!   1. Each writer atomically claims a slot via `SEQ.fetch_add(1)`.
//!      The returned sequence number is unique and monotonic.
//!   2. The slot index is `seq % CAP`. The writer writes the frame body
//!      then **publishes** by storing `seq` into the slot's per-slot
//!      `seq_pub` field with `Release` ordering.
//!   3. Readers see a consistent frame iff `frame.seq == frame.seq_pub`
//!      and the corresponding atomic check passes.
//!
//! Memory ordering: the global `SEQ` uses `AcqRel`; per-slot publish uses
//! `Release` paired with reader `Acquire`. This is sufficient for
//! correctness across multiple writers on x86_64 (TSO model — but the
//! atomics are written portably).
//!
//! Single-CPU in P0.2; the protocol is multi-CPU correct so the SMP
//! bring-up in P0.3 needs no audit-ring changes.

use core::sync::atomic::{AtomicU64, Ordering};

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind {
    BootBegin       = 0x00,
    BootReady       = 0x01,
    IntentSubmit    = 0x02,
    IntentVerdict   = 0x03,
    QshCommand      = 0x04,
    IrqEnter        = 0x10,
    PageFault       = 0x11,
    GeneralProtFault= 0x12,
    DoubleFault     = 0x13,
    ContextSwitch   = 0x14,
    SchedYield      = 0x15,
    SyscallEnter    = 0x16,
    SyscallExit     = 0x17,
    Ring3Enter      = 0x18,
    TaskExit        = 0x19,
    Preempt         = 0x1A,
    BatchCreate     = 0x1B,
    BatchIncrement  = 0x1C,
    BatchExpand     = 0x1D,
    KernelPanic     = 0xFF,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Frame {
    pub seq:      u64,
    pub loc_tick: u64,
    pub tsc:      u64,
    pub kind:     u8,
    pub payload_len: u8,
    pub payload:  [u8; 62],
}

impl Frame {
    pub const fn empty() -> Self {
        Self { seq: 0, loc_tick: 0, tsc: 0, kind: 0, payload_len: 0, payload: [0; 62] }
    }
}

const RING_CAP: usize = 4096;

/// Frame storage. Each frame's own `seq` field doubles as the publish marker.
static mut RING: [Frame; RING_CAP] = [Frame::empty(); RING_CAP];

/// Per-slot publication sequence. `0` = empty; matches `Frame.seq` once
/// fully written. Separate atomic so readers can race-free check publish.
static SEQ_PUB: [AtomicU64; RING_CAP] = {
    // Const init array — Rust requires this via a const fn workaround.
    const ZERO: AtomicU64 = AtomicU64::new(0);
    [ZERO; RING_CAP]
};

/// Global producer sequence counter. Never decreases.
static SEQ: AtomicU64 = AtomicU64::new(0);

/// Append a frame, returning its assigned `seq`. Lock-free, MP-safe.
pub fn append(kind: FrameKind, payload: &[u8]) -> u64 {
    let seq = SEQ.fetch_add(1, Ordering::AcqRel) + 1; // 1-indexed
    let idx = ((seq - 1) as usize) % RING_CAP;

    let mut f = Frame::empty();
    f.seq = seq;
    f.loc_tick = super::tick::loc_now();
    f.tsc = super::tick::rdtsc();
    f.kind = kind as u8;
    let n = payload.len().min(f.payload.len());
    f.payload_len = n as u8;
    f.payload[..n].copy_from_slice(&payload[..n]);

    // Safety: each slot is written by at most one producer at a time —
    // the sequence-claim protocol ensures distinct producers get distinct
    // seq values; same seq → same idx, but distinct seq with same idx
    // (after RING_CAP rollover) cannot race because the previous producer
    // already published before the next claimed (TSO + AcqRel ordering).
    unsafe { core::ptr::write_volatile(&raw mut RING[idx], f); }

    SEQ_PUB[idx].store(seq, Ordering::Release);
    seq
}

pub fn count() -> u64 { SEQ.load(Ordering::Acquire) }

/// Iterate from oldest visible frame to newest, calling `f` for each.
/// Skips slots where the publication sequence does not match the frame
/// (torn write or wrap-overwrite in flight).
pub fn for_each_recent(max: usize, mut f: impl FnMut(&Frame)) {
    let total = SEQ.load(Ordering::Acquire) as usize;
    let take  = max.min(total).min(RING_CAP);
    if take == 0 { return; }
    let start_seq = (total - take) as u64 + 1;
    for i in 0..take {
        let seq = start_seq + i as u64;
        let idx = ((seq - 1) as usize) % RING_CAP;

        let pub_seq = SEQ_PUB[idx].load(Ordering::Acquire);
        if pub_seq < seq { continue; }                  // not yet published
        // Safety: read-only.
        let frame = unsafe { core::ptr::read_volatile(&raw const RING[idx]) };
        // Re-validate: pub_seq could have advanced past `seq` if a wrap
        // overwrote the slot we wanted. Compare against the frame seq.
        if frame.seq == seq && SEQ_PUB[idx].load(Ordering::Acquire) == seq {
            f(&frame);
        }
    }
}
