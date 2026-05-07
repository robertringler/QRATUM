//! Persistent audit (P0.3 skeleton, format v2).
//!
//! Format spec: `audit/format_v2.md`. The block-device backing is
//! deferred to P0.3.1; this module owns the segment-builder + chain
//! hashing logic so that the on-disk layout is finalized today and the
//! flush is a one-call swap once a block driver lands.
//!
//! Intentionally `no_std`: uses a tiny incremental Blake3-like
//! function. We **do not** pull in the `blake3` crate here because the
//! kernel must boot without an allocator-backed dependency and Blake3
//! relies on `std` for SIMD detection. The host crate
//! (`qratum-arbiter`) uses real Blake3; this kernel-side digest is a
//! placeholder marked `XXX-blake3` for the P0.3.1 swap.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

pub const SEG_SIZE: usize = 4096;
pub const HEADER_SIZE: usize = 32;
pub const TRAILER_SIZE: usize = 32;
pub const MAX_PAYLOAD: usize = 256;

const MAGIC: u32 = u32::from_le_bytes(*b"QAUD");
const VERSION: u16 = 2;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct SegmentHeader {
    pub magic: u32,
    pub version: u16,
    pub flags: u16,
    pub seq: u64,
    pub frame_count: u32,
    pub bytes_used: u32,
    pub prev_tail_h: u64,
}

impl SegmentHeader {
    pub fn new(seq: u64, prev_tail_h: u64) -> Self {
        Self { magic: MAGIC, version: VERSION, flags: 0, seq, frame_count: 0,
               bytes_used: HEADER_SIZE as u32, prev_tail_h }
    }
    pub fn write_into(&self, buf: &mut [u8; HEADER_SIZE]) {
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        buf[6..8].copy_from_slice(&self.flags.to_le_bytes());
        buf[8..16].copy_from_slice(&self.seq.to_le_bytes());
        buf[16..20].copy_from_slice(&self.frame_count.to_le_bytes());
        buf[20..24].copy_from_slice(&self.bytes_used.to_le_bytes());
        buf[24..32].copy_from_slice(&self.prev_tail_h.to_le_bytes());
    }
}

pub struct SegmentBuilder {
    pub buf: [u8; SEG_SIZE],
    pub header: SegmentHeader,
    /// 32-byte rolling chain hash (placeholder — see file-level comment).
    pub chain: [u8; 32],
    pub sealed: bool,
}

impl SegmentBuilder {
    pub const fn new(seq: u64) -> Self {
        Self {
            buf: [0u8; SEG_SIZE],
            header: SegmentHeader { magic: MAGIC, version: VERSION, flags: 0,
                                    seq, frame_count: 0,
                                    bytes_used: HEADER_SIZE as u32, prev_tail_h: 0 },
            chain: [0u8; 32], sealed: false,
        }
    }

    /// Append a frame. Returns false if the segment is full.
    pub fn append(&mut self, kind: u8, seq: u32, payload: &[u8]) -> bool {
        if self.sealed { return false; }
        let plen = payload.len().min(MAX_PAYLOAD);
        let frame_len = 7 + plen; // u16 len + u8 kind + u32 seq + payload
        if (self.header.bytes_used as usize) + frame_len + TRAILER_SIZE > SEG_SIZE {
            return false;
        }
        let off = self.header.bytes_used as usize;
        self.buf[off..off+2].copy_from_slice(&(plen as u16).to_le_bytes());
        self.buf[off+2] = kind;
        self.buf[off+3..off+7].copy_from_slice(&seq.to_le_bytes());
        self.buf[off+7..off+7+plen].copy_from_slice(&payload[..plen]);

        // Update chain hash — XXX-blake3: replace with real Blake3 in P0.3.1.
        weak_chain(&mut self.chain, &self.buf[off..off+frame_len]);

        self.header.bytes_used += frame_len as u32;
        self.header.frame_count += 1;
        true
    }

    /// Seal the segment by writing the trailer + flushing the header.
    pub fn seal(&mut self) {
        if self.sealed { return; }
        let mut hb = [0u8; HEADER_SIZE];
        self.header.flags |= 1; // sealed
        self.header.write_into(&mut hb);
        self.buf[..HEADER_SIZE].copy_from_slice(&hb);
        let trailer_off = SEG_SIZE - TRAILER_SIZE;
        self.buf[trailer_off..].copy_from_slice(&self.chain);
        self.sealed = true;
    }
}

/// Placeholder chain function — XOR-feedback over rolling SHA-style mix.
/// Replaced by Blake3 in P0.3.1 when allocator-free Blake3 lands.
fn weak_chain(state: &mut [u8; 32], frame: &[u8]) {
    let mut acc: u64 = u64::from_le_bytes([
        state[0], state[1], state[2], state[3],
        state[4], state[5], state[6], state[7],
    ]);
    for &b in frame {
        acc = acc.wrapping_mul(0x100000001b3).wrapping_add(b as u64);
    }
    let mix = acc.to_le_bytes();
    for i in 0..32 { state[i] ^= mix[i % 8]; }
}

// ---- Active segment registry (single-core, P0.3) ----

static ACTIVE_SEQ: AtomicU64 = AtomicU64::new(0);
static FLUSHED: AtomicU64 = AtomicU64::new(0);

pub fn next_seq() -> u64 { ACTIVE_SEQ.fetch_add(1, Ordering::AcqRel) }
pub fn flushed_count() -> u64 { FLUSHED.load(Ordering::Relaxed) }

/// Mark one segment as flushed (call from the future block-driver path).
pub fn note_flush() { FLUSHED.fetch_add(1, Ordering::Release); }

// ============================================================
// Closure invariant (P0.3 Task 5)
// ============================================================
//
// The closure rule: every kernel state transition emits exactly one
// audit event. P0.3 enforces the *counting* form of this invariant —
// the live audit ring sequence (`audit::count()`) must never lag the
// arbiter+ahtc-k transition counter, and must never exceed it by more
// than the number of independent emit-sites (`MAX_INDEPENDENT_EMITS`).
//
// `MAX_INDEPENDENT_EMITS` is the count of single-shot emit-sites that
// run outside the arbiter→ahtc-k pipeline (BootBegin/BootReady/
// SyscallEnter/SyscallExit/Preempt/IrqEnter/PageFault/etc.). It is
// declared here so that the invariant is checkable without scanning
// the kernel.

pub const MAX_INDEPENDENT_EMITS: u64 = 1024;

/// Returns true iff `audit::count() == ahtc_k::stats().submits +
/// (independent emits ≤ MAX_INDEPENDENT_EMITS)`. Caller passes
/// `transition_count` from `runtime_model::ExecutionState`. Failure
/// trips an invariant in the strict variant `audit_replay_assert`.
pub fn audit_closure_holds(audit_seq: u64, transition_count: u64) -> bool {
    audit_seq >= transition_count
        && audit_seq <= transition_count.saturating_add(MAX_INDEPENDENT_EMITS)
}

/// Strict form — trips `invariants::trip` on failure (no silent drop).
pub fn audit_replay_assert(audit_seq: u64, transition_count: u64) {
    if !audit_closure_holds(audit_seq, transition_count) {
        super::invariants::trip(
            super::invariants::InvariantCode::AuditDrop,
            b"audit_persist:closure_violated",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn build_and_seal() {
        let mut b = SegmentBuilder::new(1);
        for i in 0..100u32 { assert!(b.append(0x10, i, b"hello world")); }
        b.seal();
        assert!(b.sealed);
        assert!(b.header.frame_count >= 100);
        assert!(b.buf[0..4] == *b"QAUD");
    }
}
