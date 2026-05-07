//! `state_projection` — kernel ExecutionState → host arbiter state.
//!
//! Projection is **lossy by design**: only the fields the host arbiter
//! observes (inflight, holder, override-window state) project into the
//! arbiter view. Determinism is asserted on the *projection*, not the
//! full kernel state — see `spec/runtime_model.md` §3.

use super::execution_lattice::ExecutionState;

pub const KERNEL_STATE_HASH_LEN: usize = 32;

/// Hand-rolled FNV-1a-64 over a fixed canonical byte layout. Bytewise
/// stable across runs of the same binary (K1 from AHTC-K spec).
fn fnv1a_64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME:  u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    for &b in bytes { h ^= b as u64; h = h.wrapping_mul(PRIME); }
    h
}

/// Canonical byte serialization of `ExecutionState` for the projection
/// hash. Order is fixed and documented; never reorder.
pub fn projection_bytes(s: &ExecutionState) -> [u8; 96] {
    let mut o = [0u8; 96];
    let mut i = 0;
    macro_rules! put { ($v:expr, $n:expr) => {
        let b = $v.to_le_bytes(); o[i..i + $n].copy_from_slice(&b); i += $n;
    }; }
    put!(s.arbiter.inflight,            4);
    o[i] = s.arbiter.holder_authority; i += 1;
    o[i] = 0; i += 1; o[i] = 0; i += 1; o[i] = 0; i += 1; // pad to 8
    put!(s.arbiter.last_loc_tick,       8);

    put!(s.scheduler.current,           4);
    put!(s.scheduler.ready_count,       4);
    put!(s.scheduler.running_count,     4);
    put!(0u32,                          4); // pad
    put!(s.scheduler.switches,          8);
    put!(s.scheduler.preempt_switches,  8);

    put!(s.ahtc_k.submits,              8);
    put!(s.ahtc_k.fold_hits,            8);
    put!(s.ahtc_k.fold_inserts,         8);
    put!(s.ahtc_k.enqueue_ops,          8);
    put!(s.ahtc_k.queue_len,            4);
    o[i] = s.smp.current_cpu; i += 1;
    o[i] = 0; i += 1; o[i] = 0; i += 1; o[i] = 0; i += 1;

    put!(s.audit.live_seq,              8);
    put!(s.audit.flushed_seq,           8);
    let _ = i;
    o
}

/// 32-byte deterministic kernel state hash (FNV-1a-64 four-times
/// chained — Blake3 swap is part of the kernel-side P0.3.1 sweep).
pub fn projection_hash(s: &ExecutionState) -> [u8; KERNEL_STATE_HASH_LEN] {
    let bytes = projection_bytes(s);
    let mut out = [0u8; 32];
    let mut h = fnv1a_64(&bytes);
    for chunk in 0..4 {
        out[chunk*8..chunk*8+8].copy_from_slice(&h.to_le_bytes());
        h = fnv1a_64(&h.to_le_bytes());
    }
    out
}

/// `project_to_arbiter` — return the (inflight, holder, last_tick) tuple
/// that the **host arbiter twin** must agree with at the same logical
/// position (post-arbitrate, pre-AHTC-K). Mirrors `LockState` fields.
pub fn project_to_arbiter(s: &ExecutionState) -> (u32, u8, u64) {
    (s.arbiter.inflight, s.arbiter.holder_authority, s.arbiter.last_loc_tick)
}
