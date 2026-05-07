//! P0.5 — Lock acquisition graph + cycle detector (INV-R).
//!
//! Every lock acquired is added to a directed adjacency list keyed by
//! the *previous* lock id held on the same logical thread. A cycle in
//! this graph is a deadlock-class hazard.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

pub const MAX_LOCKS: usize = 64;
pub const MAX_EDGES: usize = 256;

#[derive(Copy, Clone, Default, Debug)]
struct Edge { from: u32, to: u32 }

static mut EDGES: [Edge; MAX_EDGES] = [Edge { from: 0, to: 0 }; MAX_EDGES];
static EDGE_COUNT: AtomicU32 = AtomicU32::new(0);
static EDGE_HASH:  AtomicU64 = AtomicU64::new(0);

/// Record a `from → to` lock-acquisition edge. Callers should pass
/// `from = 0` if no prior lock is held.
pub fn record_edge(from: u32, to: u32) {
    let i = EDGE_COUNT.fetch_add(1, Ordering::AcqRel) as usize;
    if i < MAX_EDGES {
        unsafe { EDGES[i] = Edge { from, to }; }
    }
    let mut h = EDGE_HASH.load(Ordering::Acquire);
    for b in from.to_le_bytes() { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
    for b in to.to_le_bytes()   { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
    EDGE_HASH.store(h, Ordering::Release);
}

/// Return true if the recorded graph contains any directed cycle.
/// O(V·E) DFS — bounded by `MAX_LOCKS × MAX_EDGES`.
pub fn has_cycle() -> bool {
    let n = (EDGE_COUNT.load(Ordering::Acquire) as usize).min(MAX_EDGES);
    let edges = unsafe { &EDGES[..n] };
    let mut color = [0u8; MAX_LOCKS]; // 0=white 1=gray 2=black
    for v in 0..MAX_LOCKS as u32 {
        if color[v as usize] == 0 && dfs(v, edges, &mut color) {
            return true;
        }
    }
    false
}

fn dfs(v: u32, edges: &[Edge], color: &mut [u8; MAX_LOCKS]) -> bool {
    if (v as usize) >= MAX_LOCKS { return false; }
    color[v as usize] = 1;
    for e in edges {
        if e.from != v { continue; }
        if (e.to as usize) >= MAX_LOCKS { continue; }
        match color[e.to as usize] {
            1 => return true,
            0 => if dfs(e.to, edges, color) { return true; },
            _ => {}
        }
    }
    color[v as usize] = 2;
    false
}

#[inline]
pub fn edge_count() -> u32 { EDGE_COUNT.load(Ordering::Acquire) }

#[inline]
pub fn edge_hash() -> u64 { EDGE_HASH.load(Ordering::Acquire) }

pub fn reset() {
    EDGE_COUNT.store(0, Ordering::Release);
    EDGE_HASH.store(0, Ordering::Release);
    unsafe { EDGES = [Edge { from: 0, to: 0 }; MAX_EDGES]; }
}
