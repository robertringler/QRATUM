//! P0.5 — NUMA distance + locality metrics (kernel side).

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

pub const MAX_NODES: usize = 4;

/// Static distance matrix (lower = closer). Identity on the diagonal.
const fn default_distance() -> [[u8; MAX_NODES]; MAX_NODES] {
    let mut m = [[10u8; MAX_NODES]; MAX_NODES];
    let mut i = 0;
    while i < MAX_NODES { m[i][i] = 0; i += 1; }
    m
}

static mut DISTANCE: [[u8; MAX_NODES]; MAX_NODES] = default_distance();
static LOCAL_HITS:   AtomicU64 = AtomicU64::new(0);
static REMOTE_HITS:  AtomicU64 = AtomicU64::new(0);

pub fn set_distance(a: u8, b: u8, d: u8) {
    if (a as usize) < MAX_NODES && (b as usize) < MAX_NODES {
        unsafe { DISTANCE[a as usize][b as usize] = d; }
    }
}

#[inline]
pub fn distance(a: u8, b: u8) -> u8 {
    if (a as usize) >= MAX_NODES || (b as usize) >= MAX_NODES { 255 }
    else { unsafe { DISTANCE[a as usize][b as usize] } }
}

#[inline]
pub fn note_access(node_self: u8, node_target: u8) {
    if node_self == node_target { LOCAL_HITS.fetch_add(1, Ordering::Relaxed); }
    else                         { REMOTE_HITS.fetch_add(1, Ordering::Relaxed); }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct NumaSummary {
    pub local_hits:  u64,
    pub remote_hits: u64,
    pub locality_q:  u64, // local / (local+remote) × 1e6
}

pub const SCALE: u64 = 1_000_000;

pub fn summary() -> NumaSummary {
    let l = LOCAL_HITS.load(Ordering::Relaxed);
    let r = REMOTE_HITS.load(Ordering::Relaxed);
    let q = if l + r == 0 { 0 } else { l * SCALE / (l + r) };
    NumaSummary { local_hits: l, remote_hits: r, locality_q: q }
}

pub fn reset() {
    LOCAL_HITS.store(0, Ordering::Release);
    REMOTE_HITS.store(0, Ordering::Release);
}
