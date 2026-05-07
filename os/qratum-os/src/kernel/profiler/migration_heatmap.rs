//! P0.6 — Migration heatmap. Counts migrations per (src, dst) pair.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

use super::super::smp::MAX_CPUS;

const N: usize = MAX_CPUS;
static MAP: [[AtomicU64; N]; N] = [
    [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
     AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
    [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
     AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
    [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
     AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
    [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
     AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
    [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
     AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
    [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
     AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
    [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
     AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
    [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
     AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
];

pub fn note(src: u8, dst: u8) {
    if (src as usize) < N && (dst as usize) < N {
        MAP[src as usize][dst as usize].fetch_add(1, Ordering::Relaxed);
    }
}

pub fn at(src: u8, dst: u8) -> u64 {
    if (src as usize) >= N || (dst as usize) >= N { 0 }
    else { MAP[src as usize][dst as usize].load(Ordering::Relaxed) }
}

pub fn heatmap_hash() -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for i in 0..N { for j in 0..N {
        let v = MAP[i][j].load(Ordering::Relaxed);
        for b in v.to_le_bytes() { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
    }}
    h
}

pub fn reset() {
    for i in 0..N { for j in 0..N { MAP[i][j].store(0, Ordering::Release); }}
}
