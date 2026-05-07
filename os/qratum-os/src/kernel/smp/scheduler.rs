//! P0.4 — Per-core scheduler queues.
//!
//! Determinism contract:
//!   1. Per-core run queues are append-only within a core.
//!   2. Cross-core enqueues are canonicalized via `migration::commit`
//!      and pass through the migration barrier.
//!   3. Reconciliation uses fixed-order iteration (cpu 0..N), so the
//!      composed scheduler hash is independent of physical core
//!      execution interleaving (INV-M).

#![allow(dead_code)]

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use super::core_local::CoreLocal;
use super::MAX_CPUS;

pub const RQ_CAPACITY: usize = 64;

#[derive(Copy, Clone, Default)]
pub struct RqEntry {
    pub task_id:  u32,
    pub key_hi:   u64,
    pub enq_seq:  u64,
}

#[derive(Copy, Clone)]
pub struct RunQueueLocal {
    pub buf:  [RqEntry; RQ_CAPACITY],
    pub head: u32,
    pub tail: u32,
    pub seq:  u64,
}

impl Default for RunQueueLocal {
    fn default() -> Self { Self::new() }
}

impl RunQueueLocal {
    pub const fn new() -> Self {
        Self {
            buf: [RqEntry { task_id: 0, key_hi: 0, enq_seq: 0 }; RQ_CAPACITY],
            head: 0, tail: 0, seq: 0,
        }
    }
    pub fn len(&self) -> u32 { self.tail.wrapping_sub(self.head) }
    pub fn is_full(&self)  -> bool { self.len() as usize == RQ_CAPACITY }
    pub fn is_empty(&self) -> bool { self.head == self.tail }

    pub fn push(&mut self, task_id: u32, key_hi: u64) -> bool {
        if self.is_full() { return false; }
        self.seq = self.seq.wrapping_add(1);
        let e = RqEntry { task_id, key_hi, enq_seq: self.seq };
        let idx = (self.tail as usize) % RQ_CAPACITY;
        self.buf[idx] = e;
        self.tail = self.tail.wrapping_add(1);
        true
    }

    pub fn pop(&mut self) -> Option<RqEntry> {
        if self.is_empty() { return None; }
        let idx = (self.head as usize) % RQ_CAPACITY;
        let e = self.buf[idx];
        self.head = self.head.wrapping_add(1);
        Some(e)
    }
}

pub static QUEUES: CoreLocal<RunQueueLocal> = CoreLocal::new(RunQueueLocal::new());

/// Per-core enqueue counters — total enqueues observed by each core,
/// for the reconciliation path.
pub static ENQUEUES: [AtomicU64; MAX_CPUS] = {
    const Z: AtomicU64 = AtomicU64::new(0);
    [Z, Z, Z, Z, Z, Z, Z, Z]
};
pub static DEQUEUES: [AtomicU64; MAX_CPUS] = {
    const Z: AtomicU64 = AtomicU64::new(0);
    [Z, Z, Z, Z, Z, Z, Z, Z]
};

#[inline]
pub fn enqueue_local(cpu: u8, task_id: u32, key_hi: u64) -> bool {
    // SAFETY: per-core slot, accessed from owning CPU only.
    let ok = unsafe { QUEUES.with(cpu, |q| q.push(task_id, key_hi)) };
    if ok { ENQUEUES[cpu as usize].fetch_add(1, Ordering::Relaxed); }
    ok
}

#[inline]
pub fn dequeue_local(cpu: u8) -> Option<RqEntry> {
    let r = unsafe { QUEUES.with(cpu, |q| q.pop()) };
    if r.is_some() { DEQUEUES[cpu as usize].fetch_add(1, Ordering::Relaxed); }
    r
}

/// Composed scheduler state hash — INV-M.
///
/// FNV-1a-64 over (cpu_id, head, tail, seq) for cpus in fixed order.
/// Independent of physical execution interleaving by construction.
pub fn composed_state_hash() -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for cpu in 0..MAX_CPUS as u8 {
        // SAFETY: read-only snapshot under barrier — callers must hold.
        unsafe {
            QUEUES.with(cpu, |q| {
                for byte in cpu.to_le_bytes() { h = fnv(h, byte); }
                for byte in q.head.to_le_bytes() { h = fnv(h, byte); }
                for byte in q.tail.to_le_bytes() { h = fnv(h, byte); }
                for byte in q.seq.to_le_bytes()  { h = fnv(h, byte); }
            });
        }
    }
    h
}

#[inline]
fn fnv(h: u64, b: u8) -> u64 { (h ^ b as u64).wrapping_mul(0x100_0000_01B3) }

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct PerCoreCounters {
    pub cpu_id:   u8,
    pub enqueues: u64,
    pub dequeues: u64,
    pub depth:    u32,
}

pub fn counters() -> [PerCoreCounters; MAX_CPUS] {
    let mut out = [PerCoreCounters::default(); MAX_CPUS];
    for cpu in 0..MAX_CPUS {
        // SAFETY: read-only snapshot.
        let depth = unsafe { QUEUES.with(cpu as u8, |q| q.len()) };
        out[cpu] = PerCoreCounters {
            cpu_id:   cpu as u8,
            enqueues: ENQUEUES[cpu].load(Ordering::Relaxed),
            dequeues: DEQUEUES[cpu].load(Ordering::Relaxed),
            depth,
        };
    }
    out
}

#[allow(dead_code)]
static GLOBAL_TASK_SEQ: AtomicU32 = AtomicU32::new(1);
