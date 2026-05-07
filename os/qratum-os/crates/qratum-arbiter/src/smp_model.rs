//! Host-side deterministic mirror of `kernel::smp` for P0.4 tests.
//!
//! INV-K: a task never appears on two run queues simultaneously.
//! INV-L: identical migration log → identical reconciled state hash.
//! INV-M: composed scheduler state hash is independent of physical
//!        core ordering.
//!
//! These properties are tested host-side against this mirror; kernel
//! semantics are guaranteed to match because the kernel uses the
//! identical algorithms (FNV-1a-64, fixed-iteration order).

use serde::{Deserialize, Serialize};

pub const MAX_CPUS: usize = 8;
pub const RQ_CAP:   usize = 64;

#[derive(Copy, Clone, Default, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RqEntry {
    pub task_id: u32,
    pub key_hi:  u64,
    pub enq_seq: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct CoreQueue {
    pub cpu:  u8,
    pub buf:  Vec<RqEntry>,
    pub seq:  u64,
}

impl CoreQueue {
    pub fn new(cpu: u8) -> Self { Self { cpu, buf: Vec::new(), seq: 0 } }
    pub fn push(&mut self, task_id: u32, key_hi: u64) -> bool {
        if self.buf.len() >= RQ_CAP { return false; }
        self.seq += 1;
        self.buf.push(RqEntry { task_id, key_hi, enq_seq: self.seq });
        true
    }
    pub fn pop_head(&mut self) -> Option<RqEntry> {
        if self.buf.is_empty() { None } else { Some(self.buf.remove(0)) }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SmpModel {
    pub queues:    Vec<CoreQueue>,
    pub mig_log:   Vec<MigrationFrame>,
    pub mig_seq:   u64,
    pub affinity:  Vec<(u32, u8)>,
}

#[derive(Copy, Clone, Default, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MigrationFrame {
    pub seq:     u64,
    pub task_id: u32,
    pub src:     u8,
    pub dst:     u8,
    pub key_hi:  u64,
}

impl SmpModel {
    pub fn new(online: u8) -> Self {
        assert!((online as usize) <= MAX_CPUS);
        Self {
            queues: (0..online).map(CoreQueue::new).collect(),
            mig_log: Vec::new(),
            mig_seq: 1,
            affinity: Vec::new(),
        }
    }

    /// Place a fresh task. Returns the chosen home cpu.
    pub fn create(&mut self, task_id: u32, key_hi: u64) -> u8 {
        let n = self.queues.len() as u8;
        let cpu = deterministic_placement(task_id, n);
        self.queues[cpu as usize].push(task_id, key_hi);
        self.affinity.push((task_id, cpu));
        cpu
    }

    /// INV-K: refuses if the task is already enqueued on any core.
    pub fn enqueue(&mut self, cpu: u8, task_id: u32, key_hi: u64) -> bool {
        for q in &self.queues {
            if q.buf.iter().any(|e| e.task_id == task_id) { return false; }
        }
        self.queues[cpu as usize].push(task_id, key_hi)
    }

    /// Atomic, deterministic migration. Returns false if illegal.
    pub fn migrate(&mut self, task_id: u32, src: u8, dst: u8) -> bool {
        if src == dst { return false; }
        let popped = {
            let q = &mut self.queues[src as usize];
            let pos = q.buf.iter().position(|e| e.task_id == task_id);
            match pos { Some(0) => q.buf.remove(0), _ => return false }
        };
        if !self.queues[dst as usize].push(popped.task_id, popped.key_hi) {
            // Restore source on failure.
            self.queues[src as usize].buf.insert(0, popped);
            return false;
        }
        self.mig_log.push(MigrationFrame {
            seq: self.mig_seq, task_id, src, dst, key_hi: popped.key_hi,
        });
        self.mig_seq += 1;
        for a in self.affinity.iter_mut() {
            if a.0 == task_id { a.1 = dst; }
        }
        true
    }

    /// INV-M composed scheduler state hash. Iterates 0..MAX_CPUS in
    /// fixed order — independent of physical execution.
    pub fn composed_state_hash(&self) -> u64 {
        let mut h: u64 = 0xCBF2_9CE4_8422_2325;
        for cpu in 0..MAX_CPUS {
            let q = self.queues.get(cpu);
            for byte in (cpu as u8).to_le_bytes() { h = fnv(h, byte); }
            let head = 0u32; let tail = q.map(|q| q.buf.len() as u32).unwrap_or(0);
            let seq  = q.map(|q| q.seq).unwrap_or(0);
            for byte in head.to_le_bytes() { h = fnv(h, byte); }
            for byte in tail.to_le_bytes() { h = fnv(h, byte); }
            for byte in seq.to_le_bytes()  { h = fnv(h, byte); }
        }
        h
    }

    /// INV-L: replay hash of the migration log alone. Identical
    /// migration sequences yield identical hashes regardless of
    /// when/where they ran.
    pub fn migration_replay_hash(&self) -> u64 {
        let mut h: u64 = 0xCBF2_9CE4_8422_2325;
        for f in &self.mig_log {
            for b in f.seq.to_le_bytes()     { h = fnv(h, b); }
            for b in f.task_id.to_le_bytes() { h = fnv(h, b); }
            h = fnv(h, f.src); h = fnv(h, f.dst);
            for b in f.key_hi.to_le_bytes()  { h = fnv(h, b); }
        }
        h
    }

    /// Return the set of (task_id, cpu) pairs for every queued task.
    pub fn queue_membership(&self) -> Vec<(u32, u8)> {
        let mut out = Vec::new();
        for q in &self.queues {
            for e in &q.buf { out.push((e.task_id, q.cpu)); }
        }
        out.sort();
        out
    }

    /// INV-K predicate.
    pub fn no_dual_occupancy(&self) -> bool {
        let mut seen: Vec<u32> = Vec::new();
        for q in &self.queues {
            for e in &q.buf {
                if seen.contains(&e.task_id) { return false; }
                seen.push(e.task_id);
            }
        }
        true
    }
}

/// Deterministic core selection — character-for-character identical
/// to `kernel::smp::affinity::deterministic_placement`.
pub fn deterministic_placement(task_id: u32, online: u8) -> u8 {
    if online == 0 { return 0; }
    let mut x = task_id as u64 ^ 0x9E37_79B9_7F4A_7C15;
    x ^= x >> 30; x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27; x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x % online as u64) as u8
}

#[inline]
fn fnv(h: u64, b: u8) -> u64 { (h ^ b as u64).wrapping_mul(0x100_0000_01B3) }
