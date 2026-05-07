//! `IntentBatch` + `RunQueue`. Fixed capacity, no_std, no allocation.

use super::key::CanonicalKey;

pub const MAX_BATCHES: usize = 64;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct IntentBatch {
    pub key:      CanonicalKey,
    pub count:    u32,
    pub first_ts: u64,
    pub last_ts:  u64,
    pub first_id: [u8; 16],
}

impl IntentBatch {
    pub const fn empty() -> Self {
        Self {
            key:      CanonicalKey::ZERO,
            count:    0,
            first_ts: 0,
            last_ts:  0,
            first_id: [0; 16],
        }
    }
}

/// Insertion-ordered fixed-capacity vector of batches. No reordering.
pub struct RunQueue {
    slots: [IntentBatch; MAX_BATCHES],
    len:   usize,
}

impl RunQueue {
    pub const fn new() -> Self {
        Self {
            slots: [IntentBatch::empty(); MAX_BATCHES],
            len:   0,
        }
    }

    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn capacity(&self) -> usize { MAX_BATCHES }

    pub fn get(&self, idx: usize) -> Option<&IntentBatch> {
        if idx < self.len { Some(&self.slots[idx]) } else { None }
    }

    pub(super) fn get_mut(&mut self, idx: usize) -> Option<&mut IntentBatch> {
        if idx < self.len { Some(&mut self.slots[idx]) } else { None }
    }

    /// Linear K5-strict find. Returns first index whose key bitwise
    /// matches `key`. No fuzzy match.
    pub fn find_exact(&self, key: &CanonicalKey) -> Option<usize> {
        let mut i = 0;
        while i < self.len {
            if self.slots[i].key == *key { return Some(i); }
            i += 1;
        }
        None
    }

    pub(super) fn push(&mut self, b: IntentBatch) -> Option<usize> {
        if self.len >= MAX_BATCHES { return None; }
        let idx = self.len;
        self.slots[idx] = b;
        self.len += 1;
        Some(idx)
    }

    pub fn clear(&mut self) {
        self.len = 0;
        for slot in self.slots.iter_mut() {
            *slot = IntentBatch::empty();
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &IntentBatch> {
        self.slots[..self.len].iter()
    }
}
