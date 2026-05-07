//! P0.6 — Host model of replay checkpoint save/restore.

use crate::receipt_merkle::ReceiptMerkle;

#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub seq:           u64,
    pub epoch:         u64,
    pub merkle_root:   [u8; 32],
    pub audit_cursor:  u64,
    pub barrier_hash:  u64,
}

#[derive(Default, Debug)]
pub struct CheckpointStore {
    pub items: Vec<Checkpoint>,
}

impl CheckpointStore {
    pub fn save(&mut self, epoch: u64, m: &ReceiptMerkle, audit_cursor: u64, barrier_hash: u64) -> u64 {
        let seq = self.items.len() as u64;
        self.items.push(Checkpoint {
            seq, epoch, merkle_root: m.root(),
            audit_cursor, barrier_hash,
        });
        seq
    }

    pub fn load(&self, seq: u64) -> Option<&Checkpoint> {
        self.items.get(seq as usize)
    }

    /// Restore must produce the same merkle root from the same leaf
    /// stream — this checks the property given a candidate restored
    /// merkle.
    pub fn restore_matches(&self, seq: u64, restored: &ReceiptMerkle) -> bool {
        match self.load(seq) {
            Some(c) => c.merkle_root == restored.root(),
            None => false,
        }
    }
}
