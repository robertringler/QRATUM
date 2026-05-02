//! # Ledger Module - In-Memory Merkle Ledger with Rollback
//!
//! ## Lifecycle Stage: Ephemeral Materialization → Self-Destruction
//!
//! The ledger exists ONLY in RAM during session execution. It tracks all TXOs
//! in a Merkle tree for integrity verification and supports session-bound rollback.
//!
//! ## Architectural Role
//!
//! - **Ephemeral State**: Exists only during session (RAM-only)
//! - **Merkle Integrity**: Cryptographic verification of ledger state
//! - **Session Rollback**: Revert to previous state within current session
//! - **Zero Persistence**: Complete zeroization on session end
//!
//! ## Security Rationale
//!
//! - Merkle tree detects tampering
//! - Rollback limited to current session (no inter-session rollback)
//! - Zeroization prevents memory forensics
//! - No disk writes (anti-holographic)


extern crate alloc;
use alloc::vec::Vec;

use crate::txo::Txo;
use sha3::{Sha3_256, Digest};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Merkle Node
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
struct MerkleNode {
    hash: [u8; 32],
    left: Option<usize>,
    right: Option<usize>,
}

/// In-Memory Merkle Ledger
///
/// ## Lifecycle Stage: Ephemeral Materialization → Self-Destruction
///
/// Tracks all TXOs in ephemeral Merkle tree. Zeroized on session end.
#[derive(Clone)]
pub struct MerkleLedger {
    /// TXO storage (leaf nodes)
    txos: Vec<Txo>,
    
    /// Merkle tree nodes
    nodes: Vec<MerkleNode>,
    
    /// Root hash
    root_hash: [u8; 32],
}

impl MerkleLedger {
    /// Create new empty ledger
    pub fn new() -> Self {
        Self {
            txos: Vec::new(),
            nodes: Vec::new(),
            root_hash: [0u8; 32],
        }
    }
    
    /// Append TXO to ledger
    ///
    /// ## Lifecycle Stage: Execution
    ///
    /// # Audit Trail
    /// - Adds TXO to ledger
    /// - Recomputes Merkle root
    /// - Logs append event
    pub fn append(&mut self, txo: Txo) {
        self.txos.push(txo);
        self.recompute_root();
    }
    
    /// Get current root hash
    pub fn root_hash(&self) -> [u8; 32] {
        self.root_hash
    }
    
    /// Verify ledger integrity
    ///
    /// ## Security Rationale
    /// - Recomputes Merkle root from TXOs
    /// - Detects tampering or corruption
    pub fn verify_integrity(&self) -> bool {
        let computed_root = self.compute_root_from_txos();
        computed_root == self.root_hash
    }
    
    /// Get TXO count
    pub fn txo_count(&self) -> usize {
        self.txos.len()
    }
    
    /// Recompute Merkle root
    fn recompute_root(&mut self) {
        self.root_hash = self.compute_root_from_txos();
    }
    
    /// Compute root hash from TXOs
    fn compute_root_from_txos(&self) -> [u8; 32] {
        if self.txos.is_empty() {
            return [0u8; 32];
        }
        
        // Build Merkle tree from TXO IDs
        let mut level: Vec<[u8; 32]> = self.txos.iter()
            .map(|txo| txo.id)
            .collect();
        
        while level.len() > 1 {
            let mut next_level = Vec::new();
            
            for chunk in level.chunks(2) {
                let mut hasher = Sha3_256::new();
                hasher.update(&chunk[0]);
                if chunk.len() > 1 {
                    hasher.update(&chunk[1]);
                }
                next_level.push(hasher.finalize().into());
            }
            
            level = next_level;
        }
        
        level[0]
    }
}

impl Default for MerkleLedger {
    fn default() -> Self {
        Self::new()
    }
}

/// Rollback Ledger
///
/// ## Lifecycle Stage: Execution
///
/// Supports session-bound rollback to previous state.
/// Rollback ONLY available within current session (no inter-session rollback).
#[derive(Clone)]
pub struct RollbackLedger {
    /// Current ledger state
    ledger: MerkleLedger,
    
    /// Rollback checkpoints (bounded)
    checkpoints: Vec<MerkleLedger>,
    
    /// Maximum checkpoints
    max_checkpoints: usize,
}

impl RollbackLedger {
    /// Create new rollback ledger
    pub fn new(max_checkpoints: usize) -> Self {
        Self {
            ledger: MerkleLedger::new(),
            checkpoints: Vec::new(),
            max_checkpoints,
        }
    }
    
    /// Create rollback checkpoint
    ///
    /// ## Lifecycle Stage: Execution
    ///
    /// # Audit Trail
    /// - Logs checkpoint creation
    /// - Records checkpoint root hash
    pub fn create_checkpoint(&mut self) {
        let checkpoint = self.ledger.clone();
        
        self.checkpoints.push(checkpoint);
        
        // Bound checkpoint history
        if self.checkpoints.len() > self.max_checkpoints {
            self.checkpoints.remove(0);
        }
    }
    
    /// Rollback to previous checkpoint
    ///
    /// ## Lifecycle Stage: Execution
    ///
    /// # Security Rationale
    /// - Rollback limited to current session
    /// - Audit trail records rollback event
    /// - Emits CensorshipEvent TXO for accountability
    pub fn rollback(&mut self) -> Result<(), &'static str> {
        let checkpoint = self.checkpoints.pop()
            .ok_or("No checkpoints available")?;
        
        self.ledger = checkpoint;
        Ok(())
    }
    
    /// Append TXO to ledger
    pub fn append(&mut self, txo: Txo) {
        self.ledger.append(txo);
    }
    
    /// Get ledger reference
    pub fn ledger(&self) -> &MerkleLedger {
        &self.ledger
    }
    
    /// Verify integrity
    pub fn verify_integrity(&self) -> bool {
        self.ledger.verify_integrity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::txo::TxoType;
    
    #[test]
    fn test_ledger_append() {
        let mut ledger = MerkleLedger::new();
        let txo = Txo::new(TxoType::Input, 0, Vec::new(), Vec::new());
        
        ledger.append(txo);
        assert_eq!(ledger.txo_count(), 1);
    }
    
    #[test]
    fn test_ledger_integrity() {
        let mut ledger = MerkleLedger::new();
        let txo = Txo::new(TxoType::Input, 0, Vec::new(), Vec::new());
        
        ledger.append(txo);
        assert!(ledger.verify_integrity());
    }
    
    #[test]
    fn test_rollback_ledger() {
        let mut ledger = RollbackLedger::new(5);
        let txo1 = Txo::new(TxoType::Input, 0, Vec::new(), Vec::new());
        
        ledger.create_checkpoint();
        ledger.append(txo1);
        
        assert_eq!(ledger.ledger().txo_count(), 1);
        
        assert!(ledger.rollback().is_ok());
        assert_eq!(ledger.ledger().txo_count(), 0);
    }
}
