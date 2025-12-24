//! Merkle Ledger Implementation
//!
//! Append-only, zone-aware, reversible ledger with Merkle tree structure.
//! Implements zone promotion logic (Z0→Z1→Z2→Z3) and rollback capability.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use alloc::string::String;
use sha3::{Digest, Sha3_256};
use minicbor::{Encode, Decode};

use crate::txo::TXO;
use crate::rtf::api::{Zone, RTFError};

/// Merkle ledger node
#[derive(Debug, Clone, Encode, Decode)]
pub struct LedgerNode {
    /// Node hash (SHA3-256)
    #[n(0)]
    pub node_hash: [u8; 32],
    
    /// Parent node hash (creates chain)
    #[n(1)]
    pub parent_hash: [u8; 32],
    
    /// TXO hash
    #[n(2)]
    pub txo_hash: [u8; 32],
    
    /// Epoch ID
    #[n(3)]
    pub epoch_id: u64,
    
    /// Zone
    #[n(4)]
    pub zone: u8, // 0=Z0, 1=Z1, 2=Z2, 3=Z3
    
    /// Timestamp
    #[n(5)]
    pub timestamp: u64,
}

impl LedgerNode {
    /// Create a new ledger node
    pub fn new(
        parent_hash: [u8; 32],
        txo_hash: [u8; 32],
        epoch_id: u64,
        zone: Zone,
        timestamp: u64,
    ) -> Self {
        let zone_id = match zone {
            Zone::Z0 => 0,
            Zone::Z1 => 1,
            Zone::Z2 => 2,
            Zone::Z3 => 3,
        };
        
        // Compute node hash
        let mut hasher = Sha3_256::new();
        hasher.update(&parent_hash);
        hasher.update(&txo_hash);
        hasher.update(&epoch_id.to_le_bytes());
        hasher.update(&[zone_id]);
        hasher.update(&timestamp.to_le_bytes());
        
        let result = hasher.finalize();
        let mut node_hash = [0u8; 32];
        node_hash.copy_from_slice(&result);
        
        Self {
            node_hash,
            parent_hash,
            txo_hash,
            epoch_id,
            zone: zone_id,
            timestamp,
        }
    }
}

/// Epoch snapshot for rollback
#[derive(Debug, Clone, Encode, Decode)]
pub struct EpochSnapshot {
    /// Epoch ID
    #[n(0)]
    pub epoch_id: u64,
    
    /// Merkle root at this epoch
    #[n(1)]
    pub merkle_root: [u8; 32],
    
    /// Number of nodes at this epoch
    #[n(2)]
    pub node_count: usize,
    
    /// Zone at this epoch
    #[n(3)]
    pub zone: u8,
    
    /// Timestamp
    #[n(4)]
    pub timestamp: u64,
}

/// Merkle ledger - append-only with zone awareness
pub struct MerkleLedger {
    /// Genesis root (immutable anchor)
    genesis_root: [u8; 32],
    
    /// Current Merkle root
    current_root: [u8; 32],
    
    /// All ledger nodes (append-only)
    nodes: Vec<LedgerNode>,
    
    /// Epoch snapshots for rollback
    snapshots: Vec<EpochSnapshot>,
    
    /// Current zone
    current_zone: Zone,
}

impl MerkleLedger {
    /// Create a new Merkle ledger
    ///
    /// # Arguments
    /// * `genesis_root` - Genesis Merkle root (Z0 anchor)
    pub fn new(genesis_root: [u8; 32]) -> Self {
        // Create genesis snapshot
        let genesis_snapshot = EpochSnapshot {
            epoch_id: 0,
            merkle_root: genesis_root,
            node_count: 0,
            zone: 0, // Z0
            timestamp: 0,
        };
        
        Self {
            genesis_root,
            current_root: genesis_root,
            nodes: Vec::new(),
            snapshots: alloc::vec![genesis_snapshot],
            current_zone: Zone::Z0,
        }
    }
    
    /// Append a TXO to the ledger
    ///
    /// # Arguments
    /// * `txo` - Transaction object to append
    /// * `zone` - Current zone
    pub fn append_txo(&mut self, txo: &TXO, zone: Zone) {
        let txo_hash = txo.compute_hash();
        
        let node = LedgerNode::new(
            self.current_root,
            txo_hash,
            txo.epoch_id,
            zone,
            txo.timestamp,
        );
        
        // Update current root
        self.current_root = node.node_hash;
        
        // Append node
        self.nodes.push(node);
    }
    
    /// Create a snapshot at current epoch
    ///
    /// # Arguments
    /// * `epoch_id` - Epoch identifier
    /// * `timestamp` - Snapshot timestamp
    pub fn create_snapshot(&mut self, epoch_id: u64, timestamp: u64) {
        let zone_id = match self.current_zone {
            Zone::Z0 => 0,
            Zone::Z1 => 1,
            Zone::Z2 => 2,
            Zone::Z3 => 3,
        };
        
        let snapshot = EpochSnapshot {
            epoch_id,
            merkle_root: self.current_root,
            node_count: self.nodes.len(),
            zone: zone_id,
            timestamp,
        };
        
        self.snapshots.push(snapshot);
    }
    
    /// Rollback to a previous epoch
    ///
    /// # Arguments
    /// * `target_epoch` - Epoch to rollback to
    ///
    /// # Returns
    /// * `Ok(())` if rollback succeeds
    /// * `Err(RTFError)` if rollback fails
    pub fn rollback_to_epoch(&mut self, target_epoch: u64) -> Result<(), RTFError> {
        // Find target snapshot
        let snapshot = self.snapshots
            .iter()
            .find(|s| s.epoch_id == target_epoch)
            .ok_or(RTFError::EpochNotFound)?;
        
        // Restore state from snapshot
        self.current_root = snapshot.merkle_root;
        self.nodes.truncate(snapshot.node_count);
        
        // Remove snapshots after target epoch
        self.snapshots.retain(|s| s.epoch_id <= target_epoch);
        
        Ok(())
    }
    
    /// Promote to next zone
    ///
    /// # Arguments
    /// * `target_zone` - Zone to promote to
    ///
    /// # Returns
    /// * `Ok(())` if promotion succeeds
    /// * `Err(RTFError)` if promotion fails
    pub fn promote_zone(&mut self, target_zone: Zone) -> Result<(), RTFError> {
        // Validate zone transition
        let valid_transition = match (self.current_zone, target_zone) {
            (Zone::Z0, Zone::Z1) => true,
            (Zone::Z1, Zone::Z2) => true,
            (Zone::Z2, Zone::Z3) => true,
            _ => false,
        };
        
        if !valid_transition {
            return Err(RTFError::InvalidZoneTransition);
        }
        
        // Update current zone
        self.current_zone = target_zone;
        
        Ok(())
    }
    
    /// Get current Merkle root
    pub fn get_current_root(&self) -> [u8; 32] {
        self.current_root
    }
    
    /// Get genesis root
    pub fn get_genesis_root(&self) -> [u8; 32] {
        self.genesis_root
    }
    
    /// Get number of nodes in ledger
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    
    /// Get current zone
    pub fn current_zone(&self) -> Zone {
        self.current_zone
    }
    
    /// Verify Merkle chain integrity
    ///
    /// # Returns
    /// * `true` if chain is valid, `false` otherwise
    pub fn verify_chain(&self) -> bool {
        if self.nodes.is_empty() {
            return true;
        }
        
        // Verify first node links to genesis
        if self.nodes[0].parent_hash != self.genesis_root {
            return false;
        }
        
        // Verify each subsequent node links to previous
        for i in 1..self.nodes.len() {
            if self.nodes[i].parent_hash != self.nodes[i - 1].node_hash {
                return false;
            }
        }
        
        // Verify current root matches last node
        if let Some(last_node) = self.nodes.last() {
            if self.current_root != last_node.node_hash {
                return false;
            }
        }
        
        true
    }
    
    /// Export ledger to CBOR
    pub fn to_cbor(&self) -> Result<Vec<u8>, minicbor::encode::Error<core::convert::Infallible>> {
        let mut buffer = Vec::new();
        let mut encoder = minicbor::Encoder::new(&mut buffer);
        
        // Encode genesis root
        encoder.array(3)?;
        encoder.bytes(&self.genesis_root)?;
        
        // Encode nodes
        encoder.array(self.nodes.len() as u64)?;
        for node in &self.nodes {
            node.encode(&mut encoder, &mut ())?;
        }
        
        // Encode snapshots
        encoder.array(self.snapshots.len() as u64)?;
        for snapshot in &self.snapshots {
            snapshot.encode(&mut encoder, &mut ())?;
        }
        
        Ok(buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::txo::{Sender, Receiver, Payload, IdentityType, OperationClass, PayloadType};
    
    #[test]
    fn test_ledger_creation() {
        let genesis_root = [1u8; 32];
        let ledger = MerkleLedger::new(genesis_root);
        
        assert_eq!(ledger.get_genesis_root(), genesis_root);
        assert_eq!(ledger.get_current_root(), genesis_root);
        assert_eq!(ledger.node_count(), 0);
    }
    
    #[test]
    fn test_append_txo() {
        let genesis_root = [1u8; 32];
        let mut ledger = MerkleLedger::new(genesis_root);
        
        let sender = Sender {
            identity_type: IdentityType::Operator,
            id: [2u8; 16],
            biokey_present: false,
            fido2_signed: false,
            zk_proof: None,
        };
        
        let receiver = Receiver {
            identity_type: IdentityType::Node,
            id: [3u8; 16],
        };
        
        let payload = Payload {
            payload_type: PayloadType::Genome,
            content_hash: [4u8; 32],
            encrypted: true,
        };
        
        let txo = TXO::new(
            [5u8; 16],
            sender,
            receiver,
            OperationClass::Genomic,
            payload,
        );
        
        ledger.append_txo(&txo, Zone::Z1);
        
        assert_eq!(ledger.node_count(), 1);
        assert_ne!(ledger.get_current_root(), genesis_root);
    }
    
    #[test]
    fn test_chain_verification() {
        let genesis_root = [1u8; 32];
        let mut ledger = MerkleLedger::new(genesis_root);
        
        let sender = Sender {
            identity_type: IdentityType::Operator,
            id: [2u8; 16],
            biokey_present: false,
            fido2_signed: false,
            zk_proof: None,
        };
        
        let receiver = Receiver {
            identity_type: IdentityType::Node,
            id: [3u8; 16],
        };
        
        let payload = Payload {
            payload_type: PayloadType::Genome,
            content_hash: [4u8; 32],
            encrypted: true,
        };
        
        // Append multiple TXOs
        for i in 0..5 {
            let mut txo = TXO::new(
                [i as u8; 16],
                sender.clone(),
                receiver.clone(),
                OperationClass::Genomic,
                payload.clone(),
            );
            txo.epoch_id = i as u64;
            ledger.append_txo(&txo, Zone::Z1);
        }
        
        // Verify chain integrity
        assert!(ledger.verify_chain());
    }
    
    #[test]
    fn test_snapshot_and_rollback() {
        let genesis_root = [1u8; 32];
        let mut ledger = MerkleLedger::new(genesis_root);
        
        let sender = Sender {
            identity_type: IdentityType::Operator,
            id: [2u8; 16],
            biokey_present: false,
            fido2_signed: false,
            zk_proof: None,
        };
        
        let receiver = Receiver {
            identity_type: IdentityType::Node,
            id: [3u8; 16],
        };
        
        let payload = Payload {
            payload_type: PayloadType::Genome,
            content_hash: [4u8; 32],
            encrypted: true,
        };
        
        // Append TXO and create snapshot at epoch 1
        let mut txo1 = TXO::new(
            [5u8; 16],
            sender.clone(),
            receiver.clone(),
            OperationClass::Genomic,
            payload.clone(),
        );
        txo1.epoch_id = 1;
        ledger.append_txo(&txo1, Zone::Z1);
        ledger.create_snapshot(1, 1000);
        
        let root_at_epoch_1 = ledger.get_current_root();
        
        // Append another TXO at epoch 2
        let mut txo2 = TXO::new(
            [6u8; 16],
            sender,
            receiver,
            OperationClass::Genomic,
            payload,
        );
        txo2.epoch_id = 2;
        ledger.append_txo(&txo2, Zone::Z1);
        
        assert_eq!(ledger.node_count(), 2);
        
        // Rollback to epoch 1
        ledger.rollback_to_epoch(1).unwrap();
        
        assert_eq!(ledger.node_count(), 1);
        assert_eq!(ledger.get_current_root(), root_at_epoch_1);
    }
    
    #[test]
    fn test_zone_promotion() {
        let genesis_root = [1u8; 32];
        let mut ledger = MerkleLedger::new(genesis_root);
        
        assert_eq!(ledger.current_zone(), Zone::Z0);
        
        // Z0 -> Z1
        assert!(ledger.promote_zone(Zone::Z1).is_ok());
        assert_eq!(ledger.current_zone(), Zone::Z1);
        
        // Z1 -> Z2
        assert!(ledger.promote_zone(Zone::Z2).is_ok());
        assert_eq!(ledger.current_zone(), Zone::Z2);
        
        // Z2 -> Z0 (invalid)
        assert_eq!(ledger.promote_zone(Zone::Z0), Err(RTFError::InvalidZoneTransition));
    }
}
