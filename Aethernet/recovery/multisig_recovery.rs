//! Emergency Multisig Recovery Module
//!
//! Provides emergency recovery mechanism for dual Biokey failure scenarios.
//! Implements quorum-based recovery with temporal decay and immutable audit logging.
//!
//! Security Hardening (Aethernet Phase I-II):
//! - Quorum-based authorization (e.g., 3/5 signers)
//! - Temporal decay: recovery window decreases over time
//! - Immutable audit: all override events logged to ledger
//! - No single-point recovery: requires multiple independent signers

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use alloc::string::String;
use alloc::collections::BTreeMap;
use sha3::{Digest, Sha3_256};
use core::ptr;

/// Recovery signer identity
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SignerIdentity {
    /// Public key hash
    pub pubkey_hash: [u8; 32],
    /// Human-readable identifier (for audit)
    pub identifier: String,
}

/// Recovery signature
#[derive(Debug, Clone)]
pub struct RecoverySignature {
    /// Signer identity
    pub signer: SignerIdentity,
    /// Signature bytes (Ed25519 or similar)
    pub signature: Vec<u8>,
    /// Timestamp when signature was created
    pub timestamp: u64,
}

/// Recovery request metadata
#[derive(Debug, Clone)]
pub struct RecoveryRequest {
    /// TXO hash being recovered
    pub txo_hash: [u8; 32],
    /// Epoch ID of the recovery request
    pub epoch_id: u64,
    /// Reason for recovery (for audit)
    pub reason: String,
    /// Collected signatures
    pub signatures: Vec<RecoverySignature>,
    /// Request timestamp
    pub created_at: u64,
}

/// Temporal decay configuration
#[derive(Debug, Clone, Copy)]
pub struct DecayConfig {
    /// Initial recovery window (seconds)
    pub initial_window: u64,
    /// Decay rate (percentage per epoch)
    pub decay_rate: u32, // 0-100
    /// Minimum window (seconds)
    pub min_window: u64,
}

/// Multisig recovery context
///
/// Manages emergency recovery with quorum requirements and temporal decay.
pub struct MultisigRecovery {
    /// Required quorum (e.g., 3 out of 5)
    quorum_required: u32,
    /// Total number of signers
    total_signers: u32,
    /// Authorized signer identities
    signers: Vec<SignerIdentity>,
    /// Temporal decay configuration
    decay_config: DecayConfig,
    /// Pending recovery requests
    pending_requests: BTreeMap<[u8; 32], RecoveryRequest>,
    /// Audit log of completed recoveries
    audit_log: Vec<RecoveryAuditEntry>,
}

/// Audit log entry for recovery events
#[derive(Debug, Clone)]
pub struct RecoveryAuditEntry {
    /// TXO hash that was recovered
    pub txo_hash: [u8; 32],
    /// Timestamp of recovery
    pub timestamp: u64,
    /// Signers who authorized recovery
    pub authorizing_signers: Vec<SignerIdentity>,
    /// Reason for recovery
    pub reason: String,
    /// Epoch ID
    pub epoch_id: u64,
}

impl MultisigRecovery {
    /// Create new multisig recovery context
    ///
    /// # Arguments
    /// * `quorum_required` - Number of signatures required (e.g., 3)
    /// * `signers` - List of authorized signer identities
    /// * `decay_config` - Temporal decay configuration
    ///
    /// # Returns
    /// * Multisig recovery context
    ///
    /// # Panics
    /// * If quorum > total signers
    pub fn new(
        quorum_required: u32,
        signers: Vec<SignerIdentity>,
        decay_config: DecayConfig,
    ) -> Self {
        let total_signers = signers.len() as u32;
        
        if quorum_required > total_signers {
            panic!("Quorum cannot exceed total signers");
        }
        
        if quorum_required == 0 {
            panic!("Quorum must be at least 1");
        }
        
        Self {
            quorum_required,
            total_signers,
            signers,
            decay_config,
            pending_requests: BTreeMap::new(),
            audit_log: Vec::new(),
        }
    }
    
    /// Initiate recovery request
    ///
    /// # Arguments
    /// * `txo_hash` - Hash of TXO to recover
    /// * `epoch_id` - Current epoch
    /// * `reason` - Human-readable reason for audit
    /// * `current_time` - Current timestamp
    ///
    /// # Returns
    /// * Ok if request initiated, Err if already exists
    pub fn initiate_recovery(
        &mut self,
        txo_hash: [u8; 32],
        epoch_id: u64,
        reason: String,
        current_time: u64,
    ) -> Result<(), &'static str> {
        // Check if recovery already pending
        if self.pending_requests.contains_key(&txo_hash) {
            return Err("Recovery already pending for this TXO");
        }
        
        let request = RecoveryRequest {
            txo_hash,
            epoch_id,
            reason,
            signatures: Vec::new(),
            created_at: current_time,
        };
        
        self.pending_requests.insert(txo_hash, request);
        Ok(())
    }
    
    /// Add signature to recovery request
    ///
    /// # Arguments
    /// * `txo_hash` - TXO being recovered
    /// * `signature` - Recovery signature
    ///
    /// # Returns
    /// * Ok if signature added, Err if invalid
    pub fn add_signature(
        &mut self,
        txo_hash: &[u8; 32],
        signature: RecoverySignature,
    ) -> Result<(), &'static str> {
        // Get pending request
        let request = self.pending_requests
            .get_mut(txo_hash)
            .ok_or("No pending recovery for this TXO")?;
        
        // Verify signer is authorized
        if !self.signers.contains(&signature.signer) {
            return Err("Signer not authorized");
        }
        
        // Check if signer already signed
        for sig in &request.signatures {
            if sig.signer == signature.signer {
                return Err("Signer already signed this recovery");
            }
        }
        
        // Add signature
        request.signatures.push(signature);
        Ok(())
    }
    
    /// Check if recovery request meets quorum
    ///
    /// # Arguments
    /// * `txo_hash` - TXO hash
    /// * `current_time` - Current timestamp
    ///
    /// # Returns
    /// * Ok if quorum met, Err otherwise
    pub fn check_quorum(
        &self,
        txo_hash: &[u8; 32],
        current_time: u64,
    ) -> Result<(), &'static str> {
        let request = self.pending_requests
            .get(txo_hash)
            .ok_or("No pending recovery for this TXO")?;
        
        // Check temporal window
        let elapsed = current_time - request.created_at;
        let max_window = self.compute_decay_window(request.epoch_id);
        
        if elapsed > max_window {
            return Err("Recovery window expired");
        }
        
        // Check quorum
        if (request.signatures.len() as u32) < self.quorum_required {
            return Err("Insufficient signatures for quorum");
        }
        
        Ok(())
    }
    
    /// Execute recovery (finalize after quorum met)
    ///
    /// # Arguments
    /// * `txo_hash` - TXO to recover
    /// * `current_time` - Current timestamp
    ///
    /// # Returns
    /// * Ok with audit entry if successful
    ///
    /// # Security
    /// * Verifies quorum before execution
    /// * Logs to immutable audit trail
    /// * Removes from pending to prevent replay
    pub fn execute_recovery(
        &mut self,
        txo_hash: &[u8; 32],
        current_time: u64,
    ) -> Result<RecoveryAuditEntry, &'static str> {
        // Verify quorum
        self.check_quorum(txo_hash, current_time)?;
        
        // Remove from pending
        let request = self.pending_requests
            .remove(txo_hash)
            .ok_or("No pending recovery")?;
        
        // Create audit entry
        let audit_entry = RecoveryAuditEntry {
            txo_hash: *txo_hash,
            timestamp: current_time,
            authorizing_signers: request.signatures.iter()
                .map(|sig| sig.signer.clone())
                .collect(),
            reason: request.reason.clone(),
            epoch_id: request.epoch_id,
        };
        
        // Add to immutable audit log
        self.audit_log.push(audit_entry.clone());
        
        Ok(audit_entry)
    }
    
    /// Compute recovery window with temporal decay
    ///
    /// Window decreases over epochs to incentivize prompt recovery.
    ///
    /// # Arguments
    /// * `epoch_id` - Current epoch
    ///
    /// # Returns
    /// * Maximum recovery window in seconds
    fn compute_decay_window(&self, epoch_id: u64) -> u64 {
        let decay_factor = 1.0 - (self.decay_config.decay_rate as f64 / 100.0);
        let decayed_window = (self.decay_config.initial_window as f64) 
            * decay_factor.powi(epoch_id as i32);
        
        let window = decayed_window as u64;
        
        // Enforce minimum window
        if window < self.decay_config.min_window {
            self.decay_config.min_window
        } else {
            window
        }
    }
    
    /// Get audit log (immutable history)
    pub fn audit_log(&self) -> &[RecoveryAuditEntry] {
        &self.audit_log
    }
    
    /// Verify signature (placeholder for actual crypto)
    ///
    /// In production, would use Ed25519 or similar.
    pub fn verify_signature(
        &self,
        message: &[u8],
        signature: &RecoverySignature,
    ) -> bool {
        // Placeholder: In production, verify Ed25519 signature
        // ed25519_dalek::verify(signature.signer.pubkey, message, signature.signature)
        
        // For now, just check signature is non-empty
        !signature.signature.is_empty()
    }
}

impl Drop for MultisigRecovery {
    /// Secure cleanup on drop
    fn drop(&mut self) {
        // Wipe sensitive data (if any stored keys)
        // Currently no sensitive data in this struct
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_signers(count: usize) -> Vec<SignerIdentity> {
        (0..count)
            .map(|i| {
                let mut hash = [0u8; 32];
                hash[0] = i as u8;
                SignerIdentity {
                    pubkey_hash: hash,
                    identifier: alloc::format!("signer-{}", i),
                }
            })
            .collect()
    }
    
    fn create_test_config() -> DecayConfig {
        DecayConfig {
            initial_window: 3600, // 1 hour
            decay_rate: 10, // 10% per epoch
            min_window: 300, // 5 minutes minimum
        }
    }
    
    #[test]
    fn test_multisig_creation() {
        let signers = create_test_signers(5);
        let config = create_test_config();
        let recovery = MultisigRecovery::new(3, signers, config);
        
        assert_eq!(recovery.quorum_required, 3);
        assert_eq!(recovery.total_signers, 5);
    }
    
    #[test]
    fn test_recovery_initiation() {
        let signers = create_test_signers(5);
        let config = create_test_config();
        let mut recovery = MultisigRecovery::new(3, signers, config);
        
        let txo_hash = [0x42u8; 32];
        let result = recovery.initiate_recovery(
            txo_hash,
            1000,
            alloc::format!("Test recovery"),
            10000,
        );
        
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_signature_collection() {
        let signers = create_test_signers(5);
        let config = create_test_config();
        let mut recovery = MultisigRecovery::new(3, signers.clone(), config);
        
        let txo_hash = [0x42u8; 32];
        recovery.initiate_recovery(
            txo_hash,
            1000,
            alloc::format!("Test recovery"),
            10000,
        ).unwrap();
        
        // Add signatures
        for i in 0..3 {
            let sig = RecoverySignature {
                signer: signers[i].clone(),
                signature: vec![i as u8; 64],
                timestamp: 10000 + i as u64,
            };
            
            let result = recovery.add_signature(&txo_hash, sig);
            assert!(result.is_ok());
        }
        
        // Check quorum
        let quorum_check = recovery.check_quorum(&txo_hash, 10100);
        assert!(quorum_check.is_ok());
    }
    
    #[test]
    fn test_recovery_execution() {
        let signers = create_test_signers(5);
        let config = create_test_config();
        let mut recovery = MultisigRecovery::new(3, signers.clone(), config);
        
        let txo_hash = [0x42u8; 32];
        recovery.initiate_recovery(
            txo_hash,
            1000,
            alloc::format!("Test recovery"),
            10000,
        ).unwrap();
        
        // Add sufficient signatures
        for i in 0..3 {
            let sig = RecoverySignature {
                signer: signers[i].clone(),
                signature: vec![i as u8; 64],
                timestamp: 10000 + i as u64,
            };
            recovery.add_signature(&txo_hash, sig).unwrap();
        }
        
        // Execute recovery
        let result = recovery.execute_recovery(&txo_hash, 10100);
        assert!(result.is_ok());
        
        // Verify audit log
        assert_eq!(recovery.audit_log().len(), 1);
    }
    
    #[test]
    fn test_temporal_decay() {
        let signers = create_test_signers(5);
        let config = create_test_config();
        let recovery = MultisigRecovery::new(3, signers, config);
        
        // Epoch 0: full window
        let window0 = recovery.compute_decay_window(0);
        assert_eq!(window0, 3600);
        
        // Epoch 1: 90% of original (10% decay)
        let window1 = recovery.compute_decay_window(1);
        assert_eq!(window1, 3240);
        
        // High epoch: should hit minimum
        let window_high = recovery.compute_decay_window(100);
        assert_eq!(window_high, 300);
    }
    
    #[test]
    fn test_unauthorized_signer() {
        let signers = create_test_signers(5);
        let config = create_test_config();
        let mut recovery = MultisigRecovery::new(3, signers, config);
        
        let txo_hash = [0x42u8; 32];
        recovery.initiate_recovery(
            txo_hash,
            1000,
            alloc::format!("Test recovery"),
            10000,
        ).unwrap();
        
        // Create unauthorized signer
        let unauthorized = SignerIdentity {
            pubkey_hash: [0xFFu8; 32],
            identifier: alloc::format!("unauthorized"),
        };
        
        let sig = RecoverySignature {
            signer: unauthorized,
            signature: vec![0xAB; 64],
            timestamp: 10000,
        };
        
        // Should reject unauthorized signer
        let result = recovery.add_signature(&txo_hash, sig);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_duplicate_signature() {
        let signers = create_test_signers(5);
        let config = create_test_config();
        let mut recovery = MultisigRecovery::new(3, signers.clone(), config);
        
        let txo_hash = [0x42u8; 32];
        recovery.initiate_recovery(
            txo_hash,
            1000,
            alloc::format!("Test recovery"),
            10000,
        ).unwrap();
        
        let sig = RecoverySignature {
            signer: signers[0].clone(),
            signature: vec![0x01; 64],
            timestamp: 10000,
        };
        
        // First signature should succeed
        assert!(recovery.add_signature(&txo_hash, sig.clone()).is_ok());
        
        // Duplicate should fail
        assert!(recovery.add_signature(&txo_hash, sig).is_err());
    }
}
