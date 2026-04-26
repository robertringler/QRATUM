//! RTF (Reversible Transaction Framework) API
//!
//! Provides execute_txo, commit_txo, and rollback_txo primitives
//! with zone enforcement (Z0-Z3) and dual-control validation.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use alloc::string::String;
use core::result::Result;

use crate::txo::{TXO, OperationClass, IdentityType};
use crate::ledger::MerkleLedger;

/// Zone identifier (Z0-Z3)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Zone {
    Z0,  // Genesis - immutable
    Z1,  // Staging - development
    Z2,  // Production - validated
    Z3,  // Archive - air-gapped
}

/// RTF Error types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RTFError {
    /// Zone policy violation
    ZonePolicyViolation,
    /// Missing required signature
    MissingSignature,
    /// Invalid signature
    InvalidSignature,
    /// Dual control requirement not met
    DualControlFailure,
    /// Non-reversible TXO cannot be rolled back
    NonReversible,
    /// Epoch not found
    EpochNotFound,
    /// Invalid zone transition
    InvalidZoneTransition,
    /// Operation not allowed in current zone
    OperationNotAllowed,
}

/// RTF execution context
pub struct RTFContext {
    /// Current zone
    pub current_zone: Zone,
    /// Merkle ledger reference
    pub ledger: MerkleLedger,
    /// Current epoch
    pub current_epoch: u64,
}

impl RTFContext {
    /// Create a new RTF context
    pub fn new(zone: Zone, ledger: MerkleLedger) -> Self {
        Self {
            current_zone: zone,
            ledger,
            current_epoch: 0,
        }
    }
    
    /// Execute a TXO - validate and prepare for commit
    ///
    /// # Arguments
    /// * `txo` - Transaction object to execute
    ///
    /// # Returns
    /// * `Ok(())` if execution succeeds
    /// * `Err(RTFError)` if validation fails
    pub fn execute_txo(&mut self, txo: &mut TXO) -> Result<(), RTFError> {
        // Validate zone policy
        self.validate_zone_policy(txo)?;
        
        // Validate signatures
        self.validate_signatures(txo)?;
        
        // Check dual control if required
        if txo.dual_control_required && !txo.verify_dual_control() {
            return Err(RTFError::DualControlFailure);
        }
        
        // Set epoch from current context
        txo.epoch_id = self.current_epoch;
        
        // Add audit entry for execution
        let audit_entry = crate::txo::AuditEntry {
            actor_id: txo.sender.id,
            action: String::from("EXECUTE"),
            timestamp: txo.timestamp,
        };
        txo.add_audit_entry(audit_entry);
        
        Ok(())
    }
    
    /// Commit a TXO to the ledger
    ///
    /// # Arguments
    /// * `txo` - Transaction object to commit
    ///
    /// # Returns
    /// * `Ok(())` if commit succeeds
    /// * `Err(RTFError)` if commit fails
    pub fn commit_txo(&mut self, txo: &mut TXO) -> Result<(), RTFError> {
        // Add to ledger
        self.ledger.append_txo(txo, self.current_zone);
        
        // Add audit entry for commit
        let audit_entry = crate::txo::AuditEntry {
            actor_id: txo.sender.id,
            action: String::from("COMMIT"),
            timestamp: txo.timestamp,
        };
        txo.add_audit_entry(audit_entry);
        
        Ok(())
    }
    
    /// Rollback to a previous epoch
    ///
    /// # Arguments
    /// * `target_epoch` - Epoch to rollback to
    /// * `reason` - Human-readable rollback reason
    ///
    /// # Returns
    /// * `Ok(())` if rollback succeeds
    /// * `Err(RTFError)` if rollback fails
    pub fn rollback_txo(&mut self, target_epoch: u64, reason: String) -> Result<(), RTFError> {
        // Validate zone allows rollback
        if !self.zone_allows_rollback() {
            return Err(RTFError::NonReversible);
        }
        
        // Validate target epoch exists
        if target_epoch > self.current_epoch {
            return Err(RTFError::EpochNotFound);
        }
        
        // Perform rollback on ledger
        self.ledger.rollback_to_epoch(target_epoch)?;
        
        // Update current epoch
        self.current_epoch = target_epoch;
        
        Ok(())
    }
    
    /// Validate zone policy for TXO
    fn validate_zone_policy(&self, txo: &TXO) -> Result<(), RTFError> {
        match self.current_zone {
            Zone::Z0 => {
                // Genesis zone - only genesis operations
                if txo.operation_class != OperationClass::Admin {
                    return Err(RTFError::OperationNotAllowed);
                }
            }
            Zone::Z1 => {
                // Staging zone - all operations allowed
                // No signature requirements
            }
            Zone::Z2 => {
                // Production zone - most operations allowed
                // Single signature required
                if txo.signatures.is_empty() {
                    return Err(RTFError::MissingSignature);
                }
                // Admin operations not allowed
                if txo.operation_class == OperationClass::Admin {
                    return Err(RTFError::OperationNotAllowed);
                }
            }
            Zone::Z3 => {
                // Archive zone - only audit operations
                // Dual signatures required
                if txo.signatures.len() < 2 {
                    return Err(RTFError::DualControlFailure);
                }
                if txo.operation_class != OperationClass::Compliance {
                    return Err(RTFError::OperationNotAllowed);
                }
            }
        }
        
        Ok(())
    }
    
    /// Validate signatures on TXO
    fn validate_signatures(&self, txo: &TXO) -> Result<(), RTFError> {
        // In production, this would verify Ed25519 signatures
        // For now, just check that required signatures exist
        
        match self.current_zone {
            Zone::Z0 | Zone::Z1 => {
                // No signature requirements
                Ok(())
            }
            Zone::Z2 => {
                // Single signature required
                if txo.signatures.is_empty() {
                    return Err(RTFError::MissingSignature);
                }
                Ok(())
            }
            Zone::Z3 => {
                // Dual signatures required
                if txo.signatures.len() < 2 {
                    return Err(RTFError::DualControlFailure);
                }
                Ok(())
            }
        }
    }
    
    /// Check if current zone allows rollback
    fn zone_allows_rollback(&self) -> bool {
        match self.current_zone {
            Zone::Z0 => false,  // Genesis is immutable
            Zone::Z1 => true,   // Staging allows rollback
            Zone::Z2 => true,   // Production allows emergency rollback
            Zone::Z3 => false,  // Archive is immutable
        }
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
        
        // Promote ledger
        self.ledger.promote_zone(target_zone)?;
        
        // Update current zone
        self.current_zone = target_zone;
        
        // Increment epoch on promotion
        self.current_epoch += 1;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::txo::{Sender, Receiver, Payload, PayloadType, Signature, SignatureType};
    
    #[test]
    fn test_execute_txo_z1() {
        let ledger = MerkleLedger::new([0u8; 32]);
        let mut ctx = RTFContext::new(Zone::Z1, ledger);
        
        let sender = Sender {
            identity_type: IdentityType::Operator,
            id: [1u8; 16],
            biokey_present: false,
            fido2_signed: false,
            zk_proof: None,
        };
        
        let receiver = Receiver {
            identity_type: IdentityType::Node,
            id: [2u8; 16],
        };
        
        let payload = Payload {
            payload_type: PayloadType::Genome,
            content_hash: [3u8; 32],
            encrypted: true,
        };
        
        let mut txo = TXO::new(
            [4u8; 16],
            sender,
            receiver,
            OperationClass::Genomic,
            payload,
        );
        
        // Should succeed in Z1 without signatures
        assert!(ctx.execute_txo(&mut txo).is_ok());
    }
    
    #[test]
    fn test_execute_txo_z2_requires_signature() {
        let ledger = MerkleLedger::new([0u8; 32]);
        let mut ctx = RTFContext::new(Zone::Z2, ledger);
        
        let sender = Sender {
            identity_type: IdentityType::Operator,
            id: [1u8; 16],
            biokey_present: false,
            fido2_signed: false,
            zk_proof: None,
        };
        
        let receiver = Receiver {
            identity_type: IdentityType::Node,
            id: [2u8; 16],
        };
        
        let payload = Payload {
            payload_type: PayloadType::Genome,
            content_hash: [3u8; 32],
            encrypted: true,
        };
        
        let mut txo = TXO::new(
            [4u8; 16],
            sender,
            receiver,
            OperationClass::Genomic,
            payload,
        );
        
        // Should fail in Z2 without signature
        assert_eq!(ctx.execute_txo(&mut txo), Err(RTFError::MissingSignature));
        
        // Add signature
        txo.add_signature(Signature {
            sig_type: SignatureType::Fido2,
            signer_id: [5u8; 16],
            signature: vec![0u8; 64],
        });
        
        // Should succeed with signature
        assert!(ctx.execute_txo(&mut txo).is_ok());
    }
    
    #[test]
    fn test_zone_promotion() {
        let ledger = MerkleLedger::new([0u8; 32]);
        let mut ctx = RTFContext::new(Zone::Z0, ledger);
        
        // Z0 -> Z1 should succeed
        assert!(ctx.promote_zone(Zone::Z1).is_ok());
        assert_eq!(ctx.current_zone, Zone::Z1);
        
        // Z1 -> Z2 should succeed
        assert!(ctx.promote_zone(Zone::Z2).is_ok());
        assert_eq!(ctx.current_zone, Zone::Z2);
        
        // Z2 -> Z1 should fail (backwards)
        assert_eq!(ctx.promote_zone(Zone::Z1), Err(RTFError::InvalidZoneTransition));
    }
    
    #[test]
    fn test_rollback_in_z1() {
        let ledger = MerkleLedger::new([0u8; 32]);
        let mut ctx = RTFContext::new(Zone::Z1, ledger);
        
        ctx.current_epoch = 5;
        
        // Rollback should succeed in Z1
        assert!(ctx.rollback_txo(3, String::from("Test rollback")).is_ok());
        assert_eq!(ctx.current_epoch, 3);
    }
    
    #[test]
    fn test_rollback_in_z0_fails() {
        let ledger = MerkleLedger::new([0u8; 32]);
        let mut ctx = RTFContext::new(Zone::Z0, ledger);
        
        ctx.current_epoch = 5;
        
        // Rollback should fail in Z0 (immutable)
        assert_eq!(
            ctx.rollback_txo(3, String::from("Test rollback")),
            Err(RTFError::NonReversible)
        );
    }
}
