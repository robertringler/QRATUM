//! TXO (Transaction Object) Implementation
//!
//! Core data structure for Aethernet overlay network transactions.
//! Supports CBOR-primary encoding with JSON-secondary, dual-control signatures,
//! and zone-aware reversibility.

#![no_std]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use minicbor::{Decode, Encode};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};

/// Identity type for sender/receiver
#[derive(Debug, Clone, Copy, Encode, Decode, Serialize, Deserialize, PartialEq)]
#[cbor(index_only)]
pub enum IdentityType {
    #[n(0)] Operator,
    #[n(1)] Node,
    #[n(2)] System,
}

/// Operation class for TXO
#[derive(Debug, Clone, Copy, Encode, Decode, Serialize, Deserialize, PartialEq)]
#[cbor(index_only)]
pub enum OperationClass {
    #[n(0)] Genomic,
    #[n(1)] Network,
    #[n(2)] Compliance,
    #[n(3)] Admin,
}

/// Payload type
#[derive(Debug, Clone, Copy, Encode, Decode, Serialize, Deserialize, PartialEq)]
#[cbor(index_only)]
pub enum PayloadType {
    #[n(0)] Genome,
    #[n(1)] Metadata,
    #[n(2)] Control,
    #[n(3)] Audit,
}

/// Signature type
#[derive(Debug, Clone, Copy, Encode, Decode, Serialize, Deserialize, PartialEq)]
#[cbor(index_only)]
pub enum SignatureType {
    #[n(0)] Fido2,
    #[n(1)] Biokey,
}

/// Sender identity with biokey support
#[derive(Debug, Clone, Encode, Decode, Serialize, Deserialize)]
pub struct Sender {
    /// Type of sender identity
    #[n(0)]
    pub identity_type: IdentityType,
    
    /// Sender UUID (128-bit)
    #[n(1)]
    pub id: [u8; 16],
    
    /// Whether biometric key is present
    #[n(2)]
    pub biokey_present: bool,
    
    /// Whether FIDO2 signature is present
    #[n(3)]
    pub fido2_signed: bool,
    
    /// Optional zero-knowledge proof for biokey (ephemeral SNP-loci)
    #[n(4)]
    pub zk_proof: Option<Vec<u8>>,
}

/// Receiver identity
#[derive(Debug, Clone, Encode, Decode, Serialize, Deserialize)]
pub struct Receiver {
    /// Type of receiver identity
    #[n(0)]
    pub identity_type: IdentityType,
    
    /// Receiver UUID (128-bit)
    #[n(1)]
    pub id: [u8; 16],
}

/// Payload structure
#[derive(Debug, Clone, Encode, Decode, Serialize, Deserialize)]
pub struct Payload {
    /// Payload content type
    #[n(0)]
    pub payload_type: PayloadType,
    
    /// SHA3-256 hash of payload content
    #[n(1)]
    pub content_hash: [u8; 32],
    
    /// Encryption status
    #[n(2)]
    pub encrypted: bool,
}

/// Cryptographic signature
#[derive(Debug, Clone, Encode, Decode, Serialize, Deserialize)]
pub struct Signature {
    /// Signature type (FIDO2 or Biokey)
    #[n(0)]
    pub sig_type: SignatureType,
    
    /// Signer UUID (128-bit)
    #[n(1)]
    pub signer_id: [u8; 16],
    
    /// Signature bytes (64 bytes for Ed25519)
    #[n(2)]
    pub signature: Vec<u8>,
}

/// Rollback history entry
#[derive(Debug, Clone, Encode, Decode, Serialize, Deserialize)]
pub struct RollbackEntry {
    /// Source epoch
    #[n(0)]
    pub from_epoch: u64,
    
    /// Target epoch for rollback
    #[n(1)]
    pub to_epoch: u64,
    
    /// Human-readable rollback reason
    #[n(2)]
    pub reason: String,
}

/// Audit trail entry
#[derive(Debug, Clone, Encode, Decode, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Actor UUID (128-bit)
    #[n(0)]
    pub actor_id: [u8; 16],
    
    /// Action description
    #[n(1)]
    pub action: String,
    
    /// Unix timestamp (seconds since epoch)
    #[n(2)]
    pub timestamp: u64,
}

/// Transaction Object (TXO) - Core Aethernet data structure
#[derive(Debug, Clone, Encode, Decode, Serialize, Deserialize)]
pub struct TXO {
    /// Schema version
    #[n(0)]
    pub version: u32,
    
    /// Unique transaction identifier (UUID v4, 128-bit)
    #[n(1)]
    pub txo_id: [u8; 16],
    
    /// Unix timestamp (seconds since epoch)
    #[n(2)]
    pub timestamp: u64,
    
    /// Ledger snapshot epoch
    #[n(3)]
    pub epoch_id: u64,
    
    /// SHA3-256 hash of execution container
    #[n(4)]
    pub container_hash: [u8; 32],
    
    /// Sender identity
    #[n(5)]
    pub sender: Sender,
    
    /// Receiver identity
    #[n(6)]
    pub receiver: Receiver,
    
    /// Operation classification
    #[n(7)]
    pub operation_class: OperationClass,
    
    /// Reversibility flag
    #[n(8)]
    pub reversibility_flag: bool,
    
    /// Payload
    #[n(9)]
    pub payload: Payload,
    
    /// Dual control requirement
    #[n(10)]
    pub dual_control_required: bool,
    
    /// Cryptographic signatures
    #[n(11)]
    pub signatures: Vec<Signature>,
    
    /// Rollback history
    #[n(12)]
    pub rollback_history: Vec<RollbackEntry>,
    
    /// Audit trail
    #[n(13)]
    pub audit_trail: Vec<AuditEntry>,
}

impl TXO {
    /// Create a new TXO with default values
    pub fn new(
        txo_id: [u8; 16],
        sender: Sender,
        receiver: Receiver,
        operation_class: OperationClass,
        payload: Payload,
    ) -> Self {
        Self {
            version: 1,
            txo_id,
            timestamp: 0, // Set by RTF layer
            epoch_id: 0,  // Set by ledger
            container_hash: [0u8; 32], // Set by RTF layer
            sender,
            receiver,
            operation_class,
            reversibility_flag: true,
            payload,
            dual_control_required: false,
            signatures: Vec::new(),
            rollback_history: Vec::new(),
            audit_trail: Vec::new(),
        }
    }
    
    /// Compute SHA3-256 hash of TXO content (merkle chaining)
    pub fn compute_hash(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        
        // Serialize to CBOR for deterministic hashing
        let mut cbor_buffer = Vec::new();
        let mut encoder = minicbor::Encoder::new(&mut cbor_buffer);
        if self.encode(&mut encoder, &mut ()).is_ok() {
            hasher.update(&cbor_buffer);
        }
        
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }
    
    /// Add a signature to the TXO
    pub fn add_signature(&mut self, signature: Signature) {
        self.signatures.push(signature);
    }
    
    /// Add an audit entry
    pub fn add_audit_entry(&mut self, entry: AuditEntry) {
        self.audit_trail.push(entry);
    }
    
    /// Add a rollback entry
    pub fn add_rollback_entry(&mut self, entry: RollbackEntry) {
        self.rollback_history.push(entry);
    }
    
    /// Verify dual control (requires at least 2 signatures)
    pub fn verify_dual_control(&self) -> bool {
        if !self.dual_control_required {
            return true;
        }
        self.signatures.len() >= 2
    }
    
    /// Serialize to CBOR (primary encoding)
    pub fn to_cbor(&self) -> Result<Vec<u8>, minicbor::encode::Error<core::convert::Infallible>> {
        let mut buffer = Vec::new();
        let mut encoder = minicbor::Encoder::new(&mut buffer);
        self.encode(&mut encoder, &mut ())?;
        Ok(buffer)
    }
    
    /// Deserialize from CBOR
    pub fn from_cbor(data: &[u8]) -> Result<Self, minicbor::decode::Error> {
        minicbor::decode(data)
    }
}

impl fmt::Display for TXO {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TXO(id={:?}, epoch={}, class={:?}, reversible={})",
            &self.txo_id[..4],
            self.epoch_id,
            self.operation_class,
            self.reversibility_flag
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_txo_creation() {
        let sender = Sender {
            identity_type: IdentityType::Operator,
            id: [1u8; 16],
            biokey_present: false,
            fido2_signed: true,
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
        
        let txo = TXO::new(
            [0u8; 16],
            sender,
            receiver,
            OperationClass::Genomic,
            payload,
        );
        
        assert_eq!(txo.version, 1);
        assert_eq!(txo.operation_class, OperationClass::Genomic);
    }
    
    #[test]
    fn test_txo_cbor_roundtrip() {
        let sender = Sender {
            identity_type: IdentityType::System,
            id: [5u8; 16],
            biokey_present: false,
            fido2_signed: false,
            zk_proof: None,
        };
        
        let receiver = Receiver {
            identity_type: IdentityType::Node,
            id: [6u8; 16],
        };
        
        let payload = Payload {
            payload_type: PayloadType::Metadata,
            content_hash: [7u8; 32],
            encrypted: false,
        };
        
        let txo = TXO::new(
            [8u8; 16],
            sender,
            receiver,
            OperationClass::Network,
            payload,
        );
        
        // Serialize to CBOR
        let cbor_data = txo.to_cbor().unwrap();
        
        // Deserialize from CBOR
        let txo_decoded = TXO::from_cbor(&cbor_data).unwrap();
        
        assert_eq!(txo.txo_id, txo_decoded.txo_id);
        assert_eq!(txo.operation_class, txo_decoded.operation_class);
    }
    
    #[test]
    fn test_dual_control() {
        let sender = Sender {
            identity_type: IdentityType::Operator,
            id: [1u8; 16],
            biokey_present: false,
            fido2_signed: false,
            zk_proof: None,
        };
        
        let receiver = Receiver {
            identity_type: IdentityType::System,
            id: [2u8; 16],
        };
        
        let payload = Payload {
            payload_type: PayloadType::Control,
            content_hash: [3u8; 32],
            encrypted: true,
        };
        
        let mut txo = TXO::new(
            [4u8; 16],
            sender,
            receiver,
            OperationClass::Admin,
            payload,
        );
        
        txo.dual_control_required = true;
        
        // Should fail with no signatures
        assert!(!txo.verify_dual_control());
        
        // Add first signature
        txo.add_signature(Signature {
            sig_type: SignatureType::Fido2,
            signer_id: [5u8; 16],
            signature: vec![0u8; 64],
        });
        
        // Should still fail with only one signature
        assert!(!txo.verify_dual_control());
        
        // Add second signature
        txo.add_signature(Signature {
            sig_type: SignatureType::Fido2,
            signer_id: [6u8; 16],
            signature: vec![0u8; 64],
        });
        
        // Should pass with two signatures
        assert!(txo.verify_dual_control());
    }
}
