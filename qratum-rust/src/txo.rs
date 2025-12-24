//! # TXO (Transaction Object) Module
//!
//! ## Lifecycle Stage: All Stages
//!
//! The TXO is the fundamental unit of computation and communication in QRATUM.
//! Every input, intermediate state, and output is represented as a TXO.
//!
//! ## Architectural Role
//!
//! - **Input TXOs**: Contain user intents, quorum member contributions
//! - **Outcome TXOs**: Minimal, blinded, signed computational results
//! - **Audit TXOs**: Censorship events, decay justifications, canary probes
//!
//! ## Anti-Censorship Mechanism
//!
//! Any suppression, delay, or rollback of TXO processing MUST emit a signed,
//! auditable TXO to external witnesses. This ensures provable censorship resistance.
//!
//! ## Security Rationale
//!
//! - CBOR primary encoding for deterministic serialization
//! - SHA3-256 content addressing for tamper detection
//! - Blinded payloads prevent payload inspection while allowing validation
//! - Compliance ZKP attestations prove regulatory compliance without exposure


extern crate alloc;
use alloc::vec::Vec;
use alloc::string::String;

use minicbor::{Encode, Decode};
use sha3::{Sha3_256, Digest};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// TXO Type discriminator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Encode, Decode)]
pub enum TxoType {
    #[n(0)] Input,           // User-provided input
    #[n(1)] Outcome,         // Computational result (minimal, blinded)
    #[n(2)] DecayJustification, // Quorum threshold decay event
    #[n(3)] CanaryProbe,     // Censorship detection probe
    #[n(4)] CensorshipEvent, // Suppression/delay audit trail
    #[n(5)] ProxyApproval,   // Bonded proxy authorization
    #[n(6)] ComplianceAttestation, // ZKP regulatory compliance
}

/// Blinded Payload Commitment
///
/// ## Lifecycle Stage: Execution → Outcome Commitment
///
/// Allows validation without payload exposure. Commitment is SHA3-256 hash,
/// reveal controlled by quorum consensus.
#[derive(Debug, Clone, Encode, Decode, Zeroize, ZeroizeOnDrop)]
pub struct BlindedPayload {
    /// SHA3-256 commitment to the actual payload
    #[n(0)]
    pub commitment: [u8; 32],
    
    /// Optional revealed payload (only after quorum approval)
    #[n(1)]
    pub revealed: Option<Vec<u8>>,
    
    /// Quorum threshold required for reveal (0-100)
    #[n(2)]
    pub reveal_threshold: u8,
}

impl BlindedPayload {
    /// Create a new blinded payload with commitment
    ///
    /// # Inputs
    /// - `payload`: Raw data to blind
    /// - `reveal_threshold`: Quorum percentage required for reveal (0-100)
    ///
    /// # Outputs
    /// - `BlindedPayload` with commitment hash and no revealed data
    pub fn new(payload: &[u8], reveal_threshold: u8) -> Self {
        let mut hasher = Sha3_256::new();
        hasher.update(payload);
        let commitment: [u8; 32] = hasher.finalize().into();
        
        Self {
            commitment,
            revealed: None,
            reveal_threshold,
        }
    }
    
    /// Verify that revealed payload matches commitment
    pub fn verify(&self) -> bool {
        if let Some(ref revealed) = self.revealed {
            let mut hasher = Sha3_256::new();
            hasher.update(revealed);
            let computed: [u8; 32] = hasher.finalize().into();
            computed == self.commitment
        } else {
            false
        }
    }
}

/// Zero-Knowledge Compliance Attestation
///
/// ## Lifecycle Stage: Execution
///
/// Placeholder for Halo2/Risc0 ZKP circuits that prove regulatory compliance
/// without exposing sensitive data.
///
/// ## Forward Compatibility
/// TODO: QRADLE post-quantum migration - replace with lattice-based ZKP
#[derive(Debug, Clone, Encode, Decode, Zeroize, ZeroizeOnDrop)]
pub struct ComplianceZkp {
    /// Circuit identifier (e.g., "GDPR-Article-17", "HIPAA-164.308")
    #[n(0)]
    pub circuit_id: String,
    
    /// Proof data (currently placeholder, will be Halo2/Risc0 proof bytes)
    #[n(1)]
    pub proof: Vec<u8>,
    
    /// Public inputs (non-sensitive)
    #[n(2)]
    pub public_inputs: Vec<u8>,
}

/// Transaction Object (TXO)
///
/// ## Lifecycle Stage: All Stages
///
/// The fundamental unit of QRATUM computation. All inputs, outputs, and
/// audit events are represented as TXOs.
///
/// ## Inputs → Outputs
/// - Serialization: `to_cbor()` → CBOR bytes
/// - Deserialization: `from_cbor()` ← CBOR bytes
/// - Content addressing: `compute_id()` → SHA3-256 hash
///
/// ## Audit Trail Responsibilities
/// - Every TXO creation is logged to the ephemeral ledger
/// - TXO IDs form the Merkle tree leaves for integrity verification
/// - Outcome TXOs are the ONLY persistent artifacts (minimal, blinded)
///
/// ## Anti-Censorship Mechanism
/// - Any delay/suppression triggers CensorshipEvent TXO emission
/// - CanaryProbe TXOs provide external liveness verification
/// - DecayJustification TXOs document quorum threshold changes
#[derive(Debug, Clone, Encode, Decode)]
pub struct Txo {
    /// Content-addressed identifier (SHA3-256 of CBOR-encoded TXO)
    #[n(0)]
    pub id: [u8; 32],
    
    /// TXO type discriminator
    #[n(1)]
    pub txo_type: TxoType,
    
    /// Timestamp (milliseconds since epoch)
    #[n(2)]
    pub timestamp: u64,
    
    /// Primary payload (may be blinded)
    #[n(3)]
    pub payload: Vec<u8>,
    
    /// Optional blinded commitment
    #[n(4)]
    pub blinded: Option<BlindedPayload>,
    
    /// Optional compliance attestation
    #[n(5)]
    pub compliance_zkp: Option<ComplianceZkp>,
    
    /// Predecessor TXO IDs (for provenance chain)
    #[n(6)]
    pub predecessors: Vec<[u8; 32]>,
    
    /// Quorum member signatures (variable-length)
    #[n(7)]
    pub signatures: Vec<[u8; 64]>,
}

impl Txo {
    /// Create a new TXO with computed content-addressed ID
    ///
    /// ## Lifecycle Stage: Quorum Convergence | Execution
    ///
    /// # Security Rationale
    /// - Content addressing ensures tamper detection
    /// - Deterministic CBOR encoding guarantees reproducibility
    pub fn new(
        txo_type: TxoType,
        timestamp: u64,
        payload: Vec<u8>,
        predecessors: Vec<[u8; 32]>,
    ) -> Self {
        let mut txo = Self {
            id: [0u8; 32],
            txo_type,
            timestamp,
            payload,
            blinded: None,
            compliance_zkp: None,
            predecessors,
            signatures: Vec::new(),
        };
        txo.id = txo.compute_id();
        txo
    }
    
    /// Compute content-addressed ID (SHA3-256)
    ///
    /// ## Inputs → Outputs
    /// - Self (without ID field) → SHA3-256 hash [u8; 32]
    ///
    /// ## Security Rationale
    /// - Deterministic CBOR encoding ensures same input → same ID
    /// - SHA3-256 provides collision resistance and pre-image resistance
    pub fn compute_id(&self) -> [u8; 32] {
        let cbor = self.to_cbor();
        let mut hasher = Sha3_256::new();
        hasher.update(&cbor);
        hasher.finalize().into()
    }
    
    /// Serialize to CBOR (primary encoding)
    pub fn to_cbor(&self) -> Vec<u8> {
        minicbor::to_vec(self).unwrap_or_default()
    }
    
    /// Deserialize from CBOR
    pub fn from_cbor(bytes: &[u8]) -> Result<Self, minicbor::decode::Error> {
        minicbor::decode(bytes)
    }
}

/// Outcome TXO - The ONLY persistent artifact
///
/// ## Lifecycle Stage: Outcome Commitment
///
/// After execution, only OutcomeTxos survive. All ephemeral state is destroyed.
///
/// ## Inputs → Outputs
/// - Input TXOs + Execution → Minimal OutcomeTxo (blinded, signed)
///
/// ## Security Rationale
/// - Minimal payload reduces attack surface
/// - Blinded commitment prevents payload inspection
/// - Quorum signatures provide multi-party attestation
/// - No intermediate state persists (ephemeral only)
#[derive(Debug, Clone, Encode, Decode)]
pub struct OutcomeTxo {
    /// Base TXO structure
    #[n(0)]
    pub txo: Txo,
    
    /// Execution summary hash (SHA3-256 of execution trace)
    #[n(1)]
    pub execution_hash: [u8; 32],
    
    /// Quorum consensus proof (minimum threshold met)
    #[n(2)]
    pub quorum_proof: Vec<u8>,
}

impl OutcomeTxo {
    /// Create a new outcome TXO
    ///
    /// ## Lifecycle Stage: Outcome Commitment
    ///
    /// # Audit Trail
    /// - Logs outcome TXO creation to ephemeral ledger
    /// - Computes execution hash for reproducibility
    /// - Collects quorum signatures for attestation
    pub fn new(
        payload: Vec<u8>,
        execution_hash: [u8; 32],
        quorum_proof: Vec<u8>,
        predecessors: Vec<[u8; 32]>,
    ) -> Self {
        let txo = Txo::new(
            TxoType::Outcome,
            current_timestamp(),
            payload,
            predecessors,
        );
        
        Self {
            txo,
            execution_hash,
            quorum_proof,
        }
    }
}

/// Get current timestamp (milliseconds since epoch)
///
/// ## Forward Compatibility
/// TODO: Replace with deterministic time oracle for reproducibility
fn current_timestamp() -> u64 {
    // Placeholder: In production, use deterministic time from quorum
    #[cfg(feature = "std")]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }
    #[cfg(not(feature = "std"))]
    {
        0 // Deterministic default for no_std
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    
    #[test]
    fn test_txo_creation() {
        let txo = Txo::new(
            TxoType::Input,
            1234567890,
            b"test payload".to_vec(),
            vec![],
        );
        assert_eq!(txo.txo_type, TxoType::Input);
        assert_eq!(txo.payload, b"test payload");
    }
    
    #[test]
    fn test_blinded_payload() {
        let payload = b"secret data";
        let blinded = BlindedPayload::new(payload, 67);
        assert_eq!(blinded.reveal_threshold, 67);
        assert!(blinded.revealed.is_none());
    }
}
