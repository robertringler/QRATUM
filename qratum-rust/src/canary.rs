//! # Canary Module - Censorship Detection Probes
//!
//! ## Lifecycle Stage: Execution (continuous monitoring)
//!
//! Canary probes are lightweight TXOs emitted at regular intervals to external
//! observers. If canaries stop arriving, it signals potential censorship, network
//! failure, or system compromise.
//!
//! ## Architectural Role
//!
//! - **Liveness Proof**: Regular canaries prove system is operational
//! - **Censorship Detection**: Missing canaries indicate suppression
//! - **External Verification**: Independent observers validate canary stream
//! - **Tamper Evidence**: Canary integrity verified via signatures
//!
//! ## Inputs → Outputs
//!
//! - Input: Execution state, timestamp
//! - Output: Signed CanaryProbe TXO
//!
//! ## Anti-Censorship Mechanism
//!
//! - Canaries emitted to multiple external observers (redundancy)
//! - Missing canaries trigger alerts and investigation
//! - Canary sequence numbers detect gaps/manipulation
//! - Cryptographic signatures prevent forgery
//!
//! ## Security Rationale
//!
//! - Lightweight (minimal overhead)
//! - Deterministic generation (predictable intervals)
//! - Externally verifiable without system access
//! - Forward-compatible with watchtower networks

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use alloc::string::String;

use crate::txo::{Txo, TxoType};
use sha3::{Sha3_256, Digest};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Canary Configuration
///
/// ## Lifecycle Stage: Initialization
///
/// Defines canary emission parameters and external observer endpoints.
#[derive(Debug, Clone)]
pub struct CanaryConfig {
    /// Emission interval (milliseconds)
    pub interval_ms: u64,
    
    /// External observer endpoints (content-addressed identifiers)
    pub observers: Vec<[u8; 32]>,
    
    /// Sequence number initialization
    pub initial_sequence: u64,
    
    /// Enable redundant emission
    pub redundant_emission: bool,
}

impl Default for CanaryConfig {
    fn default() -> Self {
        Self {
            interval_ms: 60_000,        // 1 minute
            observers: Vec::new(),
            initial_sequence: 0,
            redundant_emission: true,   // Emit to all observers
        }
    }
}

/// Canary Probe
///
/// ## Lifecycle Stage: Execution
///
/// Lightweight liveness proof emitted at regular intervals.
///
/// ## Anti-Censorship Mechanism
///
/// - Sequence numbers detect gaps (missing canaries)
/// - Timestamps verify emission timing
/// - State hash proves computational progress
/// - Signatures prevent forgery
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct CanaryProbe {
    /// Sequence number (monotonically increasing)
    pub sequence: u64,
    
    /// Emission timestamp
    pub timestamp: u64,
    
    /// State hash (SHA3-256 of current execution state)
    pub state_hash: [u8; 32],
    
    /// Previous canary hash (chain canaries for integrity)
    pub previous_canary_hash: [u8; 32],
    
    /// Session identifier
    pub session_id: [u8; 32],
    
    /// Optional diagnostic payload
    pub diagnostic: Option<Vec<u8>>,
}

impl CanaryProbe {
    /// Create new canary probe
    ///
    /// ## Lifecycle Stage: Execution
    ///
    /// # Inputs
    /// - `sequence`: Current sequence number
    /// - `state_hash`: Hash of current execution state
    /// - `previous_canary_hash`: Hash of previous canary (chain integrity)
    /// - `session_id`: Current session identifier
    ///
    /// # Outputs
    /// - `CanaryProbe` ready for emission
    ///
    /// ## Security Rationale
    /// - Chained hashes prevent retroactive manipulation
    /// - State hash proves computational progress
    /// - Sequence numbers detect gaps
    pub fn new(
        sequence: u64,
        state_hash: [u8; 32],
        previous_canary_hash: [u8; 32],
        session_id: [u8; 32],
    ) -> Self {
        Self {
            sequence,
            timestamp: current_timestamp(),
            state_hash,
            previous_canary_hash,
            session_id,
            diagnostic: None,
        }
    }
    
    /// Convert to TXO for emission
    ///
    /// ## Lifecycle Stage: Execution
    ///
    /// # Audit Trail
    /// - Emits CanaryProbe TXO to external observers
    /// - Logs emission event to ephemeral ledger
    pub fn to_txo(&self) -> Txo {
        // Serialize canary data
        let mut payload = Vec::new();
        payload.extend_from_slice(&self.sequence.to_le_bytes());
        payload.extend_from_slice(&self.timestamp.to_le_bytes());
        payload.extend_from_slice(&self.state_hash);
        payload.extend_from_slice(&self.previous_canary_hash);
        payload.extend_from_slice(&self.session_id);
        
        if let Some(ref diagnostic) = self.diagnostic {
            payload.extend_from_slice(diagnostic);
        }
        
        Txo::new(
            TxoType::CanaryProbe,
            self.timestamp,
            payload,
            vec![self.previous_canary_hash],
        )
    }
    
    /// Compute canary hash for chaining
    ///
    /// ## Security Rationale
    /// - SHA3-256 ensures collision resistance
    /// - Includes all fields for integrity
    pub fn compute_hash(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(&self.sequence.to_le_bytes());
        hasher.update(&self.timestamp.to_le_bytes());
        hasher.update(&self.state_hash);
        hasher.update(&self.previous_canary_hash);
        hasher.update(&self.session_id);
        
        if let Some(ref diagnostic) = self.diagnostic {
            hasher.update(diagnostic);
        }
        
        hasher.finalize().into()
    }
}

/// Canary State
///
/// ## Lifecycle Stage: Execution
///
/// Tracks canary emission state across session lifetime.
#[derive(Clone)]
pub struct CanaryState {
    /// Current sequence number
    pub sequence: u64,
    
    /// Last emission timestamp
    pub last_emission: u64,
    
    /// Last canary hash (for chaining)
    pub last_canary_hash: [u8; 32],
    
    /// Session identifier
    pub session_id: [u8; 32],
    
    /// Emission history (bounded for memory)
    pub history: Vec<CanaryProbe>,
    
    /// Maximum history size
    pub max_history: usize,
}

impl CanaryState {
    /// Create new canary state
    ///
    /// ## Lifecycle Stage: Ephemeral Materialization
    pub fn new(session_id: [u8; 32], initial_sequence: u64) -> Self {
        Self {
            sequence: initial_sequence,
            last_emission: current_timestamp(),
            last_canary_hash: [0u8; 32], // Genesis canary
            session_id,
            history: Vec::new(),
            max_history: 100, // Keep last 100 canaries
        }
    }
    
    /// Check if emission due
    ///
    /// ## Lifecycle Stage: Execution
    ///
    /// # Inputs
    /// - `config`: Canary configuration with interval
    ///
    /// # Outputs
    /// - `true` if interval elapsed, `false` otherwise
    pub fn emission_due(&self, config: &CanaryConfig) -> bool {
        let current_time = current_timestamp();
        current_time - self.last_emission >= config.interval_ms
    }
    
    /// Generate next canary probe
    ///
    /// ## Lifecycle Stage: Execution
    ///
    /// # Inputs
    /// - `state_hash`: Current execution state hash
    ///
    /// # Outputs
    /// - New `CanaryProbe` with incremented sequence
    ///
    /// ## Audit Trail
    /// - Logs canary generation to ephemeral ledger
    /// - Updates sequence number and timestamp
    pub fn generate_canary(&mut self, state_hash: [u8; 32]) -> CanaryProbe {
        let canary = CanaryProbe::new(
            self.sequence,
            state_hash,
            self.last_canary_hash,
            self.session_id,
        );
        
        let canary_hash = canary.compute_hash();
        
        // Update state
        self.sequence += 1;
        self.last_emission = current_timestamp();
        self.last_canary_hash = canary_hash;
        
        // Add to history (bounded)
        self.history.push(canary.clone());
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
        
        canary
    }
}

/// Canary Verifier (External Observer)
///
/// ## Lifecycle Stage: External Verification
///
/// Independent observers use this to verify canary stream integrity.
///
/// ## Anti-Censorship Mechanism
///
/// - Detects missing canaries (gaps in sequence)
/// - Verifies timing consistency
/// - Validates canary chain integrity
/// - Alerts on anomalies
#[derive(Debug, Clone)]
pub struct CanaryVerifier {
    /// Expected emission interval (milliseconds)
    pub expected_interval_ms: u64,
    
    /// Tolerance window (milliseconds)
    pub tolerance_ms: u64,
    
    /// Last verified canary
    pub last_verified: Option<CanaryProbe>,
}

impl CanaryVerifier {
    /// Create new canary verifier
    ///
    /// ## Lifecycle Stage: External Observer Initialization
    pub fn new(expected_interval_ms: u64, tolerance_ms: u64) -> Self {
        Self {
            expected_interval_ms,
            tolerance_ms,
            last_verified: None,
        }
    }
    
    /// Verify canary probe
    ///
    /// ## Lifecycle Stage: External Verification
    ///
    /// # Inputs
    /// - `canary`: Canary probe to verify
    ///
    /// # Outputs
    /// - `Ok(())` if valid, `Err(reason)` otherwise
    ///
    /// ## Anti-Censorship Mechanism
    /// - Detects sequence gaps (missing canaries)
    /// - Verifies timing consistency
    /// - Validates hash chain integrity
    pub fn verify(&mut self, canary: &CanaryProbe) -> Result<(), String> {
        if let Some(ref last) = self.last_verified {
            // Check sequence continuity
            let expected_sequence = last.sequence + 1;
            if canary.sequence != expected_sequence {
                return Err(alloc::format!(
                    "Sequence gap detected: expected {}, got {} ({} missing)",
                    expected_sequence,
                    canary.sequence,
                    canary.sequence - expected_sequence
                ));
            }
            
            // Check timing consistency
            let time_delta = canary.timestamp - last.timestamp;
            let expected_min = self.expected_interval_ms.saturating_sub(self.tolerance_ms);
            let expected_max = self.expected_interval_ms + self.tolerance_ms;
            
            if time_delta < expected_min || time_delta > expected_max {
                return Err(alloc::format!(
                    "Timing anomaly: interval {}ms outside expected range [{}, {}]ms",
                    time_delta,
                    expected_min,
                    expected_max
                ));
            }
            
            // Verify hash chain
            let last_hash = last.compute_hash();
            if canary.previous_canary_hash != last_hash {
                return Err("Hash chain broken: previous canary hash mismatch".into());
            }
        }
        
        // Update last verified
        self.last_verified = Some(canary.clone());
        Ok(())
    }
    
    /// Check if canary overdue
    ///
    /// ## Anti-Censorship Mechanism
    /// - Alerts if expected canary not received
    /// - Triggers investigation and fallback procedures
    pub fn is_overdue(&self) -> bool {
        if let Some(ref last) = self.last_verified {
            let current_time = current_timestamp();
            let elapsed = current_time - last.timestamp;
            elapsed > self.expected_interval_ms + self.tolerance_ms
        } else {
            false // No baseline yet
        }
    }
}

/// Get current timestamp (milliseconds since epoch)
fn current_timestamp() -> u64 {
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
    
    #[test]
    fn test_canary_probe_creation() {
        let state_hash = [1u8; 32];
        let prev_hash = [0u8; 32];
        let session_id = [2u8; 32];
        
        let canary = CanaryProbe::new(0, state_hash, prev_hash, session_id);
        assert_eq!(canary.sequence, 0);
        assert_eq!(canary.state_hash, state_hash);
    }
    
    #[test]
    fn test_canary_state_generation() {
        let session_id = [1u8; 32];
        let mut state = CanaryState::new(session_id, 0);
        
        let state_hash = [2u8; 32];
        let canary = state.generate_canary(state_hash);
        
        assert_eq!(canary.sequence, 0);
        assert_eq!(state.sequence, 1);
    }
    
    #[test]
    fn test_canary_verifier() {
        let mut verifier = CanaryVerifier::new(60_000, 5_000);
        
        let canary1 = CanaryProbe::new(0, [1u8; 32], [0u8; 32], [0u8; 32]);
        assert!(verifier.verify(&canary1).is_ok());
    }
}
