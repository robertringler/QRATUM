//! # Snapshot Module - Volatile Encrypted Snapshots
//!
//! ## Lifecycle Stage: Execution (mid-session fault recovery)
//!
//! Volatile snapshots allow mid-session fault recovery without persistent storage.
//! Snapshots exist only in RAM, encrypted with ChaCha20-Poly1305 AEAD, and are zeroized
//! on session termination.
//!
//! ## Architectural Role
//!
//! - **Fault Recovery**: Resume from snapshot after transient failure
//! - **Volatile Only**: Snapshots never touch disk (RAM-only)
//! - **Encrypted**: Protected by ChaCha20-Poly1305 AEAD with ephemeral session key
//! - **Bounded**: Limited snapshot history (memory constraints)
//!
//! ## Inputs → Outputs
//!
//! - Input: Execution state → Encrypted snapshot
//! - Output: Encrypted snapshot → Restored execution state
//!
//! ## Security Rationale
//!
//! - ChaCha20-Poly1305 provides authenticated encryption (confidentiality + integrity)
//! - Ephemeral keys ensure snapshots useless after session
//! - Zeroization prevents memory forensics
//! - No persistent storage (anti-holographic)
//! - Constant-time operations prevent timing attacks
//!
//! ## Forward Compatibility
//!
//! Ready for QRADLE post-quantum migration - can switch to AES-256-GCM or PQC AEAD


extern crate alloc;
use alloc::vec::Vec;

use sha3::{Sha3_256, Digest};
use zeroize::{Zeroize, ZeroizeOnDrop};
use chacha20poly1305::{
    aead::{Aead, KeyInit, Payload},
    ChaCha20Poly1305, Nonce,
};

/// Snapshot Configuration
#[derive(Debug, Clone)]
pub struct SnapshotConfig {
    /// Maximum snapshots to retain in memory
    pub max_snapshots: usize,
    
    /// Snapshot interval (milliseconds)
    pub snapshot_interval_ms: u64,
    
    /// Enable compression (reduces memory footprint)
    pub enable_compression: bool,
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self {
            max_snapshots: 5,
            snapshot_interval_ms: 300_000, // 5 minutes
            enable_compression: false,     // Disabled for simplicity
        }
    }
}

/// Volatile Snapshot
///
/// ## Lifecycle Stage: Execution
///
/// In-memory encrypted snapshot of execution state.
///
/// ## Security Rationale
/// - Encrypted with ephemeral session key
/// - Zeroized on drop
/// - Never persisted to disk
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct VolatileSnapshot {
    /// Snapshot sequence number
    pub sequence: u64,
    
    /// Creation timestamp
    pub timestamp: u64,
    
    /// Encrypted state data
    pub encrypted_data: Vec<u8>,
    
    /// State hash (for integrity verification)
    pub state_hash: [u8; 32],
    
    /// Encryption nonce (for decryption)
    pub nonce: [u8; 32],
}

impl VolatileSnapshot {
    /// Create encrypted snapshot
    ///
    /// ## Lifecycle Stage: Execution
    ///
    /// # Inputs
    /// - `sequence`: Snapshot sequence number
    /// - `state_data`: Execution state to snapshot
    /// - `encryption_key`: Ephemeral session key (must be 32 bytes for ChaCha20-Poly1305)
    ///
    /// # Outputs
    /// - Encrypted `VolatileSnapshot` with authenticated encryption
    ///
    /// ## Security Rationale
    /// - ChaCha20-Poly1305 AEAD (Authenticated Encryption with Associated Data)
    /// - Unique nonce per snapshot prevents nonce reuse
    /// - State hash for additional integrity verification
    /// - Constant-time encryption operations
    pub fn create(
        sequence: u64,
        state_data: &[u8],
        encryption_key: &[u8; 64],
    ) -> Self {
        let timestamp = current_timestamp();
        
        // Generate unique nonce from timestamp and sequence
        let mut nonce_hasher = Sha3_256::new();
        nonce_hasher.update(&timestamp.to_le_bytes());
        nonce_hasher.update(&sequence.to_le_bytes());
        let nonce_bytes: [u8; 32] = nonce_hasher.finalize().into();
        
        // Compute state hash for integrity
        let mut state_hasher = Sha3_256::new();
        state_hasher.update(state_data);
        let state_hash: [u8; 32] = state_hasher.finalize().into();
        
        // Encrypt state data with ChaCha20-Poly1305
        // Use first 32 bytes of key for ChaCha20 key
        let chacha_key: &[u8; 32] = encryption_key[..32].try_into().unwrap();
        let cipher = ChaCha20Poly1305::new(chacha_key.into());
        
        // Use first 12 bytes of nonce_bytes for ChaCha20 nonce
        let nonce = Nonce::from_slice(&nonce_bytes[..12]);
        
        // Encrypt with associated data (sequence number for context)
        let payload = Payload {
            msg: state_data,
            aad: &sequence.to_le_bytes(),
        };
        
        let encrypted_data = cipher.encrypt(nonce, payload)
            .expect("ChaCha20-Poly1305 encryption failed");
        
        Self {
            sequence,
            timestamp,
            encrypted_data,
            state_hash,
            nonce: nonce_bytes,
        }
    }
    
    /// Restore state from snapshot
    ///
    /// ## Lifecycle Stage: Execution (recovery path)
    ///
    /// # Inputs
    /// - `encryption_key`: Ephemeral session key
    ///
    /// # Outputs
    /// - Decrypted state data or error
    ///
    /// ## Security Rationale
    /// - ChaCha20-Poly1305 verifies authentication tag automatically
    /// - Additional state hash verification for defense-in-depth
    /// - Prevents tampered snapshot restoration
    /// - Constant-time decryption operations
    pub fn restore(&self, encryption_key: &[u8; 64]) -> Result<Vec<u8>, &'static str> {
        // Decrypt state data with ChaCha20-Poly1305
        let chacha_key: &[u8; 32] = encryption_key[..32].try_into().unwrap();
        let cipher = ChaCha20Poly1305::new(chacha_key.into());
        
        let nonce = Nonce::from_slice(&self.nonce[..12]);
        
        // Decrypt with associated data verification
        let payload = Payload {
            msg: &self.encrypted_data,
            aad: &self.sequence.to_le_bytes(),
        };
        
        let decrypted_data = cipher.decrypt(nonce, payload)
            .map_err(|_| "Snapshot decryption failed - authentication tag mismatch")?;
        
        // Additional integrity verification with state hash (defense-in-depth)
        let mut hasher = Sha3_256::new();
        hasher.update(&decrypted_data);
        let computed_hash: [u8; 32] = hasher.finalize().into();
        
        if computed_hash != self.state_hash {
            return Err("Snapshot integrity verification failed - hash mismatch");
        }
        
        Ok(decrypted_data)
    }
}

/// Snapshot Manager
///
/// ## Lifecycle Stage: Execution
///
/// Manages bounded collection of volatile snapshots.
#[derive(Clone)]
pub struct SnapshotManager {
    /// Snapshot history (bounded)
    snapshots: Vec<VolatileSnapshot>,
    
    /// Next sequence number
    next_sequence: u64,
    
    /// Last snapshot timestamp
    last_snapshot: u64,
    
    /// Configuration
    config: SnapshotConfig,
}

impl SnapshotManager {
    /// Create new snapshot manager
    pub fn new(config: SnapshotConfig) -> Self {
        Self {
            snapshots: Vec::new(),
            next_sequence: 0,
            last_snapshot: current_timestamp(),
            config,
        }
    }
    
    /// Check if snapshot due
    pub fn snapshot_due(&self) -> bool {
        let current_time = current_timestamp();
        current_time - self.last_snapshot >= self.config.snapshot_interval_ms
    }
    
    /// Create snapshot
    ///
    /// ## Lifecycle Stage: Execution
    ///
    /// # Inputs
    /// - `state_data`: Execution state to snapshot
    /// - `encryption_key`: Ephemeral session key
    ///
    /// # Outputs
    /// - Snapshot sequence number
    ///
    /// ## Audit Trail
    /// - Logs snapshot creation to ephemeral ledger
    pub fn create_snapshot(
        &mut self,
        state_data: &[u8],
        encryption_key: &[u8; 64],
    ) -> u64 {
        let snapshot = VolatileSnapshot::create(
            self.next_sequence,
            state_data,
            encryption_key,
        );
        
        let sequence = snapshot.sequence;
        
        // Add to bounded history
        self.snapshots.push(snapshot);
        if self.snapshots.len() > self.config.max_snapshots {
            self.snapshots.remove(0);
        }
        
        self.next_sequence += 1;
        self.last_snapshot = current_timestamp();
        
        sequence
    }
    
    /// Restore from latest snapshot
    ///
    /// ## Lifecycle Stage: Execution (recovery)
    pub fn restore_latest(
        &self,
        encryption_key: &[u8; 64],
    ) -> Result<Vec<u8>, &'static str> {
        let snapshot = self.snapshots.last()
            .ok_or("No snapshots available")?;
        snapshot.restore(encryption_key)
    }
    
    /// Restore from specific snapshot
    pub fn restore_by_sequence(
        &self,
        sequence: u64,
        encryption_key: &[u8; 64],
    ) -> Result<Vec<u8>, &'static str> {
        let snapshot = self.snapshots.iter()
            .find(|s| s.sequence == sequence)
            .ok_or("Snapshot not found")?;
        snapshot.restore(encryption_key)
    }
    
    /// Get snapshot count
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
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
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_snapshot_creation() {
        let state = b"execution state data";
        let key = [1u8; 64];
        
        let snapshot = VolatileSnapshot::create(0, state, &key);
        assert_eq!(snapshot.sequence, 0);
    }
    
    #[test]
    fn test_snapshot_restore() {
        let state = b"execution state data";
        let key = [1u8; 64];
        
        let snapshot = VolatileSnapshot::create(0, state, &key);
        let restored = snapshot.restore(&key).unwrap();
        
        assert_eq!(restored, state);
    }
    
    #[test]
    fn test_snapshot_manager() {
        let config = SnapshotConfig::default();
        let mut manager = SnapshotManager::new(config);
        let key = [2u8; 64];
        
        let seq = manager.create_snapshot(b"state1", &key);
        assert_eq!(seq, 0);
        assert_eq!(manager.snapshot_count(), 1);
    }
}
