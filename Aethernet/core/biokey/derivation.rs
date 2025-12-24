//! Biokey Derivation Module
//!
//! Multi-factor ephemeral key derivation from:
//! - SNP (Single Nucleotide Polymorphism) loci
//! - Device PUF (Physical Unclonable Function)
//! - Ephemeral salt (session-specific)
//! - Temporal nonce (time-bounded)
//!
//! Keys are derived on-demand, used once, and immediately wiped from RAM.
//! Automatic rotation ensures keys are never reused across sessions.
//!
//! Security Hardening (Aethernet Phase I-II):
//! - Multi-factor entropy prevents single-point compromise
//! - Temporal nonces enforce time-bounded validity
//! - Automatic rotation with decay tracking
//! - Memory scrubbing hooks prevent key material leakage

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use sha3::{Digest, Sha3_256, Sha3_512};
use core::ptr;

/// SNP loci identifier (chromosome + position)
#[derive(Debug, Clone, Copy)]
pub struct SNPLocus {
    /// Chromosome (1-22, X, Y)
    pub chromosome: u8,
    /// Position on chromosome
    pub position: u64,
    /// Reference allele (A, C, G, T)
    pub ref_allele: u8,
    /// Alternative allele (A, C, G, T)
    pub alt_allele: u8,
}

/// Device PUF (Physical Unclonable Function) data
/// Represents hardware-specific entropy from device silicon variations
#[derive(Debug, Clone)]
pub struct DevicePUF {
    /// PUF challenge-response pair hash
    pub cr_hash: [u8; 32],
    /// Device serial number or identifier
    pub device_id: [u8; 16],
}

/// Temporal nonce for time-bounded key derivation
#[derive(Debug, Clone, Copy)]
pub struct TemporalNonce {
    /// Unix timestamp (seconds since epoch)
    pub timestamp: u64,
    /// Session counter (prevents replay within same second)
    pub counter: u32,
    /// Epoch identifier for zone-aware rotation
    pub epoch_id: u64,
}

/// Key rotation metadata
#[derive(Debug, Clone, Copy)]
pub struct RotationMetadata {
    /// Generation number (increments with each rotation)
    pub generation: u32,
    /// Last rotation timestamp
    pub last_rotation: u64,
    /// Rotation interval in seconds
    pub rotation_interval: u64,
}

/// Ephemeral biokey derived from multi-factor entropy
/// 
/// Security hardening:
/// - Multi-factor: SNP + PUF + salt + nonce
/// - Automatic rotation enforced via metadata
/// - Memory scrubbing on drop
pub struct EphemeralBiokey {
    /// Raw key material (64 bytes for SHA3-512)
    key_material: [u8; 64],
    /// Creation timestamp
    created_at: u64,
    /// Time-to-live in seconds (default: 60)
    ttl: u64,
    /// Rotation metadata for automatic key cycling
    rotation_meta: RotationMetadata,
    /// Temporal nonce used in derivation
    nonce: TemporalNonce,
}

impl EphemeralBiokey {
    /// Create new ephemeral biokey with multi-factor entropy
    ///
    /// # Arguments
    /// * `loci` - Array of SNP loci for biometric component
    /// * `puf_data` - Device PUF for hardware binding
    /// * `ephemeral_salt` - Session-specific entropy
    /// * `nonce` - Temporal nonce for time-binding
    /// * `ttl` - Time-to-live in seconds
    ///
    /// # Returns
    /// * Ephemeral biokey with automatic rotation and wipe-on-drop
    ///
    /// # Security
    /// * Uses SHA3-512 for 512-bit key material
    /// * Combines all entropy sources in deterministic order
    /// * Resistant to single-factor compromise
    pub fn derive(
        loci: &[SNPLocus],
        puf_data: &DevicePUF,
        ephemeral_salt: &[u8],
        nonce: TemporalNonce,
        ttl: u64,
    ) -> Self {
        // Multi-factor key derivation using SHA3-512
        let mut hasher = Sha3_512::new();
        
        // Factor 1: SNP loci (biometric component)
        for locus in loci {
            hasher.update(&locus.chromosome.to_le_bytes());
            hasher.update(&locus.position.to_le_bytes());
            hasher.update(&[locus.ref_allele]);
            hasher.update(&[locus.alt_allele]);
        }
        
        // Factor 2: Device PUF (hardware binding)
        hasher.update(&puf_data.cr_hash);
        hasher.update(&puf_data.device_id);
        
        // Factor 3: Ephemeral salt (session-specific)
        hasher.update(ephemeral_salt);
        
        // Factor 4: Temporal nonce (time-bounded)
        hasher.update(&nonce.timestamp.to_le_bytes());
        hasher.update(&nonce.counter.to_le_bytes());
        hasher.update(&nonce.epoch_id.to_le_bytes());
        
        // Finalize hash to 512-bit key material
        let result = hasher.finalize();
        let mut key_material = [0u8; 64];
        key_material.copy_from_slice(&result);
        
        Self {
            key_material,
            created_at: nonce.timestamp,
            ttl,
            rotation_meta: RotationMetadata {
                generation: 0,
                last_rotation: nonce.timestamp,
                rotation_interval: ttl, // Rotate at same interval as TTL
            },
            nonce,
        }
    }
    
    /// Rotate biokey with new temporal nonce
    ///
    /// Creates new key material while preserving generation tracking.
    /// Old key is securely wiped before returning new key.
    ///
    /// # Arguments
    /// * `loci` - SNP loci (unchanged)
    /// * `puf_data` - Device PUF (unchanged)
    /// * `ephemeral_salt` - Fresh session salt
    /// * `new_nonce` - New temporal nonce
    ///
    /// # Returns
    /// * New biokey with incremented generation
    ///
    /// # Security
    /// * Automatic rotation prevents key reuse
    /// * Old key material is wiped before return
    pub fn rotate(
        &mut self,
        loci: &[SNPLocus],
        puf_data: &DevicePUF,
        ephemeral_salt: &[u8],
        new_nonce: TemporalNonce,
    ) -> Self {
        // Wipe current key before generating new one
        self.wipe();
        
        // Generate new key with updated nonce
        let mut new_key = Self::derive(loci, puf_data, ephemeral_salt, new_nonce, self.ttl);
        
        // Increment generation counter
        new_key.rotation_meta.generation = self.rotation_meta.generation + 1;
        new_key.rotation_meta.last_rotation = new_nonce.timestamp;
        
        new_key
    }
    
    /// Check if key needs rotation based on interval
    ///
    /// # Arguments
    /// * `current_time` - Current timestamp
    ///
    /// # Returns
    /// * `true` if rotation is due, `false` otherwise
    pub fn needs_rotation(&self, current_time: u64) -> bool {
        current_time - self.rotation_meta.last_rotation >= self.rotation_meta.rotation_interval
    }
    
    /// Get key material (use once only)
    ///
    /// # Returns
    /// * Key material bytes (64 bytes)
    ///
    /// # Security
    /// * Caller MUST wipe returned data after use
    /// * Key should be used immediately and not stored
    pub fn get_key_material(&self) -> &[u8; 64] {
        &self.key_material
    }
    
    /// Get generation number for audit trail
    pub fn generation(&self) -> u32 {
        self.rotation_meta.generation
    }
    
    /// Get temporal nonce for verification
    pub fn nonce(&self) -> &TemporalNonce {
        &self.nonce
    }
    
    /// Check if key has expired
    ///
    /// # Arguments
    /// * `current_time` - Current timestamp in seconds
    ///
    /// # Returns
    /// * `true` if expired, `false` otherwise
    pub fn is_expired(&self, current_time: u64) -> bool {
        current_time - self.created_at > self.ttl
    }
    
    /// Secure wipe of key material from memory
    ///
    /// Uses volatile writes to prevent compiler optimization.
    /// In production, would also:
    /// - Clear CPU registers
    /// - Flush cache lines
    /// - Overwrite with random data multiple times
    pub fn wipe(&mut self) {
        // Overwrite with zeros using volatile writes
        for i in 0..self.key_material.len() {
            unsafe {
                ptr::write_volatile(&mut self.key_material[i], 0);
            }
        }
        
        // Reset metadata
        unsafe {
            ptr::write_volatile(&mut self.created_at, 0);
            ptr::write_volatile(&mut self.ttl, 0);
        }
    }
}

impl Drop for EphemeralBiokey {
    /// Auto-wipe on drop
    fn drop(&mut self) {
        self.wipe();
    }
}

/// Generate zero-knowledge proof for biokey
///
/// Proves possession of valid biokey without revealing SNP data.
/// Uses commitment scheme: commit to SNP loci, prove knowledge without disclosure.
///
/// # Arguments
/// * `loci` - SNP loci array
/// * `salt` - Additional entropy
///
/// # Returns
/// * ZK proof bytes (placeholder for Risc0/Halo2 integration)
pub fn generate_zkp(loci: &[SNPLocus], salt: &[u8]) -> Vec<u8> {
    // In production, this would:
    // 1. Generate commitment to SNP loci
    // 2. Create zero-knowledge proof using Risc0 or Halo2
    // 3. Return proof that can be verified without revealing SNP data
    
    // Placeholder: hash-based commitment
    let mut hasher = Sha3_256::new();
    
    for locus in loci {
        hasher.update(&locus.chromosome.to_le_bytes());
        hasher.update(&locus.position.to_le_bytes());
        // Note: Don't include alleles in proof to maintain privacy
    }
    
    hasher.update(salt);
    
    let result = hasher.finalize();
    result.to_vec()
}

/// Select SNP loci for biokey derivation
///
/// Criteria:
/// - High heterozygosity (rare alleles)
/// - Low linkage disequilibrium (independent)
/// - Non-coding regions (privacy protection)
/// - Stable over lifetime (avoid somatic mutations)
///
/// # Arguments
/// * `vcf_data` - Variant call format data
/// * `num_loci` - Number of loci to select (default: 100)
///
/// # Returns
/// * Selected SNP loci suitable for biokey derivation
pub fn select_snp_loci(vcf_data: &[u8], num_loci: usize) -> Vec<SNPLocus> {
    // In production, this would:
    // 1. Parse VCF file
    // 2. Filter variants by criteria
    // 3. Rank by heterozygosity and independence
    // 4. Select top N loci
    
    // Placeholder: return empty vector
    Vec::new()
}

/// Secure comparison of biokeys (constant-time)
///
/// # Arguments
/// * `key1` - First biokey (64 bytes)
/// * `key2` - Second biokey (64 bytes)
///
/// # Returns
/// * `true` if keys match, `false` otherwise
///
/// # Security
/// * Uses constant-time comparison to prevent timing attacks
pub fn secure_compare(key1: &[u8; 64], key2: &[u8; 64]) -> bool {
    let mut diff = 0u8;
    
    for i in 0..64 {
        diff |= key1[i] ^ key2[i];
    }
    
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_puf() -> DevicePUF {
        DevicePUF {
            cr_hash: [0x42u8; 32],
            device_id: [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                        0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10],
        }
    }
    
    fn create_test_nonce(timestamp: u64) -> TemporalNonce {
        TemporalNonce {
            timestamp,
            counter: 1,
            epoch_id: 1000,
        }
    }
    
    #[test]
    fn test_ephemeral_biokey_derivation() {
        let loci = [
            SNPLocus {
                chromosome: 1,
                position: 12345,
                ref_allele: b'A',
                alt_allele: b'G',
            },
            SNPLocus {
                chromosome: 2,
                position: 67890,
                ref_allele: b'C',
                alt_allele: b'T',
            },
        ];
        
        let puf = create_test_puf();
        let salt = b"operator-uuid-12345";
        let nonce = create_test_nonce(1000);
        let biokey = EphemeralBiokey::derive(&loci, &puf, salt, nonce, 60);
        
        // Key material should be non-zero
        assert_ne!(biokey.key_material, [0u8; 64]);
        // Generation should start at 0
        assert_eq!(biokey.generation(), 0);
    }
    
    #[test]
    fn test_biokey_wipe() {
        let loci = [
            SNPLocus {
                chromosome: 1,
                position: 12345,
                ref_allele: b'A',
                alt_allele: b'G',
            },
        ];
        
        let puf = create_test_puf();
        let salt = b"test-salt";
        let nonce = create_test_nonce(1000);
        let mut biokey = EphemeralBiokey::derive(&loci, &puf, salt, nonce, 60);
        
        // Verify key material exists
        assert_ne!(biokey.key_material, [0u8; 64]);
        
        // Wipe key
        biokey.wipe();
        
        // Verify key material is zeroed
        assert_eq!(biokey.key_material, [0u8; 64]);
    }
    
    #[test]
    fn test_biokey_expiration() {
        let loci = [
            SNPLocus {
                chromosome: 1,
                position: 12345,
                ref_allele: b'A',
                alt_allele: b'G',
            },
        ];
        
        let puf = create_test_puf();
        let salt = b"test-salt";
        let nonce = create_test_nonce(1000);
        let biokey = EphemeralBiokey::derive(&loci, &puf, salt, nonce, 60);
        
        // Should not be expired at 1030 (30 seconds elapsed)
        assert!(!biokey.is_expired(1030));
        
        // Should be expired at 1100 (100 seconds elapsed)
        assert!(biokey.is_expired(1100));
    }
    
    #[test]
    fn test_biokey_rotation() {
        let loci = [
            SNPLocus {
                chromosome: 1,
                position: 12345,
                ref_allele: b'A',
                alt_allele: b'G',
            },
        ];
        
        let puf = create_test_puf();
        let salt = b"test-salt";
        let nonce1 = create_test_nonce(1000);
        let mut biokey = EphemeralBiokey::derive(&loci, &puf, salt, nonce1, 60);
        
        let old_material = biokey.key_material;
        let old_generation = biokey.generation();
        
        // Check if rotation is needed
        assert!(!biokey.needs_rotation(1030)); // 30 seconds - not yet
        assert!(biokey.needs_rotation(1100)); // 100 seconds - rotation due
        
        // Rotate key
        let nonce2 = create_test_nonce(1100);
        let new_biokey = biokey.rotate(&loci, &puf, b"new-salt", nonce2);
        
        // Key material should be different
        assert_ne!(new_biokey.key_material, old_material);
        // Generation should increment
        assert_eq!(new_biokey.generation(), old_generation + 1);
    }
    
    #[test]
    fn test_multi_factor_entropy() {
        // Test that different inputs produce different keys
        let loci = [
            SNPLocus {
                chromosome: 1,
                position: 12345,
                ref_allele: b'A',
                alt_allele: b'G',
            },
        ];
        
        let puf1 = create_test_puf();
        let mut puf2 = create_test_puf();
        puf2.device_id[0] = 0xFF; // Different device
        
        let salt = b"test-salt";
        let nonce1 = create_test_nonce(1000);
        let nonce2 = create_test_nonce(1001); // Different timestamp
        
        let key1 = EphemeralBiokey::derive(&loci, &puf1, salt, nonce1, 60);
        let key2 = EphemeralBiokey::derive(&loci, &puf2, salt, nonce1, 60); // Different PUF
        let key3 = EphemeralBiokey::derive(&loci, &puf1, salt, nonce2, 60); // Different nonce
        
        // All keys should be different
        assert_ne!(key1.key_material, key2.key_material);
        assert_ne!(key1.key_material, key3.key_material);
        assert_ne!(key2.key_material, key3.key_material);
    }
    
    #[test]
    fn test_zkp_generation() {
        let loci = [
            SNPLocus {
                chromosome: 1,
                position: 12345,
                ref_allele: b'A',
                alt_allele: b'G',
            },
        ];
        
        let salt = b"test-salt";
        let zkp = generate_zkp(&loci, salt);
        
        // ZKP should be generated (32 bytes for SHA3-256)
        assert_eq!(zkp.len(), 32);
    }
    
    #[test]
    fn test_secure_compare() {
        let key1 = [1u8; 64];
        let key2 = [1u8; 64];
        let key3 = [2u8; 64];
        
        // Same keys should match
        assert!(secure_compare(&key1, &key2));
        
        // Different keys should not match
        assert!(!secure_compare(&key1, &key3));
    }
}
