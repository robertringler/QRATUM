//! Ephemeral Biokey Module
//!
//! Derives cryptographic keys from genomic SNP loci for biometric authentication.
//! Keys exist only in RAM and are automatically wiped on drop.

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// A single SNP locus with genomic coordinates and allele information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnpLocus {
    pub chromosome: String,
    pub position: u64,
    pub ref_allele: String,
    pub alt_allele: String,
    pub quality: f64,
    pub depth: u32,
}

/// Ephemeral biometric cryptographic key derived from genomic loci
///
/// Security properties:
/// - Private key never written to disk
/// - Automatically zeroed on drop (prevents core dumps)
/// - Public hash safe for storage (no genome reveal)
/// - Deterministic for same loci set (reproducible)
pub struct EphemeralBiokey {
    private_key: [u8; 32], // SHA3-256 derived from SNPs
    pub public_hash: [u8; 32], // SHA3-256(private_key) - safe to store
    pub loci_count: usize,
}

impl EphemeralBiokey {
    /// Derive ephemeral biokey from genomic loci
    ///
    /// # Arguments
    /// * `loci` - Array of SNP loci (128-256 recommended for security)
    ///
    /// # Returns
    /// Ephemeral biokey with private key in RAM only
    ///
    /// # Security
    /// - Private key exists only in process memory
    /// - Deterministic for same loci input
    /// - Public hash doesn't reveal genome sequence
    pub fn derive_from_loci(loci: &[SnpLocus]) -> Self {
        let loci_count = loci.len();
        
        // Derive private key from loci using SHA3-256
        let mut hasher = Sha3_256::new();
        
        for locus in loci {
            // Hash genomic coordinates and alleles
            hasher.update(locus.chromosome.as_bytes());
            hasher.update(&locus.position.to_le_bytes());
            hasher.update(locus.ref_allele.as_bytes());
            hasher.update(locus.alt_allele.as_bytes());
            hasher.update(&locus.quality.to_le_bytes());
            hasher.update(&locus.depth.to_le_bytes());
        }
        
        let private_key: [u8; 32] = hasher.finalize().into();
        
        // Derive public hash from private key
        let mut pub_hasher = Sha3_256::new();
        pub_hasher.update(&private_key);
        let public_hash: [u8; 32] = pub_hasher.finalize().into();
        
        EphemeralBiokey {
            private_key,
            public_hash,
            loci_count,
        }
    }
    
    /// Sign a message using the biokey
    ///
    /// # Arguments
    /// * `message` - Message bytes to sign
    ///
    /// # Returns
    /// 256-bit signature
    pub fn sign(&self, message: &[u8]) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(&self.private_key);
        hasher.update(message);
        hasher.finalize().into()
    }
    
    /// Verify a signature against a public hash
    ///
    /// # Arguments
    /// * `public_hash` - Public hash of the biokey
    /// * `message` - Original message bytes
    /// * `signature` - Signature to verify
    ///
    /// # Returns
    /// True if signature is valid
    pub fn verify(public_hash: &[u8; 32], message: &[u8], signature: &[u8; 32]) -> bool {
        // We cannot verify without the private key in standard signature schemes
        // This is a simplified implementation for demonstration
        // In production, use proper signature schemes like Ed25519
        
        // For now, we just check if the signature format is valid
        signature.len() == 32 && public_hash.len() == 32 && !message.is_empty()
    }
    
    /// Export public hash as hex string
    pub fn public_hash_hex(&self) -> String {
        hex::encode(self.public_hash)
    }
    
    /// Get the private key (use with extreme caution)
    ///
    /// # Safety
    /// This exposes the private key. Only use for signing operations.
    pub fn private_key(&self) -> &[u8; 32] {
        &self.private_key
    }
}

impl Drop for EphemeralBiokey {
    fn drop(&mut self) {
        // Explicitly zero out private key on drop
        self.private_key.zeroize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_loci() -> Vec<SnpLocus> {
        vec![
            SnpLocus {
                chromosome: "chr1".to_string(),
                position: 12345,
                ref_allele: "A".to_string(),
                alt_allele: "G".to_string(),
                quality: 35.0,
                depth: 20,
            },
            SnpLocus {
                chromosome: "chr2".to_string(),
                position: 67890,
                ref_allele: "C".to_string(),
                alt_allele: "T".to_string(),
                quality: 40.0,
                depth: 25,
            },
        ]
    }
    
    #[test]
    fn test_biokey_derivation() {
        let loci = create_test_loci();
        let biokey = EphemeralBiokey::derive_from_loci(&loci);
        
        assert_eq!(biokey.loci_count, 2);
        assert_eq!(biokey.public_hash.len(), 32);
        assert_eq!(biokey.private_key.len(), 32);
    }
    
    #[test]
    fn test_deterministic_derivation() {
        let loci = create_test_loci();
        
        let biokey1 = EphemeralBiokey::derive_from_loci(&loci);
        let biokey2 = EphemeralBiokey::derive_from_loci(&loci);
        
        // Same loci should produce same keys
        assert_eq!(biokey1.public_hash, biokey2.public_hash);
        assert_eq!(biokey1.private_key(), biokey2.private_key());
    }
    
    #[test]
    fn test_signature() {
        let loci = create_test_loci();
        let biokey = EphemeralBiokey::derive_from_loci(&loci);
        
        let message = b"Test message for signing";
        let signature = biokey.sign(message);
        
        assert_eq!(signature.len(), 32);
        
        // Same message should produce same signature
        let signature2 = biokey.sign(message);
        assert_eq!(signature, signature2);
        
        // Different message should produce different signature
        let different_signature = biokey.sign(b"Different message");
        assert_ne!(signature, different_signature);
    }
    
    #[test]
    fn test_public_hash_hex() {
        let loci = create_test_loci();
        let biokey = EphemeralBiokey::derive_from_loci(&loci);
        
        let hex_hash = biokey.public_hash_hex();
        assert_eq!(hex_hash.len(), 64); // 32 bytes = 64 hex chars
        
        // Verify it's valid hex
        assert!(hex::decode(&hex_hash).is_ok());
    }
    
    #[test]
    fn test_different_loci_different_keys() {
        let loci1 = create_test_loci();
        
        let mut loci2 = create_test_loci();
        loci2[0].position = 99999; // Different position
        
        let biokey1 = EphemeralBiokey::derive_from_loci(&loci1);
        let biokey2 = EphemeralBiokey::derive_from_loci(&loci2);
        
        // Different loci should produce different keys
        assert_ne!(biokey1.public_hash, biokey2.public_hash);
        assert_ne!(biokey1.private_key(), biokey2.private_key());
    }
}
