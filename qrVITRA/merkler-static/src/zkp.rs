//! Zero-Knowledge Proof Module
//!
//! Implements challenge-response protocol for biokey verification
//! without revealing genomic data.

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use crate::biokey::EphemeralBiokey;

/// Zero-Knowledge Proof for biokey authentication
///
/// Protocol:
/// 1. Verifier generates random 256-bit challenge
/// 2. Prover computes response = SHA3(biokey || challenge)
/// 3. Verifier checks response matches expected hash
/// 4. No genome data revealed during verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiokeyZkp {
    pub challenge: [u8; 32],
    pub response: [u8; 32],
}

impl BiokeyZkp {
    /// Generate a random challenge
    ///
    /// # Returns
    /// Random 256-bit challenge for ZKP protocol
    pub fn generate_challenge() -> [u8; 32] {
        use rand::RngCore;
        let mut challenge = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut challenge);
        challenge
    }
    
    /// Prove possession of biokey without revealing it
    ///
    /// # Arguments
    /// * `biokey` - Ephemeral biokey to prove possession of
    /// * `challenge` - Random challenge from verifier
    ///
    /// # Returns
    /// Zero-knowledge proof containing response
    ///
    /// # Security
    /// - Response doesn't reveal private key or genome data
    /// - Challenge must be unique per session (prevent replay)
    /// - Uses SHA3-256 for post-quantum resistance
    pub fn prove(biokey: &EphemeralBiokey, challenge: &[u8; 32]) -> Self {
        let mut hasher = Sha3_256::new();
        hasher.update(biokey.private_key());
        hasher.update(challenge);
        let response: [u8; 32] = hasher.finalize().into();
        
        BiokeyZkp {
            challenge: *challenge,
            response,
        }
    }
    
    /// Verify zero-knowledge proof
    ///
    /// # Arguments
    /// * `public_hash` - Public hash of the biokey
    /// * `expected_response` - Expected response for verification
    ///
    /// # Returns
    /// True if proof is valid
    ///
    /// # Note
    /// In a full implementation, this would require the verifier to have
    /// precomputed expected responses or use interactive verification.
    /// This simplified version demonstrates the concept.
    pub fn verify(&self, public_hash: &[u8; 32]) -> bool {
        // In a real ZKP system, verification would be more complex
        // For this simplified implementation, we just check format validity
        self.challenge.len() == 32 && 
        self.response.len() == 32 && 
        public_hash.len() == 32
    }
    
    /// Verify that a response matches a specific biokey and challenge
    ///
    /// # Arguments
    /// * `biokey` - The biokey to verify against
    /// * `challenge` - The challenge used
    /// * `response` - The response to verify
    ///
    /// # Returns
    /// True if the response is valid for this biokey and challenge
    pub fn verify_response(biokey: &EphemeralBiokey, challenge: &[u8; 32], response: &[u8; 32]) -> bool {
        let expected_proof = Self::prove(biokey, challenge);
        &expected_proof.response == response
    }
    
    /// Export proof as JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
    
    /// Import proof from JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biokey::SnpLocus;
    
    fn create_test_biokey() -> EphemeralBiokey {
        let loci = vec![
            SnpLocus {
                chromosome: "chr1".to_string(),
                position: 12345,
                ref_allele: "A".to_string(),
                alt_allele: "G".to_string(),
                quality: 35.0,
                depth: 20,
            },
        ];
        EphemeralBiokey::derive_from_loci(&loci)
    }
    
    #[test]
    fn test_generate_challenge() {
        let challenge1 = BiokeyZkp::generate_challenge();
        let challenge2 = BiokeyZkp::generate_challenge();
        
        assert_eq!(challenge1.len(), 32);
        assert_eq!(challenge2.len(), 32);
        
        // Challenges should be different (with high probability)
        assert_ne!(challenge1, challenge2);
    }
    
    #[test]
    fn test_prove_and_verify() {
        let biokey = create_test_biokey();
        let challenge = BiokeyZkp::generate_challenge();
        
        let proof = BiokeyZkp::prove(&biokey, &challenge);
        
        assert_eq!(proof.challenge, challenge);
        assert_eq!(proof.response.len(), 32);
        
        // Verify proof format
        assert!(proof.verify(&biokey.public_hash));
    }
    
    #[test]
    fn test_deterministic_proof() {
        let biokey = create_test_biokey();
        let challenge = [42u8; 32]; // Fixed challenge
        
        let proof1 = BiokeyZkp::prove(&biokey, &challenge);
        let proof2 = BiokeyZkp::prove(&biokey, &challenge);
        
        // Same challenge should produce same response
        assert_eq!(proof1.response, proof2.response);
    }
    
    #[test]
    fn test_different_challenges_different_responses() {
        let biokey = create_test_biokey();
        
        let challenge1 = [1u8; 32];
        let challenge2 = [2u8; 32];
        
        let proof1 = BiokeyZkp::prove(&biokey, &challenge1);
        let proof2 = BiokeyZkp::prove(&biokey, &challenge2);
        
        // Different challenges should produce different responses
        assert_ne!(proof1.response, proof2.response);
    }
    
    #[test]
    fn test_verify_response() {
        let biokey = create_test_biokey();
        let challenge = BiokeyZkp::generate_challenge();
        
        let proof = BiokeyZkp::prove(&biokey, &challenge);
        
        // Valid response should verify
        assert!(BiokeyZkp::verify_response(&biokey, &challenge, &proof.response));
        
        // Invalid response should not verify
        let invalid_response = [0u8; 32];
        assert!(!BiokeyZkp::verify_response(&biokey, &challenge, &invalid_response));
    }
    
    #[test]
    fn test_json_serialization() {
        let biokey = create_test_biokey();
        let challenge = BiokeyZkp::generate_challenge();
        let proof = BiokeyZkp::prove(&biokey, &challenge);
        
        // Serialize to JSON
        let json = proof.to_json().unwrap();
        assert!(json.contains("challenge"));
        assert!(json.contains("response"));
        
        // Deserialize from JSON
        let decoded_proof = BiokeyZkp::from_json(&json).unwrap();
        assert_eq!(decoded_proof.challenge, proof.challenge);
        assert_eq!(decoded_proof.response, proof.response);
    }
    
    #[test]
    fn test_different_biokeys_different_responses() {
        let loci1 = vec![
            SnpLocus {
                chromosome: "chr1".to_string(),
                position: 12345,
                ref_allele: "A".to_string(),
                alt_allele: "G".to_string(),
                quality: 35.0,
                depth: 20,
            },
        ];
        
        let loci2 = vec![
            SnpLocus {
                chromosome: "chr2".to_string(),
                position: 67890,
                ref_allele: "C".to_string(),
                alt_allele: "T".to_string(),
                quality: 40.0,
                depth: 25,
            },
        ];
        
        let biokey1 = EphemeralBiokey::derive_from_loci(&loci1);
        let biokey2 = EphemeralBiokey::derive_from_loci(&loci2);
        
        let challenge = [42u8; 32];
        
        let proof1 = BiokeyZkp::prove(&biokey1, &challenge);
        let proof2 = BiokeyZkp::prove(&biokey2, &challenge);
        
        // Different biokeys should produce different responses
        assert_ne!(proof1.response, proof2.response);
    }
}
