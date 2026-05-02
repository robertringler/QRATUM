//! Post-Quantum Cryptography Module
//!
//! Provides NIST-standardized post-quantum cryptographic primitives for QRATUM:
//! - SPHINCS+: Stateless hash-based signatures
//! - CRYSTALS-Kyber: Lattice-based key encapsulation
//! - CRYSTALS-Dilithium: Lattice-based digital signatures
//!
//! These algorithms replace classical crypto with quantum-resistant alternatives
//! to protect QRATUM against future quantum computer attacks.

pub mod sphincs_plus;
pub mod crystals_kyber;
pub mod crystals_dilithium;

pub use sphincs_plus::{
    PublicKey as SPHINCSPublicKey,
    SecretKey as SPHINCSSecretKey,
    Signature as SPHINCSSignature,
    generate_keypair as sphincs_generate_keypair,
    sign as sphincs_sign,
    verify as sphincs_verify,
    SPHINCSError,
};

pub use crystals_kyber::{
    PublicKey as KyberPublicKey,
    SecretKey as KyberSecretKey,
    Ciphertext as KyberCiphertext,
    SharedSecret as KyberSharedSecret,
    generate_keypair as kyber_generate_keypair,
    encapsulate as kyber_encapsulate,
    decapsulate as kyber_decapsulate,
    KyberError,
};

pub use crystals_dilithium::{
    PublicKey as DilithiumPublicKey,
    SecretKey as DilithiumSecretKey,
    Signature as DilithiumSignature,
    generate_keypair as dilithium_generate_keypair,
    sign as dilithium_sign,
    verify as dilithium_verify,
    sign_with_context as dilithium_sign_with_context,
    verify_with_context as dilithium_verify_with_context,
    DilithiumError,
};

/// PQC algorithm selection for different use cases
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PQCAlgorithm {
    /// SPHINCS+ for long-term signatures (larger sigs, stateless)
    SPHINCSPlus,
    /// Dilithium for general signatures (smaller sigs, faster)
    Dilithium,
    /// Kyber for key exchange
    Kyber,
}

impl PQCAlgorithm {
    pub fn recommended_for_signatures() -> Self {
        // Dilithium is faster and has smaller signatures
        PQCAlgorithm::Dilithium
    }
    
    pub fn recommended_for_long_term_signatures() -> Self {
        // SPHINCS+ is stateless and hash-based (more conservative)
        PQCAlgorithm::SPHINCSPlus
    }
    
    pub fn recommended_for_key_exchange() -> Self {
        PQCAlgorithm::Kyber
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sphincs_integration() {
        let (pk, sk) = sphincs_generate_keypair().unwrap();
        let message = b"QRATUM PQC Test";
        let sig = sphincs_sign(message, &sk).unwrap();
        assert!(sphincs_verify(message, &sig, &pk).unwrap());
    }
    
    #[test]
    fn test_kyber_integration() {
        let (pk, sk) = kyber_generate_keypair().unwrap();
        let (ss1, ct) = kyber_encapsulate(&pk).unwrap();
        let ss2 = kyber_decapsulate(&ct, &sk).unwrap();
        assert_eq!(ss1.data.len(), ss2.data.len());
    }
    
    #[test]
    fn test_dilithium_integration() {
        let (pk, sk) = dilithium_generate_keypair().unwrap();
        let message = b"QRATUM PQC Test";
        let sig = dilithium_sign(message, &sk).unwrap();
        assert!(dilithium_verify(message, &sig, &pk).unwrap());
    }
    
    #[test]
    fn test_algorithm_recommendations() {
        assert_eq!(
            PQCAlgorithm::recommended_for_signatures(),
            PQCAlgorithm::Dilithium
        );
        assert_eq!(
            PQCAlgorithm::recommended_for_long_term_signatures(),
            PQCAlgorithm::SPHINCSPlus
        );
        assert_eq!(
            PQCAlgorithm::recommended_for_key_exchange(),
            PQCAlgorithm::Kyber
        );
    }
}
