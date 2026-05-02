//! Key Derivation Function Module
//!
//! Provides cryptographic key derivation for QRATUM:
//! - HKDF-SHA3-512 (RFC 5869 compliant with SHA3)
//! - Labeled key derivation for domain separation
//! - Key schedule derivation for encryption/MAC
//!
//! Security Properties:
//! - SHA3-512 based for post-quantum security margin
//! - Explicit zeroization of sensitive data
//! - Constant-time operations where applicable
//! - Support for labeled derivation

pub mod hkdf;

pub use hkdf::{
    Hkdf,
    Prk,
    KeySchedule,
    HkdfError,
    derive,
    derive_fixed,
    derive_labeled,
    HASH_LENGTH,
    MAX_OUTPUT_LENGTH,
};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_module_exports() {
        // Verify exports work
        let okm = derive(None, b"test", b"info", 32).unwrap();
        assert_eq!(okm.len(), 32);
    }
}
