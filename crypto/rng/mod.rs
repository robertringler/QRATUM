//! Random Number Generation Module
//!
//! Provides cryptographically secure random number generation for QRATUM:
//! - HMAC-DRBG (NIST SP 800-90A compliant)
//! - Entropy pooling from multiple sources
//! - Automatic reseeding with prediction resistance
//!
//! Security Properties:
//! - All sensitive state zeroized on drop
//! - Memory fencing for secure operations
//! - Constant-time operations where applicable

pub mod drbg;

pub use drbg::{
    HmacDrbg,
    SecureDrbg,
    EntropyPool,
    EntropySource,
    SystemEntropySource,
    TimestampEntropySource,
    DrbgError,
    SECURITY_STRENGTH,
    SEED_LENGTH,
    MAX_BYTES_PER_REQUEST,
    RESEED_INTERVAL,
    MIN_ENTROPY,
};

/// Generate cryptographically secure random bytes using the global DRBG
///
/// This is a convenience function that creates and uses a secure DRBG.
/// For high-performance scenarios, prefer creating your own SecureDrbg instance.
pub fn generate_random(output: &mut [u8]) -> Result<(), DrbgError> {
    let drbg = SecureDrbg::new(Some(b"QRATUM-GLOBAL"))?;
    drbg.generate(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_generate_random() {
        let mut output = [0u8; 32];
        let result = generate_random(&mut output);
        assert!(result.is_ok());
        assert!(output.iter().any(|&b| b != 0));
    }
}
