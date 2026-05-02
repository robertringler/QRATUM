//! HKDF-SHA3-512 Key Derivation Function
//!
//! Implementation of RFC 5869 HKDF using SHA3-512 as the underlying
//! hash function for post-quantum security margin.
//!
//! HKDF (HMAC-based Key Derivation Function) consists of two stages:
//! 1. Extract: Create a fixed-length pseudorandom key from input keying material
//! 2. Expand: Expand the pseudorandom key into additional keying material
//!
//! Security Properties:
//! - 512-bit output per iteration (SHA3-512)
//! - Constant-time operations
//! - Explicit zeroization on drop
//! - Domain separation via info parameter

use sha3::{Sha3_512, Digest};
use zeroize::{Zeroize, ZeroizeOnDrop};
use std::error::Error;
use std::fmt;

/// SHA3-512 output length in bytes
pub const HASH_LENGTH: usize = 64;

/// Maximum output key material length (255 * HASH_LENGTH per RFC 5869)
pub const MAX_OUTPUT_LENGTH: usize = 255 * HASH_LENGTH;

#[derive(Debug, Clone)]
pub enum HkdfError {
    InvalidSalt,
    InvalidIkm,
    OutputTooLong,
    InvalidLength,
}

impl fmt::Display for HkdfError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            HkdfError::InvalidSalt => write!(f, "Invalid salt"),
            HkdfError::InvalidIkm => write!(f, "Invalid input keying material"),
            HkdfError::OutputTooLong => write!(f, "Requested output length too long"),
            HkdfError::InvalidLength => write!(f, "Invalid length parameter"),
        }
    }
}

impl Error for HkdfError {}

/// HMAC-SHA3-512 implementation
///
/// Constant-time HMAC construction using SHA3-512.
fn hmac_sha3_512(key: &[u8], data: &[u8]) -> [u8; HASH_LENGTH] {
    const IPAD: u8 = 0x36;
    const OPAD: u8 = 0x5c;
    const BLOCK_SIZE: usize = 72;  // SHA3-512 rate
    
    // Prepare key (hash if too long, pad if too short)
    let mut padded_key = [0u8; BLOCK_SIZE];
    if key.len() > BLOCK_SIZE {
        let mut hasher = Sha3_512::new();
        hasher.update(key);
        let hashed: [u8; HASH_LENGTH] = hasher.finalize().into();
        padded_key[..HASH_LENGTH].copy_from_slice(&hashed);
    } else {
        padded_key[..key.len()].copy_from_slice(key);
    }
    
    // Inner hash: H((K ⊕ ipad) || data)
    let mut inner_hasher = Sha3_512::new();
    for byte in padded_key.iter() {
        inner_hasher.update(&[byte ^ IPAD]);
    }
    inner_hasher.update(data);
    let mut inner_hash: [u8; HASH_LENGTH] = inner_hasher.finalize().into();
    
    // Outer hash: H((K ⊕ opad) || inner_hash)
    let mut outer_hasher = Sha3_512::new();
    for byte in padded_key.iter() {
        outer_hasher.update(&[byte ^ OPAD]);
    }
    outer_hasher.update(&inner_hash);
    
    // Zeroize all sensitive intermediate data
    padded_key.zeroize();
    inner_hash.zeroize();
    
    outer_hasher.finalize().into()
}

/// HKDF Pseudorandom Key (PRK)
///
/// Output of the Extract phase, used as input to Expand.
/// Automatically zeroized on drop.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct Prk {
    data: [u8; HASH_LENGTH],
}

impl Prk {
    /// Get PRK bytes (use with caution)
    pub fn as_bytes(&self) -> &[u8; HASH_LENGTH] {
        &self.data
    }
}

/// HKDF-SHA3-512 Implementation
///
/// Provides Extract-then-Expand key derivation per RFC 5869.
///
/// ## Security Properties
/// - Input keying material (IKM) can be any length
/// - Salt should be random or application-specific
/// - Info provides domain separation
/// - All sensitive data zeroized on drop
#[derive(Clone)]
pub struct Hkdf {
    /// Pseudorandom key from Extract phase
    prk: [u8; HASH_LENGTH],
}

impl Hkdf {
    /// HKDF-Extract: Extract a pseudorandom key from input keying material
    ///
    /// # Arguments
    /// * `salt` - Optional salt (if None, uses zeros)
    /// * `ikm` - Input keying material
    ///
    /// # Returns
    /// HKDF instance ready for Expand operations
    ///
    /// # Security
    /// - Salt should be random or unique per application
    /// - IKM should contain sufficient entropy
    pub fn extract(salt: Option<&[u8]>, ikm: &[u8]) -> Self {
        // Use zero-salt if none provided (per RFC 5869)
        let salt = salt.unwrap_or(&[0u8; HASH_LENGTH]);
        
        // PRK = HMAC-Hash(salt, IKM)
        let prk = hmac_sha3_512(salt, ikm);
        
        Self { prk }
    }
    
    /// HKDF-Extract with explicit PRK return
    ///
    /// Returns the pseudorandom key for inspection or storage.
    pub fn extract_prk(salt: Option<&[u8]>, ikm: &[u8]) -> Prk {
        let hkdf = Self::extract(salt, ikm);
        Prk { data: hkdf.prk }
    }
    
    /// Create HKDF from existing PRK
    ///
    /// Useful when PRK is stored or transmitted separately.
    pub fn from_prk(prk: &[u8]) -> Result<Self, HkdfError> {
        if prk.len() != HASH_LENGTH {
            return Err(HkdfError::InvalidLength);
        }
        
        let mut prk_array = [0u8; HASH_LENGTH];
        prk_array.copy_from_slice(prk);
        
        Ok(Self { prk: prk_array })
    }
    
    /// HKDF-Expand: Expand PRK into output keying material
    ///
    /// # Arguments
    /// * `info` - Application-specific context/label
    /// * `length` - Desired output length in bytes
    ///
    /// # Returns
    /// Output keying material (OKM) of the specified length
    ///
    /// # Security
    /// - Info should be unique per derived key use
    /// - Maximum output length is 255 * HASH_LENGTH
    pub fn expand(&self, info: &[u8], length: usize) -> Result<Vec<u8>, HkdfError> {
        if length > MAX_OUTPUT_LENGTH {
            return Err(HkdfError::OutputTooLong);
        }
        
        if length == 0 {
            return Ok(Vec::new());
        }
        
        // N = ceil(L/HashLen)
        let n = (length + HASH_LENGTH - 1) / HASH_LENGTH;
        
        let mut okm = Vec::with_capacity(n * HASH_LENGTH);
        let mut t_prev: Vec<u8> = Vec::new();
        
        for i in 1..=n {
            // T(i) = HMAC-Hash(PRK, T(i-1) || info || i)
            let mut data = Vec::with_capacity(t_prev.len() + info.len() + 1);
            data.extend_from_slice(&t_prev);
            data.extend_from_slice(info);
            data.push(i as u8);
            
            let t_i = hmac_sha3_512(&self.prk, &data);
            okm.extend_from_slice(&t_i);
            
            t_prev = t_i.to_vec();
            
            // Zeroize intermediate data
            data.zeroize();
        }
        
        t_prev.zeroize();
        okm.truncate(length);
        
        Ok(okm)
    }
    
    /// Expand into a fixed-size array
    ///
    /// Convenience method for fixed-length key derivation.
    pub fn expand_fixed<const N: usize>(&self, info: &[u8]) -> Result<[u8; N], HkdfError> {
        if N > MAX_OUTPUT_LENGTH {
            return Err(HkdfError::OutputTooLong);
        }
        
        let okm = self.expand(info, N)?;
        let mut result = [0u8; N];
        result.copy_from_slice(&okm);
        
        Ok(result)
    }
}

impl Drop for Hkdf {
    fn drop(&mut self) {
        self.prk.zeroize();
    }
}

/// One-shot HKDF derivation
///
/// Convenience function for simple key derivation.
///
/// # Arguments
/// * `salt` - Optional salt (random or application-specific)
/// * `ikm` - Input keying material
/// * `info` - Application-specific context
/// * `length` - Desired output length
pub fn derive(
    salt: Option<&[u8]>,
    ikm: &[u8],
    info: &[u8],
    length: usize,
) -> Result<Vec<u8>, HkdfError> {
    let hkdf = Hkdf::extract(salt, ikm);
    hkdf.expand(info, length)
}

/// Derive a fixed-size key
///
/// Convenience function for fixed-length derivation.
pub fn derive_fixed<const N: usize>(
    salt: Option<&[u8]>,
    ikm: &[u8],
    info: &[u8],
) -> Result<[u8; N], HkdfError> {
    let hkdf = Hkdf::extract(salt, ikm);
    hkdf.expand_fixed(info)
}

/// Labeled HKDF for domain separation
///
/// Provides structured info parameter with protocol label.
///
/// # Format
/// info = length(2) || label || context
pub fn derive_labeled(
    salt: Option<&[u8]>,
    ikm: &[u8],
    label: &str,
    context: &[u8],
    length: usize,
) -> Result<Vec<u8>, HkdfError> {
    // Build labeled info: len || label || context
    let mut info = Vec::new();
    info.extend_from_slice(&(length as u16).to_be_bytes());
    info.extend_from_slice(label.as_bytes());
    info.extend_from_slice(context);
    
    derive(salt, ikm, &info, length)
}

/// Key schedule derivation for encryption/MAC keys
///
/// Derives separate keys for encryption and authentication
/// from a single master secret.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct KeySchedule {
    pub encryption_key: Vec<u8>,
    pub mac_key: Vec<u8>,
    pub iv: Vec<u8>,
}

impl KeySchedule {
    /// Derive encryption and MAC keys from master secret
    ///
    /// # Arguments
    /// * `master_secret` - Input keying material
    /// * `salt` - Optional salt
    /// * `context` - Application context
    /// * `enc_key_len` - Encryption key length
    /// * `mac_key_len` - MAC key length
    /// * `iv_len` - IV length
    pub fn derive(
        master_secret: &[u8],
        salt: Option<&[u8]>,
        context: &[u8],
        enc_key_len: usize,
        mac_key_len: usize,
        iv_len: usize,
    ) -> Result<Self, HkdfError> {
        let hkdf = Hkdf::extract(salt, master_secret);
        
        let mut enc_info = b"encryption".to_vec();
        enc_info.extend_from_slice(context);
        let encryption_key = hkdf.expand(&enc_info, enc_key_len)?;
        
        let mut mac_info = b"authentication".to_vec();
        mac_info.extend_from_slice(context);
        let mac_key = hkdf.expand(&mac_info, mac_key_len)?;
        
        let mut iv_info = b"initialization".to_vec();
        iv_info.extend_from_slice(context);
        let iv = hkdf.expand(&iv_info, iv_len)?;
        
        enc_info.zeroize();
        mac_info.zeroize();
        iv_info.zeroize();
        
        Ok(Self {
            encryption_key,
            mac_key,
            iv,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_extract_expand() {
        let ikm = b"input keying material";
        let salt = b"random salt";
        let info = b"application context";
        
        let hkdf = Hkdf::extract(Some(salt), ikm);
        let okm = hkdf.expand(info, 32).unwrap();
        
        assert_eq!(okm.len(), 32);
        // Should be deterministic
        let okm2 = hkdf.expand(info, 32).unwrap();
        assert_eq!(okm, okm2);
    }
    
    #[test]
    fn test_derive_convenience() {
        let ikm = b"master secret";
        let okm = derive(None, ikm, b"test", 64).unwrap();
        
        assert_eq!(okm.len(), 64);
    }
    
    #[test]
    fn test_derive_fixed() {
        let ikm = b"master secret";
        let okm: [u8; 32] = derive_fixed(None, ikm, b"test").unwrap();
        
        assert_eq!(okm.len(), 32);
    }
    
    #[test]
    fn test_output_too_long() {
        let ikm = b"master secret";
        let result = derive(None, ikm, b"test", MAX_OUTPUT_LENGTH + 1);
        
        assert!(matches!(result, Err(HkdfError::OutputTooLong)));
    }
    
    #[test]
    fn test_different_info_different_output() {
        let ikm = b"master secret";
        
        let okm1 = derive(None, ikm, b"key1", 32).unwrap();
        let okm2 = derive(None, ikm, b"key2", 32).unwrap();
        
        assert_ne!(okm1, okm2);
    }
    
    #[test]
    fn test_key_schedule() {
        let master = b"master secret key";
        
        let schedule = KeySchedule::derive(
            master,
            None,
            b"test context",
            32,  // enc key
            32,  // mac key
            16,  // iv
        ).unwrap();
        
        assert_eq!(schedule.encryption_key.len(), 32);
        assert_eq!(schedule.mac_key.len(), 32);
        assert_eq!(schedule.iv.len(), 16);
        
        // Keys should all be different
        assert_ne!(schedule.encryption_key, schedule.mac_key);
    }
    
    #[test]
    fn test_labeled_derivation() {
        let ikm = b"input key";
        
        let okm = derive_labeled(
            None,
            ikm,
            "QRATUM-KEY",
            b"context",
            32,
        ).unwrap();
        
        assert_eq!(okm.len(), 32);
    }
    
    #[test]
    fn test_prk_extraction() {
        let ikm = b"input keying material";
        let salt = b"salt";
        
        let prk = Hkdf::extract_prk(Some(salt), ikm);
        assert_eq!(prk.as_bytes().len(), HASH_LENGTH);
        
        // Can reconstruct HKDF from PRK
        let hkdf = Hkdf::from_prk(prk.as_bytes()).unwrap();
        let okm = hkdf.expand(b"info", 32).unwrap();
        assert_eq!(okm.len(), 32);
    }
}
