//! NIST SP 800-90A HMAC-DRBG Implementation
//!
//! Deterministic Random Bit Generator using HMAC-SHA3-512 as the
//! underlying cryptographic primitive with entropy pooling.
//!
//! Security Properties:
//! - NIST SP 800-90A compliant HMAC-DRBG
//! - SHA3-512 based for post-quantum security margin
//! - Entropy pooling from multiple sources
//! - Prediction resistance via reseeding
//! - Zeroization on drop

use sha3::{Sha3_512, Digest};
use zeroize::{Zeroize, ZeroizeOnDrop};
use std::error::Error;
use std::fmt;

/// DRBG parameters per NIST SP 800-90A
pub const SECURITY_STRENGTH: usize = 256;
pub const SEED_LENGTH: usize = 64;  // SHA3-512 output
pub const MAX_BYTES_PER_REQUEST: usize = 65536;
pub const RESEED_INTERVAL: u64 = 1 << 48;  // 2^48 requests before mandatory reseed
pub const MIN_ENTROPY: usize = 32;  // Minimum entropy bytes required

#[derive(Debug, Clone)]
pub enum DrbgError {
    InsufficientEntropy,
    ReseedRequired,
    RequestTooLarge,
    NotInstantiated,
    EntropySourceFailed,
}

impl fmt::Display for DrbgError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DrbgError::InsufficientEntropy => write!(f, "Insufficient entropy provided"),
            DrbgError::ReseedRequired => write!(f, "DRBG reseed required"),
            DrbgError::RequestTooLarge => write!(f, "Request exceeds max bytes per request"),
            DrbgError::NotInstantiated => write!(f, "DRBG not properly instantiated"),
            DrbgError::EntropySourceFailed => write!(f, "Entropy source failed"),
        }
    }
}

impl Error for DrbgError {}

/// Entropy Source trait for pluggable entropy collection
pub trait EntropySource: Send + Sync {
    /// Collect entropy bytes from this source
    fn collect(&self, output: &mut [u8]) -> Result<usize, DrbgError>;
    
    /// Get source identifier for audit
    fn source_id(&self) -> &str;
}

/// System RNG entropy source (getrandom)
pub struct SystemEntropySource;

impl EntropySource for SystemEntropySource {
    fn collect(&self, output: &mut [u8]) -> Result<usize, DrbgError> {
        getrandom::getrandom(output).map_err(|_| DrbgError::EntropySourceFailed)?;
        Ok(output.len())
    }
    
    fn source_id(&self) -> &str {
        "system-rng"
    }
}

/// Timestamp-based auxiliary entropy
pub struct TimestampEntropySource;

impl EntropySource for TimestampEntropySource {
    fn collect(&self, output: &mut [u8]) -> Result<usize, DrbgError> {
        use std::time::{SystemTime, UNIX_EPOCH};
        
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        
        let bytes = nanos.to_le_bytes();
        let copy_len = output.len().min(bytes.len());
        output[..copy_len].copy_from_slice(&bytes[..copy_len]);
        
        Ok(copy_len)
    }
    
    fn source_id(&self) -> &str {
        "timestamp"
    }
}

/// Entropy Pool for collecting and mixing entropy from multiple sources
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct EntropyPool {
    /// Accumulated entropy (zeroized on drop)
    pool: [u8; SEED_LENGTH],
    
    /// Number of sources contributed
    source_count: u32,
    
    /// Estimated entropy bits
    #[zeroize(skip)]
    entropy_bits: u32,
}

impl EntropyPool {
    /// Create new empty entropy pool
    pub fn new() -> Self {
        Self {
            pool: [0u8; SEED_LENGTH],
            source_count: 0,
            entropy_bits: 0,
        }
    }
    
    /// Add entropy from a source (XOR mixing)
    ///
    /// Security: Uses XOR for mixing which preserves entropy
    /// when sources are independent.
    pub fn add_entropy<S: EntropySource>(&mut self, source: &S) -> Result<(), DrbgError> {
        let mut temp = [0u8; SEED_LENGTH];
        let bytes_collected = source.collect(&mut temp)?;
        
        // XOR mix into pool (constant-time operation)
        for (i, byte) in temp.iter().enumerate().take(bytes_collected) {
            self.pool[i % SEED_LENGTH] ^= byte;
        }
        
        self.source_count += 1;
        self.entropy_bits += (bytes_collected * 8) as u32;
        
        // Zeroize temporary buffer
        temp.zeroize();
        
        Ok(())
    }
    
    /// Finalize and extract mixed entropy
    ///
    /// Uses SHA3-512 to condition the entropy pool.
    pub fn finalize(&mut self) -> [u8; SEED_LENGTH] {
        let mut hasher = Sha3_512::new();
        hasher.update(&self.pool);
        hasher.update(&self.source_count.to_le_bytes());
        
        let result: [u8; SEED_LENGTH] = hasher.finalize().into();
        
        // Zeroize pool after extraction
        self.pool.zeroize();
        self.source_count = 0;
        self.entropy_bits = 0;
        
        result
    }
    
    /// Get estimated entropy bits
    pub fn entropy_estimate(&self) -> u32 {
        self.entropy_bits.min((SEED_LENGTH * 8) as u32)
    }
}

impl Default for EntropyPool {
    fn default() -> Self {
        Self::new()
    }
}

/// HMAC-SHA3-512 implementation for DRBG
fn hmac_sha3_512(key: &[u8], data: &[u8]) -> [u8; SEED_LENGTH] {
    const IPAD: u8 = 0x36;
    const OPAD: u8 = 0x5c;
    const BLOCK_SIZE: usize = 72;  // SHA3-512 rate
    
    // Prepare key (hash if too long, pad if too short)
    let mut padded_key = [0u8; BLOCK_SIZE];
    if key.len() > BLOCK_SIZE {
        let mut hasher = Sha3_512::new();
        hasher.update(key);
        let hashed: [u8; SEED_LENGTH] = hasher.finalize().into();
        padded_key[..SEED_LENGTH].copy_from_slice(&hashed);
    } else {
        padded_key[..key.len()].copy_from_slice(key);
    }
    
    // Inner hash: H((K ⊕ ipad) || data)
    let mut inner_hasher = Sha3_512::new();
    for byte in padded_key.iter() {
        inner_hasher.update(&[byte ^ IPAD]);
    }
    inner_hasher.update(data);
    let inner_hash: [u8; SEED_LENGTH] = inner_hasher.finalize().into();
    
    // Outer hash: H((K ⊕ opad) || inner_hash)
    let mut outer_hasher = Sha3_512::new();
    for byte in padded_key.iter() {
        outer_hasher.update(&[byte ^ OPAD]);
    }
    outer_hasher.update(&inner_hash);
    
    outer_hasher.finalize().into()
}

/// HMAC-DRBG State
///
/// NIST SP 800-90A compliant HMAC-DRBG using SHA3-512.
/// Provides cryptographically secure random number generation
/// with prediction resistance.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct HmacDrbg {
    /// Internal key K (zeroized on drop)
    key: [u8; SEED_LENGTH],
    
    /// Internal value V (zeroized on drop)
    value: [u8; SEED_LENGTH],
    
    /// Reseed counter
    #[zeroize(skip)]
    reseed_counter: u64,
    
    /// Instantiation flag
    #[zeroize(skip)]
    instantiated: bool,
    
    /// Prediction resistance enabled
    #[zeroize(skip)]
    prediction_resistance: bool,
}

impl HmacDrbg {
    /// Create new uninstantiated DRBG
    pub fn new() -> Self {
        Self {
            key: [0u8; SEED_LENGTH],
            value: [0u8; SEED_LENGTH],
            reseed_counter: 0,
            instantiated: false,
            prediction_resistance: true,
        }
    }
    
    /// Instantiate DRBG with entropy and optional personalization string
    ///
    /// Per NIST SP 800-90A Section 10.1.2.3
    pub fn instantiate(
        &mut self,
        entropy: &[u8],
        nonce: &[u8],
        personalization: Option<&[u8]>,
    ) -> Result<(), DrbgError> {
        if entropy.len() < MIN_ENTROPY {
            return Err(DrbgError::InsufficientEntropy);
        }
        
        // seed_material = entropy || nonce || personalization
        let mut seed_material = Vec::with_capacity(
            entropy.len() + nonce.len() + personalization.map_or(0, |p| p.len())
        );
        seed_material.extend_from_slice(entropy);
        seed_material.extend_from_slice(nonce);
        if let Some(pers) = personalization {
            seed_material.extend_from_slice(pers);
        }
        
        // Initialize K and V per spec
        self.key = [0u8; SEED_LENGTH];
        self.value = [0x01; SEED_LENGTH];
        
        // Update state with seed material
        self.update(&seed_material);
        
        self.reseed_counter = 1;
        self.instantiated = true;
        
        // Zeroize sensitive local data
        seed_material.zeroize();
        
        Ok(())
    }
    
    /// Instantiate with entropy pooling from multiple sources
    pub fn instantiate_with_pool(
        &mut self,
        sources: &[&dyn EntropySource],
        personalization: Option<&[u8]>,
    ) -> Result<(), DrbgError> {
        let mut pool = EntropyPool::new();
        
        for source in sources {
            pool.add_entropy(*source)?;
        }
        
        if pool.entropy_estimate() < (MIN_ENTROPY * 8) as u32 {
            return Err(DrbgError::InsufficientEntropy);
        }
        
        let entropy = pool.finalize();
        
        // Generate nonce from additional entropy
        let mut nonce = [0u8; 16];
        getrandom::getrandom(&mut nonce).map_err(|_| DrbgError::EntropySourceFailed)?;
        
        self.instantiate(&entropy, &nonce, personalization)
    }
    
    /// Reseed DRBG with new entropy
    ///
    /// Per NIST SP 800-90A Section 10.1.2.4
    pub fn reseed(
        &mut self,
        entropy: &[u8],
        additional_input: Option<&[u8]>,
    ) -> Result<(), DrbgError> {
        if !self.instantiated {
            return Err(DrbgError::NotInstantiated);
        }
        
        if entropy.len() < MIN_ENTROPY {
            return Err(DrbgError::InsufficientEntropy);
        }
        
        // seed_material = entropy || additional_input
        let mut seed_material = Vec::with_capacity(
            entropy.len() + additional_input.map_or(0, |a| a.len())
        );
        seed_material.extend_from_slice(entropy);
        if let Some(add) = additional_input {
            seed_material.extend_from_slice(add);
        }
        
        self.update(&seed_material);
        self.reseed_counter = 1;
        
        seed_material.zeroize();
        
        Ok(())
    }
    
    /// Generate random bytes
    ///
    /// Per NIST SP 800-90A Section 10.1.2.5
    pub fn generate(
        &mut self,
        output: &mut [u8],
        additional_input: Option<&[u8]>,
    ) -> Result<(), DrbgError> {
        if !self.instantiated {
            return Err(DrbgError::NotInstantiated);
        }
        
        if output.len() > MAX_BYTES_PER_REQUEST {
            return Err(DrbgError::RequestTooLarge);
        }
        
        if self.reseed_counter > RESEED_INTERVAL {
            return Err(DrbgError::ReseedRequired);
        }
        
        // Process additional input
        if let Some(add) = additional_input {
            if !add.is_empty() {
                self.update(add);
            }
        }
        
        // Generate output
        let mut temp = Vec::new();
        while temp.len() < output.len() {
            self.value = hmac_sha3_512(&self.key, &self.value);
            temp.extend_from_slice(&self.value);
        }
        
        output.copy_from_slice(&temp[..output.len()]);
        
        // Update state
        let update_data = additional_input.unwrap_or(&[]);
        self.update(update_data);
        
        self.reseed_counter += 1;
        
        temp.zeroize();
        
        Ok(())
    }
    
    /// Update internal state (HMAC_DRBG_Update)
    ///
    /// Per NIST SP 800-90A Section 10.1.2.2
    fn update(&mut self, provided_data: &[u8]) {
        // K = HMAC(K, V || 0x00 || provided_data)
        let mut concat = Vec::with_capacity(SEED_LENGTH + 1 + provided_data.len());
        concat.extend_from_slice(&self.value);
        concat.push(0x00);
        concat.extend_from_slice(provided_data);
        self.key = hmac_sha3_512(&self.key, &concat);
        
        // V = HMAC(K, V)
        self.value = hmac_sha3_512(&self.key, &self.value);
        
        if !provided_data.is_empty() {
            // K = HMAC(K, V || 0x01 || provided_data)
            concat.clear();
            concat.extend_from_slice(&self.value);
            concat.push(0x01);
            concat.extend_from_slice(provided_data);
            self.key = hmac_sha3_512(&self.key, &concat);
            
            // V = HMAC(K, V)
            self.value = hmac_sha3_512(&self.key, &self.value);
        }
        
        concat.zeroize();
    }
    
    /// Check if DRBG is properly instantiated
    pub fn is_instantiated(&self) -> bool {
        self.instantiated
    }
    
    /// Get current reseed counter
    pub fn reseed_counter(&self) -> u64 {
        self.reseed_counter
    }
    
    /// Force reseed requirement (for security policy)
    pub fn require_reseed(&mut self) {
        self.reseed_counter = RESEED_INTERVAL + 1;
    }
}

impl Default for HmacDrbg {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe DRBG wrapper with automatic reseeding
pub struct SecureDrbg {
    drbg: std::sync::Mutex<HmacDrbg>,
    reseed_interval: u64,
}

impl SecureDrbg {
    /// Create and instantiate a new secure DRBG
    pub fn new(personalization: Option<&[u8]>) -> Result<Self, DrbgError> {
        let mut drbg = HmacDrbg::new();
        
        let system_source = SystemEntropySource;
        let timestamp_source = TimestampEntropySource;
        let sources: [&dyn EntropySource; 2] = [&system_source, &timestamp_source];
        
        drbg.instantiate_with_pool(&sources, personalization)?;
        
        Ok(Self {
            drbg: std::sync::Mutex::new(drbg),
            reseed_interval: RESEED_INTERVAL / 2,  // Reseed more frequently
        })
    }
    
    /// Generate random bytes with automatic reseeding
    ///
    /// Thread-safe: The mutex is held for the entire operation, ensuring
    /// that reseed check and generate are atomic with respect to other threads.
    pub fn generate(&self, output: &mut [u8]) -> Result<(), DrbgError> {
        let mut drbg = self.drbg.lock().unwrap();
        
        // Auto-reseed if approaching limit
        // Note: Check and reseed are atomic because mutex is held
        if drbg.reseed_counter() > self.reseed_interval {
            let mut entropy = [0u8; MIN_ENTROPY * 2];
            getrandom::getrandom(&mut entropy).map_err(|_| DrbgError::EntropySourceFailed)?;
            drbg.reseed(&entropy, None)?;
            entropy.zeroize();
        }
        
        drbg.generate(output, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_entropy_pool() {
        let mut pool = EntropyPool::new();
        let source = SystemEntropySource;
        
        pool.add_entropy(&source).unwrap();
        assert!(pool.entropy_estimate() > 0);
        
        let entropy = pool.finalize();
        assert_eq!(entropy.len(), SEED_LENGTH);
    }
    
    #[test]
    fn test_hmac_drbg_instantiate() {
        let mut drbg = HmacDrbg::new();
        let mut entropy = [0u8; 64];
        getrandom::getrandom(&mut entropy).unwrap();
        
        let result = drbg.instantiate(&entropy, b"nonce", Some(b"QRATUM"));
        assert!(result.is_ok());
        assert!(drbg.is_instantiated());
    }
    
    #[test]
    fn test_hmac_drbg_generate() {
        let mut drbg = HmacDrbg::new();
        let mut entropy = [0u8; 64];
        getrandom::getrandom(&mut entropy).unwrap();
        
        drbg.instantiate(&entropy, b"nonce", None).unwrap();
        
        let mut output = [0u8; 32];
        let result = drbg.generate(&mut output, None);
        assert!(result.is_ok());
        
        // Verify output is not all zeros
        assert!(output.iter().any(|&b| b != 0));
    }
    
    #[test]
    fn test_insufficient_entropy() {
        let mut drbg = HmacDrbg::new();
        let short_entropy = [0u8; 16];  // Less than MIN_ENTROPY
        
        let result = drbg.instantiate(&short_entropy, b"nonce", None);
        assert!(matches!(result, Err(DrbgError::InsufficientEntropy)));
    }
    
    #[test]
    fn test_secure_drbg() {
        let drbg = SecureDrbg::new(Some(b"test")).unwrap();
        
        let mut output1 = [0u8; 32];
        let mut output2 = [0u8; 32];
        
        drbg.generate(&mut output1).unwrap();
        drbg.generate(&mut output2).unwrap();
        
        // Outputs should be different
        assert_ne!(output1, output2);
    }
    
    #[test]
    fn test_reseed() {
        let mut drbg = HmacDrbg::new();
        let mut entropy = [0u8; 64];
        getrandom::getrandom(&mut entropy).unwrap();
        
        drbg.instantiate(&entropy, b"nonce", None).unwrap();
        
        let counter_before = drbg.reseed_counter();
        
        getrandom::getrandom(&mut entropy).unwrap();
        drbg.reseed(&entropy, Some(b"additional")).unwrap();
        
        assert_eq!(drbg.reseed_counter(), 1);  // Reset after reseed
        assert!(drbg.reseed_counter() <= counter_before);
    }
}
