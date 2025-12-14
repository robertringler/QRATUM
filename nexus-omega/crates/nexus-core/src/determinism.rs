//! Deterministic primitives for reproducible execution

use rand::RngCore;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;
use sha3::{Digest, Sha3_256};

/// Deterministic random number generator using ChaCha20
#[derive(Debug, Clone)]
pub struct DeterministicRng {
    rng: ChaCha20Rng,
}

impl DeterministicRng {
    /// Create a new deterministic RNG with the given seed
    pub fn new(seed: u64) -> Self {
        let rng = ChaCha20Rng::seed_from_u64(seed);
        Self { rng }
    }

    /// Generate the next random u64
    pub fn next_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }

    /// Generate the next random u32
    pub fn next_u32(&mut self) -> u32 {
        self.rng.next_u32()
    }

    /// Fill a byte slice with random data
    pub fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.rng.fill_bytes(dest);
    }
}

/// Deterministic hash using SHA3-256
#[derive(Debug, Clone)]
pub struct DeterministicHash {
    hasher: Sha3_256,
}

impl DeterministicHash {
    /// Create a new deterministic hasher
    pub fn new() -> Self {
        Self {
            hasher: Sha3_256::new(),
        }
    }

    /// Update the hasher with data
    pub fn update(&mut self, data: &[u8]) {
        self.hasher.update(data);
    }

    /// Finalize and return the hash
    pub fn finalize(self) -> [u8; 32] {
        let result = self.hasher.finalize();
        result.into()
    }

    /// Compute hash of data in one call
    pub fn hash(data: &[u8]) -> [u8; 32] {
        let mut hasher = Self::new();
        hasher.update(data);
        hasher.finalize()
    }
}

impl Default for DeterministicHash {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_rng() {
        let mut rng1 = DeterministicRng::new(42);
        let mut rng2 = DeterministicRng::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_deterministic_hash() {
        let data = b"test data";
        let hash1 = DeterministicHash::hash(data);
        let hash2 = DeterministicHash::hash(data);

        assert_eq!(hash1, hash2);
    }
}
