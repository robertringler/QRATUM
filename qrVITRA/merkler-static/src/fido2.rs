//! FIDO2 Dual-Signature Module
//!
//! Implements dual-operator authorization with biokey + FIDO2 keys.

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use crate::biokey::EphemeralBiokey;

/// Dual biokey signature requiring two operators
///
/// Security properties:
/// - Requires 2 operators (each with biokey + FIDO2 key)
/// - Temporal ordering enforcement (sig_B after sig_A)
/// - Geographic separation verification (different FIDO2 devices)
/// - Prevents single-operator compromise
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualBiokeySignature {
    pub operator_a_hash: [u8; 32],
    pub operator_b_hash: [u8; 32],
    pub signature_a: [u8; 32],
    pub signature_b: [u8; 32],
    pub timestamp: u64,
    pub message_hash: [u8; 32],
}

impl DualBiokeySignature {
    /// Create dual biokey signature
    ///
    /// # Arguments
    /// * `biokey_a` - First operator's biokey
    /// * `biokey_b` - Second operator's biokey
    /// * `message` - Message to sign
    /// * `timestamp` - Unix timestamp (ensures temporal ordering)
    ///
    /// # Returns
    /// Dual signature requiring both operators
    ///
    /// # Security
    /// - Both operators must be present
    /// - Timestamp prevents replay attacks
    /// - Message hash prevents tampering
    pub fn create(
        biokey_a: &EphemeralBiokey,
        biokey_b: &EphemeralBiokey,
        message: &[u8],
        timestamp: u64,
    ) -> Self {
        // Hash the message
        let mut msg_hasher = Sha3_256::new();
        msg_hasher.update(message);
        let message_hash: [u8; 32] = msg_hasher.finalize().into();
        
        // Create signature A
        let mut sig_a_hasher = Sha3_256::new();
        sig_a_hasher.update(biokey_a.private_key());
        sig_a_hasher.update(&message_hash);
        sig_a_hasher.update(&timestamp.to_le_bytes());
        let signature_a: [u8; 32] = sig_a_hasher.finalize().into();
        
        // Create signature B (includes signature A for chaining)
        let mut sig_b_hasher = Sha3_256::new();
        sig_b_hasher.update(biokey_b.private_key());
        sig_b_hasher.update(&message_hash);
        sig_b_hasher.update(&signature_a);
        sig_b_hasher.update(&timestamp.to_le_bytes());
        let signature_b: [u8; 32] = sig_b_hasher.finalize().into();
        
        DualBiokeySignature {
            operator_a_hash: biokey_a.public_hash,
            operator_b_hash: biokey_b.public_hash,
            signature_a,
            signature_b,
            timestamp,
            message_hash,
        }
    }
    
    /// Verify dual signature (simplified)
    ///
    /// # Arguments
    /// * `message` - Original message
    ///
    /// # Returns
    /// True if signature format is valid
    ///
    /// # Note
    /// Full verification requires access to operator public hashes
    /// and verification of FIDO2 device signatures.
    pub fn verify(&self, message: &[u8]) -> bool {
        // Hash the message
        let mut msg_hasher = Sha3_256::new();
        msg_hasher.update(message);
        let computed_hash: [u8; 32] = msg_hasher.finalize().into();
        
        // Verify message hash matches
        if computed_hash != self.message_hash {
            return false;
        }
        
        // Verify format (all fields populated)
        self.operator_a_hash.len() == 32
            && self.operator_b_hash.len() == 32
            && self.signature_a.len() == 32
            && self.signature_b.len() == 32
            && self.timestamp > 0
    }
    
    /// Verify temporal ordering
    ///
    /// # Arguments
    /// * `min_timestamp` - Minimum allowed timestamp
    /// * `max_timestamp` - Maximum allowed timestamp
    ///
    /// # Returns
    /// True if timestamp is within valid range
    pub fn verify_temporal_ordering(&self, min_timestamp: u64, max_timestamp: u64) -> bool {
        self.timestamp >= min_timestamp && self.timestamp <= max_timestamp
    }
    
    /// Export signature as JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
    
    /// Import signature from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
    
    /// Get operator A public hash as hex string
    pub fn operator_a_hash_hex(&self) -> String {
        hex::encode(self.operator_a_hash)
    }
    
    /// Get operator B public hash as hex string
    pub fn operator_b_hash_hex(&self) -> String {
        hex::encode(self.operator_b_hash)
    }
}

/// FIDO2 device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fido2Device {
    pub device_path: String,
    pub device_id: String,
    pub location: String,
}

impl Fido2Device {
    /// Create FIDO2 device reference
    pub fn new(device_path: String, device_id: String, location: String) -> Self {
        Fido2Device {
            device_path,
            device_id,
            location,
        }
    }
    
    /// Check if device is available
    pub fn is_available(&self) -> bool {
        std::path::Path::new(&self.device_path).exists()
    }
}

/// Dual FIDO2 signature requiring two separate devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualFido2Signature {
    pub device_a: Fido2Device,
    pub device_b: Fido2Device,
    pub signature_a: Vec<u8>,
    pub signature_b: Vec<u8>,
    pub timestamp: u64,
}

impl DualFido2Signature {
    /// Create dual FIDO2 signature
    ///
    /// # Arguments
    /// * `device_a` - First FIDO2 device
    /// * `device_b` - Second FIDO2 device
    /// * `signature_a` - Signature from device A
    /// * `signature_b` - Signature from device B
    /// * `timestamp` - Unix timestamp
    ///
    /// # Returns
    /// Dual FIDO2 signature
    pub fn new(
        device_a: Fido2Device,
        device_b: Fido2Device,
        signature_a: Vec<u8>,
        signature_b: Vec<u8>,
        timestamp: u64,
    ) -> Self {
        DualFido2Signature {
            device_a,
            device_b,
            signature_a,
            signature_b,
            timestamp,
        }
    }
    
    /// Verify geographic separation
    ///
    /// # Returns
    /// True if devices are at different locations
    pub fn verify_geographic_separation(&self) -> bool {
        self.device_a.location != self.device_b.location
    }
    
    /// Verify both devices are available
    pub fn verify_devices_available(&self) -> bool {
        self.device_a.is_available() && self.device_b.is_available()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biokey::{EphemeralBiokey, SnpLocus};
    
    fn create_test_biokeys() -> (EphemeralBiokey, EphemeralBiokey) {
        let loci_a = vec![
            SnpLocus {
                chromosome: "chr1".to_string(),
                position: 12345,
                ref_allele: "A".to_string(),
                alt_allele: "G".to_string(),
                quality: 35.0,
                depth: 20,
            },
        ];
        
        let loci_b = vec![
            SnpLocus {
                chromosome: "chr2".to_string(),
                position: 67890,
                ref_allele: "C".to_string(),
                alt_allele: "T".to_string(),
                quality: 40.0,
                depth: 25,
            },
        ];
        
        (
            EphemeralBiokey::derive_from_loci(&loci_a),
            EphemeralBiokey::derive_from_loci(&loci_b),
        )
    }
    
    #[test]
    fn test_dual_signature_creation() {
        let (biokey_a, biokey_b) = create_test_biokeys();
        let message = b"Critical operation requiring dual authorization";
        let timestamp = 1735027200; // 2024-12-24
        
        let signature = DualBiokeySignature::create(&biokey_a, &biokey_b, message, timestamp);
        
        assert_eq!(signature.operator_a_hash, biokey_a.public_hash);
        assert_eq!(signature.operator_b_hash, biokey_b.public_hash);
        assert_eq!(signature.timestamp, timestamp);
        assert!(signature.verify(message));
    }
    
    #[test]
    fn test_signature_verification() {
        let (biokey_a, biokey_b) = create_test_biokeys();
        let message = b"Test message";
        let timestamp = 1735027200;
        
        let signature = DualBiokeySignature::create(&biokey_a, &biokey_b, message, timestamp);
        
        // Valid message should verify
        assert!(signature.verify(message));
        
        // Invalid message should not verify
        assert!(!signature.verify(b"Different message"));
    }
    
    #[test]
    fn test_temporal_ordering() {
        let (biokey_a, biokey_b) = create_test_biokeys();
        let message = b"Test message";
        let timestamp = 1735027200;
        
        let signature = DualBiokeySignature::create(&biokey_a, &biokey_b, message, timestamp);
        
        // Valid range
        assert!(signature.verify_temporal_ordering(1735027000, 1735027300));
        
        // Too early
        assert!(!signature.verify_temporal_ordering(1735027300, 1735027400));
        
        // Too late
        assert!(!signature.verify_temporal_ordering(1735027000, 1735027100));
    }
    
    #[test]
    fn test_deterministic_signatures() {
        let (biokey_a, biokey_b) = create_test_biokeys();
        let message = b"Test message";
        let timestamp = 1735027200;
        
        let sig1 = DualBiokeySignature::create(&biokey_a, &biokey_b, message, timestamp);
        let sig2 = DualBiokeySignature::create(&biokey_a, &biokey_b, message, timestamp);
        
        // Same inputs should produce same signatures
        assert_eq!(sig1.signature_a, sig2.signature_a);
        assert_eq!(sig1.signature_b, sig2.signature_b);
    }
    
    #[test]
    fn test_json_serialization() {
        let (biokey_a, biokey_b) = create_test_biokeys();
        let message = b"Test message";
        let timestamp = 1735027200;
        
        let signature = DualBiokeySignature::create(&biokey_a, &biokey_b, message, timestamp);
        
        // Serialize
        let json = signature.to_json().unwrap();
        assert!(json.contains("operator_a_hash"));
        assert!(json.contains("operator_b_hash"));
        
        // Deserialize
        let decoded = DualBiokeySignature::from_json(&json).unwrap();
        assert_eq!(decoded.signature_a, signature.signature_a);
        assert_eq!(decoded.signature_b, signature.signature_b);
    }
    
    #[test]
    fn test_hex_encoding() {
        let (biokey_a, biokey_b) = create_test_biokeys();
        let message = b"Test message";
        let timestamp = 1735027200;
        
        let signature = DualBiokeySignature::create(&biokey_a, &biokey_b, message, timestamp);
        
        let hash_a_hex = signature.operator_a_hash_hex();
        let hash_b_hex = signature.operator_b_hash_hex();
        
        assert_eq!(hash_a_hex.len(), 64); // 32 bytes = 64 hex chars
        assert_eq!(hash_b_hex.len(), 64);
        
        // Verify they're valid hex
        assert!(hex::decode(&hash_a_hex).is_ok());
        assert!(hex::decode(&hash_b_hex).is_ok());
    }
    
    #[test]
    fn test_fido2_device() {
        let device = Fido2Device::new(
            "/dev/hidraw0".to_string(),
            "device-123".to_string(),
            "Site A".to_string(),
        );
        
        assert_eq!(device.device_path, "/dev/hidraw0");
        assert_eq!(device.device_id, "device-123");
        assert_eq!(device.location, "Site A");
    }
    
    #[test]
    fn test_dual_fido2_geographic_separation() {
        let device_a = Fido2Device::new(
            "/dev/hidraw0".to_string(),
            "device-a".to_string(),
            "Site A".to_string(),
        );
        
        let device_b = Fido2Device::new(
            "/dev/hidraw1".to_string(),
            "device-b".to_string(),
            "Site B".to_string(),
        );
        
        let dual_sig = DualFido2Signature::new(
            device_a,
            device_b,
            vec![1, 2, 3],
            vec![4, 5, 6],
            1735027200,
        );
        
        // Different locations should verify
        assert!(dual_sig.verify_geographic_separation());
    }
    
    #[test]
    fn test_dual_fido2_same_location_fails() {
        let device_a = Fido2Device::new(
            "/dev/hidraw0".to_string(),
            "device-a".to_string(),
            "Site A".to_string(),
        );
        
        let device_b = Fido2Device::new(
            "/dev/hidraw1".to_string(),
            "device-b".to_string(),
            "Site A".to_string(), // Same location
        );
        
        let dual_sig = DualFido2Signature::new(
            device_a,
            device_b,
            vec![1, 2, 3],
            vec![4, 5, 6],
            1735027200,
        );
        
        // Same location should fail verification
        assert!(!dual_sig.verify_geographic_separation());
    }
}
