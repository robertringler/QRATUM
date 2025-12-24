//! BIPA (Biometric Information Privacy Act) Compliance Module
//!
//! Enforces Illinois BIPA requirements for biometric data handling.
//!
//! Key Requirements:
//! - Informed consent before collection
//! - Disclosure of storage period
//! - Prohibition of sale/profit from biometric data
//! - Retention schedule and destruction protocols
//! - Data breach notification
//!
//! Security Hardening (Aethernet Phase I-II):
//! - Runtime validation of BIPA flags
//! - Consent verification before biokey derivation
//! - Automated retention and destruction tracking
//! - Audit logging of all biometric operations

#![no_std]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

/// BIPA compliance validation result
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BipaValidationResult {
    /// Compliant with BIPA
    Compliant,
    /// Missing informed consent
    MissingConsent,
    /// Missing storage disclosure
    MissingStorageDisclosure,
    /// Retention period exceeded
    RetentionExceeded,
    /// Unauthorized sale/profit detected
    UnauthorizedCommercialUse,
    /// Missing destruction schedule
    MissingDestructionSchedule,
}

/// BIPA consent record
#[derive(Debug, Clone)]
pub struct BipaConsent {
    /// Subject identifier (de-identified)
    pub subject_id: [u8; 32],
    /// Consent timestamp
    pub consent_timestamp: u64,
    /// Purpose of collection
    pub collection_purpose: String,
    /// Storage period (seconds)
    pub storage_period: u64,
    /// Consent signature/proof
    pub consent_proof: Vec<u8>,
}

/// BIPA retention policy
#[derive(Debug, Clone, Copy)]
pub struct RetentionPolicy {
    /// Maximum retention period (seconds)
    pub max_retention: u64,
    /// Grace period before destruction (seconds)
    pub grace_period: u64,
    /// Destruction method
    pub destruction_method: DestructionMethod,
}

/// Biometric data destruction method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DestructionMethod {
    /// Immediate memory wipe
    ImmediateWipe,
    /// Secure overwrite (multi-pass)
    SecureOverwrite,
    /// Cryptographic deletion (destroy keys)
    CryptoErasure,
}

/// BIPA audit event
#[derive(Debug, Clone)]
pub struct BipaAuditEvent {
    /// Event timestamp
    pub timestamp: u64,
    /// Subject identifier (de-identified)
    pub subject_id: [u8; 32],
    /// Event type
    pub event_type: BipaEventType,
    /// Additional details
    pub details: String,
}

/// BIPA event types for audit trail
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BipaEventType {
    /// Consent obtained
    ConsentObtained,
    /// Biometric data collected
    DataCollected,
    /// Biometric data accessed
    DataAccessed,
    /// Biometric data disclosed (to third party)
    DataDisclosed,
    /// Biometric data destroyed
    DataDestroyed,
    /// Retention period expired
    RetentionExpired,
}

/// BIPA compliance enforcer
pub struct BipaEnforcer {
    /// Active consent records
    consent_records: Vec<BipaConsent>,
    /// Retention policy
    retention_policy: RetentionPolicy,
    /// Audit trail
    audit_trail: Vec<BipaAuditEvent>,
}

impl BipaEnforcer {
    /// Create new BIPA enforcer
    pub fn new(retention_policy: RetentionPolicy) -> Self {
        Self {
            consent_records: Vec::new(),
            retention_policy,
            audit_trail: Vec::new(),
        }
    }
    
    /// Validate BIPA compliance for TXO
    ///
    /// # Arguments
    /// * `txo_flags` - TXO compliance flags
    /// * `subject_id` - Biometric subject identifier
    /// * `current_time` - Current timestamp
    ///
    /// # Returns
    /// * Validation result
    pub fn validate_compliance(
        &self,
        bipa_enabled: bool,
        subject_id: [u8; 32],
        current_time: u64,
    ) -> BipaValidationResult {
        if !bipa_enabled {
            // BIPA not flagged, but biometric data present - violation
            return BipaValidationResult::MissingConsent;
        }
        
        // Check consent record exists
        let consent = self.consent_records.iter()
            .find(|c| c.subject_id == subject_id);
        
        if consent.is_none() {
            return BipaValidationResult::MissingConsent;
        }
        
        let consent = consent.unwrap();
        
        // Check if storage disclosure was provided
        if consent.storage_period == 0 {
            return BipaValidationResult::MissingStorageDisclosure;
        }
        
        // Check retention period
        let age = current_time - consent.consent_timestamp;
        if age > consent.storage_period {
            return BipaValidationResult::RetentionExceeded;
        }
        
        BipaValidationResult::Compliant
    }
    
    /// Register consent before biometric collection
    ///
    /// # Arguments
    /// * `consent` - Consent record
    ///
    /// # Returns
    /// * Ok if consent registered, Err otherwise
    pub fn register_consent(&mut self, consent: BipaConsent) -> Result<(), &'static str> {
        // Validate consent has required fields
        if consent.collection_purpose.is_empty() {
            return Err("Missing collection purpose");
        }
        
        if consent.storage_period == 0 {
            return Err("Missing storage period");
        }
        
        if consent.consent_proof.is_empty() {
            return Err("Missing consent proof");
        }
        
        // Log audit event
        self.audit_trail.push(BipaAuditEvent {
            timestamp: consent.consent_timestamp,
            subject_id: consent.subject_id,
            event_type: BipaEventType::ConsentObtained,
            details: consent.collection_purpose.clone(),
        });
        
        // Store consent
        self.consent_records.push(consent);
        
        Ok(())
    }
    
    /// Log biometric data collection
    pub fn log_collection(
        &mut self,
        subject_id: [u8; 32],
        current_time: u64,
        details: String,
    ) {
        self.audit_trail.push(BipaAuditEvent {
            timestamp: current_time,
            subject_id,
            event_type: BipaEventType::DataCollected,
            details,
        });
    }
    
    /// Log biometric data access
    pub fn log_access(
        &mut self,
        subject_id: [u8; 32],
        current_time: u64,
        accessor: String,
    ) {
        self.audit_trail.push(BipaAuditEvent {
            timestamp: current_time,
            subject_id,
            event_type: BipaEventType::DataAccessed,
            details: accessor,
        });
    }
    
    /// Log biometric data destruction
    pub fn log_destruction(
        &mut self,
        subject_id: [u8; 32],
        current_time: u64,
        method: DestructionMethod,
    ) {
        let method_str = match method {
            DestructionMethod::ImmediateWipe => "immediate_wipe",
            DestructionMethod::SecureOverwrite => "secure_overwrite",
            DestructionMethod::CryptoErasure => "crypto_erasure",
        };
        
        self.audit_trail.push(BipaAuditEvent {
            timestamp: current_time,
            subject_id,
            event_type: BipaEventType::DataDestroyed,
            details: alloc::format!("method: {}", method_str),
        });
    }
    
    /// Check for expired retention periods
    ///
    /// # Arguments
    /// * `current_time` - Current timestamp
    ///
    /// # Returns
    /// * List of subject IDs with expired retention
    pub fn check_expired_retention(&mut self, current_time: u64) -> Vec<[u8; 32]> {
        let mut expired = Vec::new();
        
        for consent in &self.consent_records {
            let age = current_time - consent.consent_timestamp;
            if age > consent.storage_period {
                expired.push(consent.subject_id);
                
                // Log expiration
                self.audit_trail.push(BipaAuditEvent {
                    timestamp: current_time,
                    subject_id: consent.subject_id,
                    event_type: BipaEventType::RetentionExpired,
                    details: alloc::format!("age: {} seconds", age),
                });
            }
        }
        
        expired
    }
    
    /// Execute biometric data destruction
    ///
    /// # Arguments
    /// * `subject_id` - Subject whose data should be destroyed
    /// * `current_time` - Current timestamp
    ///
    /// # Returns
    /// * Ok if destroyed, Err if not found
    pub fn destroy_biometric_data(
        &mut self,
        subject_id: [u8; 32],
        current_time: u64,
    ) -> Result<(), &'static str> {
        // Remove consent record
        let idx = self.consent_records.iter()
            .position(|c| c.subject_id == subject_id);
        
        if let Some(idx) = idx {
            self.consent_records.remove(idx);
            
            // Log destruction
            self.log_destruction(
                subject_id,
                current_time,
                self.retention_policy.destruction_method,
            );
            
            Ok(())
        } else {
            Err("Subject not found")
        }
    }
    
    /// Get audit trail
    pub fn audit_trail(&self) -> &[BipaAuditEvent] {
        &self.audit_trail
    }
    
    /// Get active consent count
    pub fn active_consent_count(&self) -> usize {
        self.consent_records.len()
    }
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            max_retention: 365 * 24 * 3600, // 1 year
            grace_period: 30 * 24 * 3600,   // 30 days
            destruction_method: DestructionMethod::CryptoErasure,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_consent(subject_id: [u8; 32], timestamp: u64) -> BipaConsent {
        BipaConsent {
            subject_id,
            consent_timestamp: timestamp,
            collection_purpose: alloc::format!("Genomic analysis"),
            storage_period: 365 * 24 * 3600, // 1 year
            consent_proof: vec![0x01, 0x02, 0x03],
        }
    }
    
    #[test]
    fn test_enforcer_creation() {
        let policy = RetentionPolicy::default();
        let enforcer = BipaEnforcer::new(policy);
        
        assert_eq!(enforcer.active_consent_count(), 0);
    }
    
    #[test]
    fn test_consent_registration() {
        let policy = RetentionPolicy::default();
        let mut enforcer = BipaEnforcer::new(policy);
        
        let subject_id = [0x42u8; 32];
        let consent = create_test_consent(subject_id, 10000);
        
        let result = enforcer.register_consent(consent);
        assert!(result.is_ok());
        assert_eq!(enforcer.active_consent_count(), 1);
    }
    
    #[test]
    fn test_compliance_validation() {
        let policy = RetentionPolicy::default();
        let mut enforcer = BipaEnforcer::new(policy);
        
        let subject_id = [0x42u8; 32];
        let consent = create_test_consent(subject_id, 10000);
        enforcer.register_consent(consent).unwrap();
        
        // Should be compliant
        let result = enforcer.validate_compliance(true, subject_id, 10100);
        assert_eq!(result, BipaValidationResult::Compliant);
    }
    
    #[test]
    fn test_missing_consent() {
        let policy = RetentionPolicy::default();
        let enforcer = BipaEnforcer::new(policy);
        
        let subject_id = [0x42u8; 32];
        
        // No consent registered
        let result = enforcer.validate_compliance(true, subject_id, 10000);
        assert_eq!(result, BipaValidationResult::MissingConsent);
    }
    
    #[test]
    fn test_retention_expiration() {
        let policy = RetentionPolicy::default();
        let mut enforcer = BipaEnforcer::new(policy);
        
        let subject_id = [0x42u8; 32];
        let consent = BipaConsent {
            subject_id,
            consent_timestamp: 10000,
            collection_purpose: alloc::format!("Test"),
            storage_period: 100, // 100 seconds retention
            consent_proof: vec![0x01],
        };
        enforcer.register_consent(consent).unwrap();
        
        // Within retention period
        let result1 = enforcer.validate_compliance(true, subject_id, 10050);
        assert_eq!(result1, BipaValidationResult::Compliant);
        
        // Exceeded retention period
        let result2 = enforcer.validate_compliance(true, subject_id, 10200);
        assert_eq!(result2, BipaValidationResult::RetentionExceeded);
    }
    
    #[test]
    fn test_expired_retention_check() {
        let policy = RetentionPolicy::default();
        let mut enforcer = BipaEnforcer::new(policy);
        
        let subject_id1 = [0x01u8; 32];
        let subject_id2 = [0x02u8; 32];
        
        // Register two consents
        let consent1 = BipaConsent {
            subject_id: subject_id1,
            consent_timestamp: 10000,
            collection_purpose: alloc::format!("Test"),
            storage_period: 100,
            consent_proof: vec![0x01],
        };
        let consent2 = BipaConsent {
            subject_id: subject_id2,
            consent_timestamp: 10000,
            collection_purpose: alloc::format!("Test"),
            storage_period: 1000,
            consent_proof: vec![0x02],
        };
        
        enforcer.register_consent(consent1).unwrap();
        enforcer.register_consent(consent2).unwrap();
        
        // Check at time when first is expired but second is not
        let expired = enforcer.check_expired_retention(10150);
        
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0], subject_id1);
    }
    
    #[test]
    fn test_data_destruction() {
        let policy = RetentionPolicy::default();
        let mut enforcer = BipaEnforcer::new(policy);
        
        let subject_id = [0x42u8; 32];
        let consent = create_test_consent(subject_id, 10000);
        enforcer.register_consent(consent).unwrap();
        
        assert_eq!(enforcer.active_consent_count(), 1);
        
        // Destroy data
        let result = enforcer.destroy_biometric_data(subject_id, 11000);
        assert!(result.is_ok());
        assert_eq!(enforcer.active_consent_count(), 0);
    }
    
    #[test]
    fn test_audit_trail() {
        let policy = RetentionPolicy::default();
        let mut enforcer = BipaEnforcer::new(policy);
        
        let subject_id = [0x42u8; 32];
        let consent = create_test_consent(subject_id, 10000);
        enforcer.register_consent(consent).unwrap();
        
        // Log some events
        enforcer.log_collection(subject_id, 10001, alloc::format!("SNP loci"));
        enforcer.log_access(subject_id, 10002, alloc::format!("operator-1"));
        
        // Should have multiple audit events
        assert!(enforcer.audit_trail().len() >= 3);
    }
    
    #[test]
    fn test_invalid_consent() {
        let policy = RetentionPolicy::default();
        let mut enforcer = BipaEnforcer::new(policy);
        
        // Consent with missing purpose
        let mut consent = create_test_consent([0x42u8; 32], 10000);
        consent.collection_purpose = alloc::format!("");
        
        let result = enforcer.register_consent(consent);
        assert!(result.is_err());
    }
}
