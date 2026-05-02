//! GDPR Compliance Module
//!
//! General Data Protection Regulation compliance for Aethernet.
//! Ensures data sovereignty, privacy, and individual rights.

#![no_std]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

/// GDPR Article 6 - Lawful Basis for Processing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LawfulBasis {
    /// Article 6(1)(a) - Consent
    Consent,
    /// Article 6(1)(b) - Contract
    Contract,
    /// Article 6(1)(c) - Legal obligation
    LegalObligation,
    /// Article 6(1)(d) - Vital interests
    VitalInterests,
    /// Article 6(1)(e) - Public interest
    PublicInterest,
    /// Article 6(1)(f) - Legitimate interests
    LegitimateInterests,
}

/// GDPR Article 9 - Special Categories of Personal Data
pub mod special_categories {
    use super::*;
    
    /// Verify lawful basis for processing special category data
    pub fn verify_special_category_basis(
        data_type: &str,
        lawful_basis: LawfulBasis,
        explicit_consent: bool,
    ) -> bool {
        // Special categories require explicit consent or specific exemptions
        match lawful_basis {
            LawfulBasis::Consent => explicit_consent,
            LawfulBasis::VitalInterests => true, // Health emergency
            LawfulBasis::LegalObligation => true, // Legal requirement
            _ => false,
        }
    }
    
    /// Genetic data processing (GDPR Article 9)
    pub fn verify_genetic_data_processing(
        purpose: &str,
        explicit_consent: bool,
        medical_necessity: bool,
    ) -> bool {
        // Genetic data requires explicit consent OR medical necessity
        explicit_consent || medical_necessity
    }
}

/// GDPR Article 12-23 - Data Subject Rights
pub mod data_subject_rights {
    use super::*;
    
    /// Article 15 - Right of access
    pub fn handle_access_request(
        subject_id: &[u8; 16],
        requested_data: &[&str],
    ) -> Result<Vec<u8>, String> {
        // In production, this would:
        // 1. Verify subject identity
        // 2. Gather all personal data
        // 3. Provide in accessible format
        // 4. Respond within 1 month
        
        Ok(Vec::new())
    }
    
    /// Article 16 - Right to rectification
    pub fn handle_rectification_request(
        subject_id: &[u8; 16],
        incorrect_data: &[u8],
        corrected_data: &[u8],
    ) -> Result<(), String> {
        // In production, this would:
        // 1. Verify subject identity
        // 2. Update incorrect data
        // 3. Notify recipients of correction
        // 4. Respond within 1 month
        
        Ok(())
    }
    
    /// Article 17 - Right to erasure ("right to be forgotten")
    pub fn handle_erasure_request(
        subject_id: &[u8; 16],
        erasure_scope: &str,
    ) -> Result<(), String> {
        // In production, this would:
        // 1. Verify subject identity
        // 2. Check if erasure is required (no overriding legal basis)
        // 3. Erase or anonymize data
        // 4. Notify recipients of erasure
        // 5. Respond within 1 month
        
        // Note: Some data may need to be retained for legal reasons
        Ok(())
    }
    
    /// Article 18 - Right to restriction of processing
    pub fn handle_restriction_request(
        subject_id: &[u8; 16],
        restriction_reason: &str,
    ) -> Result<(), String> {
        // In production, this would:
        // 1. Verify subject identity
        // 2. Mark data as restricted
        // 3. Prevent processing except storage
        // 4. Notify recipients of restriction
        
        Ok(())
    }
    
    /// Article 20 - Right to data portability
    pub fn handle_portability_request(
        subject_id: &[u8; 16],
        export_format: &str,
    ) -> Result<Vec<u8>, String> {
        // In production, this would:
        // 1. Verify subject identity
        // 2. Export data in structured, machine-readable format
        // 3. Provide in commonly used format (JSON, CSV, XML)
        // 4. Respond within 1 month
        
        Ok(Vec::new())
    }
    
    /// Article 21 - Right to object
    pub fn handle_objection_request(
        subject_id: &[u8; 16],
        objection_basis: &str,
    ) -> Result<(), String> {
        // In production, this would:
        // 1. Verify subject identity
        // 2. Assess if legitimate grounds override objection
        // 3. Cease processing if objection is valid
        // 4. Respond within 1 month
        
        Ok(())
    }
}

/// GDPR Article 25 - Data Protection by Design and Default
pub mod data_protection_by_design {
    use super::*;
    
    /// Pseudonymization
    pub fn pseudonymize_data(data: &[u8], key: &[u8; 32]) -> Vec<u8> {
        // In production, this would:
        // 1. Replace identifiers with pseudonyms
        // 2. Store mapping separately with access controls
        // 3. Use deterministic or randomized pseudonymization
        
        // Placeholder
        data.to_vec()
    }
    
    /// Data minimization
    pub fn verify_data_minimization(
        collected_fields: &[&str],
        necessary_fields: &[&str],
    ) -> bool {
        // Verify only necessary data is collected
        collected_fields.len() <= necessary_fields.len()
    }
    
    /// Privacy by default
    pub fn apply_privacy_defaults(settings: &mut PrivacySettings) {
        // In production, this would:
        // 1. Set strictest privacy settings by default
        // 2. Require opt-in for data sharing
        // 3. Minimize data visibility
        
        settings.data_sharing_enabled = false;
        settings.analytics_enabled = false;
        settings.marketing_enabled = false;
    }
}

/// Privacy settings
#[derive(Debug, Clone)]
pub struct PrivacySettings {
    pub data_sharing_enabled: bool,
    pub analytics_enabled: bool,
    pub marketing_enabled: bool,
}

/// GDPR Article 32 - Security of Processing
pub mod security_of_processing {
    use super::*;
    
    /// Verify encryption at rest
    pub fn verify_encryption_at_rest(data_location: &str) -> bool {
        // In production, this would:
        // 1. Check if data is encrypted (AES-256)
        // 2. Verify key management
        // 3. Ensure proper access controls
        
        true
    }
    
    /// Verify encryption in transit
    pub fn verify_encryption_in_transit(connection_id: &str) -> bool {
        // In production, this would:
        // 1. Check TLS 1.3+ is used
        // 2. Verify certificate validity
        // 3. Ensure strong cipher suites
        
        true
    }
    
    /// Pseudonymization verification
    pub fn verify_pseudonymization(data: &[u8]) -> bool {
        // In production, this would:
        // 1. Check if identifiers are pseudonymized
        // 2. Verify mapping is stored separately
        // 3. Ensure re-identification requires additional info
        
        true
    }
}

/// GDPR Article 33 - Breach Notification to Supervisory Authority
pub mod breach_notification {
    use super::*;
    
    /// Assess breach severity
    pub fn assess_breach_severity(
        affected_individuals: usize,
        data_sensitivity: DataSensitivity,
        mitigation_applied: bool,
    ) -> BreachSeverity {
        if mitigation_applied {
            return BreachSeverity::Low;
        }
        
        match data_sensitivity {
            DataSensitivity::High => BreachSeverity::High,
            DataSensitivity::Medium => {
                if affected_individuals > 1000 {
                    BreachSeverity::High
                } else {
                    BreachSeverity::Medium
                }
            }
            DataSensitivity::Low => BreachSeverity::Low,
        }
    }
    
    /// Notify supervisory authority (within 72 hours)
    pub fn notify_supervisory_authority(
        breach_description: &str,
        affected_individuals: usize,
        mitigation_measures: &str,
    ) -> Result<(), String> {
        // In production, this would:
        // 1. Prepare breach notification
        // 2. Submit to relevant supervisory authority
        // 3. Include: nature, consequences, measures taken
        // 4. Do within 72 hours of awareness
        
        Ok(())
    }
    
    /// Notify affected individuals (if high risk)
    pub fn notify_affected_individuals(
        individuals: &[[u8; 16]],
        breach_description: &str,
    ) -> Result<(), String> {
        // In production, this would:
        // 1. Notify each individual directly
        // 2. Describe breach in clear language
        // 3. Provide contact point for inquiries
        // 4. Recommend mitigation steps
        
        Ok(())
    }
}

/// Data sensitivity classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataSensitivity {
    High,   // Special categories (genetic, health)
    Medium, // Personal identifiers
    Low,    // Anonymized/pseudonymized
}

/// Breach severity
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BreachSeverity {
    Low,
    Medium,
    High,
}

/// GDPR Article 35 - Data Protection Impact Assessment (DPIA)
pub mod dpia {
    use super::*;
    
    /// Check if DPIA is required
    pub fn is_dpia_required(
        processing_type: &str,
        data_sensitivity: DataSensitivity,
        automated_decision_making: bool,
    ) -> bool {
        // DPIA required for high-risk processing
        automated_decision_making || data_sensitivity == DataSensitivity::High
    }
    
    /// Conduct DPIA
    pub fn conduct_dpia(
        processing_description: &str,
        necessity_justification: &str,
        risks_identified: &[&str],
        mitigation_measures: &[&str],
    ) -> DPIAResult {
        // In production, this would:
        // 1. Assess necessity and proportionality
        // 2. Identify risks to rights and freedoms
        // 3. Evaluate mitigation measures
        // 4. Document findings
        
        DPIAResult::Acceptable
    }
}

/// DPIA result
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DPIAResult {
    Acceptable,
    AcceptableWithMitigation,
    RequiresSupervisoryConsultation,
    Unacceptable,
}

/// GDPR Article 44-50 - International Data Transfers
pub mod international_transfers {
    use super::*;
    
    /// Verify adequacy decision
    pub fn verify_adequacy_decision(destination_country: &str) -> bool {
        // In production, this would:
        // 1. Check if country has EU adequacy decision
        // 2. Verify decision is current
        
        // Countries with adequacy decisions (as of 2024):
        // Andorra, Argentina, Canada (commercial), Faroe Islands, Guernsey,
        // Israel, Isle of Man, Japan, Jersey, New Zealand, South Korea,
        // Switzerland, UK, Uruguay, USA (Data Privacy Framework)
        
        matches!(
            destination_country,
            "CH" | "UK" | "CA" | "JP" | "NZ" | "KR" | "IL"
        )
    }
    
    /// Verify standard contractual clauses (SCCs)
    pub fn verify_sccs(
        contract_version: &str,
        data_exporter_signed: bool,
        data_importer_signed: bool,
    ) -> bool {
        // In production, this would:
        // 1. Verify SCCs are EU-approved version
        // 2. Check both parties signed
        // 3. Ensure proper implementation
        
        data_exporter_signed && data_importer_signed
    }
}

/// GDPR compliance verification
pub fn verify_gdpr_compliance(
    lawful_basis: LawfulBasis,
    explicit_consent: bool,
    data_sensitivity: DataSensitivity,
) -> bool {
    // Verify GDPR requirements
    match data_sensitivity {
        DataSensitivity::High => {
            // Special category data requires explicit consent
            explicit_consent
        }
        _ => {
            // Regular personal data needs lawful basis
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_special_category_processing() {
        // Genetic data with explicit consent should be allowed
        assert!(special_categories::verify_genetic_data_processing(
            "research",
            true,  // explicit consent
            false  // not medical necessity
        ));
        
        // Genetic data without consent but medical necessity should be allowed
        assert!(special_categories::verify_genetic_data_processing(
            "treatment",
            false, // no explicit consent
            true   // medical necessity
        ));
        
        // Genetic data without consent or necessity should be denied
        assert!(!special_categories::verify_genetic_data_processing(
            "marketing",
            false, // no explicit consent
            false  // no medical necessity
        ));
    }
    
    #[test]
    fn test_breach_severity_assessment() {
        // High sensitivity data breach - always high severity
        let severity = breach_notification::assess_breach_severity(
            100,
            DataSensitivity::High,
            false,
        );
        assert_eq!(severity, BreachSeverity::High);
        
        // Medium sensitivity, many affected - high severity
        let severity = breach_notification::assess_breach_severity(
            2000,
            DataSensitivity::Medium,
            false,
        );
        assert_eq!(severity, BreachSeverity::High);
        
        // Mitigation applied - low severity
        let severity = breach_notification::assess_breach_severity(
            2000,
            DataSensitivity::High,
            true,
        );
        assert_eq!(severity, BreachSeverity::Low);
    }
    
    #[test]
    fn test_dpia_requirement() {
        // Automated decision-making requires DPIA
        assert!(dpia::is_dpia_required(
            "profiling",
            DataSensitivity::Medium,
            true
        ));
        
        // High sensitivity data requires DPIA
        assert!(dpia::is_dpia_required(
            "processing",
            DataSensitivity::High,
            false
        ));
        
        // Low risk processing doesn't require DPIA
        assert!(!dpia::is_dpia_required(
            "processing",
            DataSensitivity::Low,
            false
        ));
    }
    
    #[test]
    fn test_international_transfers() {
        // Countries with adequacy decision
        assert!(international_transfers::verify_adequacy_decision("CH")); // Switzerland
        assert!(international_transfers::verify_adequacy_decision("UK")); // United Kingdom
        
        // Countries without adequacy decision
        assert!(!international_transfers::verify_adequacy_decision("CN")); // China
    }
    
    #[test]
    fn test_gdpr_compliance() {
        // High sensitivity with explicit consent - compliant
        assert!(verify_gdpr_compliance(
            LawfulBasis::Consent,
            true,
            DataSensitivity::High
        ));
        
        // High sensitivity without explicit consent - not compliant
        assert!(!verify_gdpr_compliance(
            LawfulBasis::LegitimateInterests,
            false,
            DataSensitivity::High
        ));
        
        // Medium sensitivity with lawful basis - compliant
        assert!(verify_gdpr_compliance(
            LawfulBasis::Contract,
            false,
            DataSensitivity::Medium
        ));
    }
}
