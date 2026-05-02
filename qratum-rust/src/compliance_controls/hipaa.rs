//! HIPAA Compliance Engine
//!
//! Executable controls for Health Insurance Portability and Accountability Act (HIPAA)
//! compliance including:
//! - Protected Health Information (PHI) tagging
//! - Access audit trail with immutable logging
//! - Minimum necessary rule enforcement
//! - Breach notification triggers
//!
//! ## Regulatory Reference
//! - 45 CFR 164.308: Administrative Safeguards
//! - 45 CFR 164.312: Technical Safeguards
//! - 45 CFR 164.530: Privacy Rule Safeguards

extern crate alloc;
use alloc::vec::Vec;
use alloc::string::String;
use alloc::collections::BTreeMap;

use sha3::{Sha3_256, Digest};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// PHI Data Categories per HIPAA 164.501
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhiCategory {
    /// Names
    Names,
    /// Geographic data smaller than state
    GeographicData,
    /// Dates related to individual (birth, death, admission, discharge)
    Dates,
    /// Phone numbers
    PhoneNumbers,
    /// Fax numbers
    FaxNumbers,
    /// Email addresses
    EmailAddresses,
    /// Social Security Numbers
    SocialSecurityNumbers,
    /// Medical Record Numbers
    MedicalRecordNumbers,
    /// Health Plan Beneficiary Numbers
    HealthPlanNumbers,
    /// Account Numbers
    AccountNumbers,
    /// Certificate/License Numbers
    CertificateNumbers,
    /// Vehicle Identifiers
    VehicleIdentifiers,
    /// Device Identifiers
    DeviceIdentifiers,
    /// Web URLs
    WebUrls,
    /// IP Addresses
    IpAddresses,
    /// Biometric Identifiers
    BiometricIdentifiers,
    /// Full Face Photographs
    Photographs,
    /// Any other unique identifier
    OtherUniqueIdentifier,
}

/// PHI Sensitivity Level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PhiSensitivity {
    /// Low sensitivity (general health info)
    Low,
    /// Medium sensitivity (diagnoses, treatments)
    Medium,
    /// High sensitivity (mental health, HIV/AIDS, substance abuse)
    High,
    /// Restricted (psychotherapy notes, genetic data)
    Restricted,
}

/// PHI Tag attached to data elements
#[derive(Debug, Clone)]
pub struct PhiTag {
    /// Data element identifier
    pub element_id: [u8; 32],
    
    /// PHI categories present
    pub categories: Vec<PhiCategory>,
    
    /// Sensitivity level
    pub sensitivity: PhiSensitivity,
    
    /// Originating entity (covered entity or business associate)
    pub originating_entity: String,
    
    /// Timestamp when tagged
    pub tagged_at: u64,
    
    /// Minimum necessary designation
    pub minimum_necessary: bool,
    
    /// Retention period in seconds (0 = indefinite with legal hold)
    pub retention_period: u64,
}

impl PhiTag {
    /// Create new PHI tag for data element
    pub fn new(
        element_id: [u8; 32],
        categories: Vec<PhiCategory>,
        sensitivity: PhiSensitivity,
        originating_entity: String,
    ) -> Self {
        Self {
            element_id,
            categories,
            sensitivity,
            originating_entity,
            tagged_at: current_timestamp(),
            minimum_necessary: false,
            retention_period: 0,
        }
    }
    
    /// Set minimum necessary designation
    pub fn with_minimum_necessary(mut self, min_necessary: bool) -> Self {
        self.minimum_necessary = min_necessary;
        self
    }
    
    /// Set retention period
    pub fn with_retention(mut self, period_seconds: u64) -> Self {
        self.retention_period = period_seconds;
        self
    }
    
    /// Check if PHI contains any high-sensitivity categories
    pub fn is_high_sensitivity(&self) -> bool {
        self.sensitivity >= PhiSensitivity::High
    }
    
    /// Generate cryptographic hash for audit reference
    pub fn audit_hash(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(&self.element_id);
        hasher.update(&self.tagged_at.to_le_bytes());
        hasher.update(self.originating_entity.as_bytes());
        hasher.finalize().into()
    }
}

/// Access Purpose per HIPAA permitted uses
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPurpose {
    /// Treatment - healthcare provider for treatment
    Treatment,
    /// Payment - processing claims
    Payment,
    /// Healthcare Operations - quality improvement, training
    HealthcareOperations,
    /// Required by Law
    RequiredByLaw,
    /// Public Health Activities
    PublicHealth,
    /// Abuse/Neglect Reporting
    AbuseReporting,
    /// Health Oversight
    HealthOversight,
    /// Judicial/Administrative Proceedings
    JudicialProceedings,
    /// Law Enforcement
    LawEnforcement,
    /// Research (IRB approved)
    Research,
    /// Averting Serious Threat
    SeriousThreat,
    /// Specialized Government Functions
    GovernmentFunctions,
    /// Workers Compensation
    WorkersCompensation,
    /// Individual Authorization
    IndividualAuthorization,
}

/// Access Audit Record for 164.312(b) audit controls
#[derive(Debug, Clone)]
pub struct AccessAuditRecord {
    /// Unique audit record ID
    pub audit_id: [u8; 32],
    
    /// Timestamp of access
    pub timestamp: u64,
    
    /// Accessor identity (user/system)
    pub accessor_id: String,
    
    /// Accessor role
    pub accessor_role: String,
    
    /// PHI element(s) accessed
    pub phi_elements: Vec<[u8; 32]>,
    
    /// Access purpose
    pub purpose: AccessPurpose,
    
    /// Action performed
    pub action: AccessAction,
    
    /// Access granted or denied
    pub granted: bool,
    
    /// Denial reason (if denied)
    pub denial_reason: Option<String>,
    
    /// Patient authorization reference (if applicable)
    pub authorization_ref: Option<[u8; 32]>,
    
    /// Minimum necessary verification performed
    pub min_necessary_verified: bool,
}

/// Access Action Types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessAction {
    /// Read/View PHI
    Read,
    /// Create new PHI
    Create,
    /// Update existing PHI
    Update,
    /// Delete PHI
    Delete,
    /// Disclose to external party
    Disclose,
    /// Export/Download PHI
    Export,
    /// Print PHI
    Print,
    /// Copy PHI
    Copy,
}

impl AccessAuditRecord {
    /// Create new audit record
    pub fn new(
        accessor_id: String,
        accessor_role: String,
        phi_elements: Vec<[u8; 32]>,
        purpose: AccessPurpose,
        action: AccessAction,
    ) -> Self {
        let timestamp = current_timestamp();
        
        // Generate audit ID
        let mut hasher = Sha3_256::new();
        hasher.update(&timestamp.to_le_bytes());
        hasher.update(accessor_id.as_bytes());
        hasher.update(&(action as u8).to_le_bytes());
        let audit_id: [u8; 32] = hasher.finalize().into();
        
        Self {
            audit_id,
            timestamp,
            accessor_id,
            accessor_role,
            phi_elements,
            purpose,
            action,
            granted: false,
            denial_reason: None,
            authorization_ref: None,
            min_necessary_verified: false,
        }
    }
    
    /// Mark access as granted
    pub fn grant(&mut self) {
        self.granted = true;
    }
    
    /// Mark access as denied with reason
    pub fn deny(&mut self, reason: String) {
        self.granted = false;
        self.denial_reason = Some(reason);
    }
    
    /// Attach authorization reference
    pub fn with_authorization(mut self, auth_ref: [u8; 32]) -> Self {
        self.authorization_ref = Some(auth_ref);
        self
    }
    
    /// Mark minimum necessary as verified
    pub fn with_min_necessary_verification(mut self) -> Self {
        self.min_necessary_verified = true;
        self
    }
}

/// Breach Assessment per 164.402
#[derive(Debug, Clone)]
pub struct BreachAssessment {
    /// Assessment ID
    pub assessment_id: [u8; 32],
    
    /// Timestamp of assessment
    pub timestamp: u64,
    
    /// Incident description
    pub incident_description: String,
    
    /// PHI involved
    pub phi_involved: Vec<[u8; 32]>,
    
    /// Number of individuals affected
    pub individuals_affected: u32,
    
    /// Nature and extent of PHI involved
    pub phi_extent: PhiExtent,
    
    /// Unauthorized person assessment
    pub unauthorized_person: Option<String>,
    
    /// PHI actually acquired or viewed
    pub phi_acquired: bool,
    
    /// Risk mitigation measures
    pub mitigation_measures: Vec<String>,
    
    /// Reportable breach determination
    pub is_reportable: bool,
    
    /// Notification deadline (if reportable)
    pub notification_deadline: Option<u64>,
}

/// Extent of PHI involved in breach
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhiExtent {
    /// Names only
    NamesOnly,
    /// Limited identifiers
    LimitedIdentifiers,
    /// Clinical information included
    ClinicalInformation,
    /// Financial information included
    FinancialInformation,
    /// Full medical records
    FullMedicalRecords,
}

/// HIPAA Compliance Engine
///
/// Provides executable controls for HIPAA compliance including:
/// - PHI tagging and tracking
/// - Access audit logging
/// - Minimum necessary enforcement
/// - Breach notification triggers
pub struct HipaaComplianceEngine {
    /// PHI tags registry
    phi_tags: BTreeMap<[u8; 32], PhiTag>,
    
    /// Access audit log (immutable)
    audit_log: Vec<AccessAuditRecord>,
    
    /// Breach assessments
    breach_assessments: Vec<BreachAssessment>,
    
    /// Role-based access matrix
    role_access_matrix: BTreeMap<String, Vec<AccessPurpose>>,
    
    /// Audit retention period (6 years per HIPAA)
    audit_retention_seconds: u64,
}

impl HipaaComplianceEngine {
    /// Create new HIPAA compliance engine
    pub fn new() -> Self {
        Self {
            phi_tags: BTreeMap::new(),
            audit_log: Vec::new(),
            breach_assessments: Vec::new(),
            role_access_matrix: BTreeMap::new(),
            audit_retention_seconds: 6 * 365 * 24 * 60 * 60, // 6 years
        }
    }
    
    /// Register PHI data element with tag
    pub fn register_phi(&mut self, tag: PhiTag) {
        self.phi_tags.insert(tag.element_id, tag);
    }
    
    /// Check and log access request
    ///
    /// Returns true if access is granted, false if denied.
    /// All access attempts are logged regardless of outcome.
    pub fn check_access(&mut self, mut record: AccessAuditRecord) -> bool {
        // Verify minimum necessary
        let mut min_necessary_ok = true;
        for phi_id in &record.phi_elements {
            if let Some(tag) = self.phi_tags.get(phi_id) {
                if tag.minimum_necessary && !record.min_necessary_verified {
                    min_necessary_ok = false;
                    break;
                }
            }
        }
        
        // Check role-based access
        let role_access_ok = self.role_access_matrix
            .get(&record.accessor_role)
            .map(|purposes| purposes.contains(&record.purpose))
            .unwrap_or(false);
        
        // Check sensitivity level access
        let sensitivity_ok = self.check_sensitivity_access(&record);
        
        // Determine access decision
        if min_necessary_ok && role_access_ok && sensitivity_ok {
            record.grant();
        } else {
            let mut reasons = Vec::new();
            if !min_necessary_ok {
                reasons.push("Minimum necessary not verified");
            }
            if !role_access_ok {
                reasons.push("Role not authorized for purpose");
            }
            if !sensitivity_ok {
                reasons.push("Sensitivity level access denied");
            }
            record.deny(reasons.join("; "));
        }
        
        let granted = record.granted;
        
        // Log audit record (immutable)
        self.audit_log.push(record);
        
        granted
    }
    
    /// Check sensitivity-based access
    fn check_sensitivity_access(&self, record: &AccessAuditRecord) -> bool {
        for phi_id in &record.phi_elements {
            if let Some(tag) = self.phi_tags.get(phi_id) {
                // Restricted PHI requires individual authorization
                if tag.sensitivity == PhiSensitivity::Restricted {
                    if record.authorization_ref.is_none() {
                        return false;
                    }
                }
                
                // High sensitivity PHI requires explicit purpose
                if tag.sensitivity == PhiSensitivity::High {
                    match record.purpose {
                        AccessPurpose::Treatment |
                        AccessPurpose::RequiredByLaw |
                        AccessPurpose::IndividualAuthorization => {}
                        _ => return false,
                    }
                }
            }
        }
        true
    }
    
    /// Configure role access permissions
    pub fn configure_role(&mut self, role: String, permitted_purposes: Vec<AccessPurpose>) {
        self.role_access_matrix.insert(role, permitted_purposes);
    }
    
    /// Perform breach assessment
    pub fn assess_breach(&mut self, assessment: BreachAssessment) -> bool {
        let is_reportable = self.determine_reportability(&assessment);
        
        let mut assessment = assessment;
        assessment.is_reportable = is_reportable;
        
        if is_reportable {
            // 60 days notification deadline
            assessment.notification_deadline = Some(
                assessment.timestamp + (60 * 24 * 60 * 60 * 1000)
            );
        }
        
        self.breach_assessments.push(assessment);
        is_reportable
    }
    
    /// Determine if breach is reportable per 164.402
    fn determine_reportability(&self, assessment: &BreachAssessment) -> bool {
        // Low probability of compromise exceptions
        if !assessment.phi_acquired {
            return false;
        }
        
        // 500+ individuals requires immediate reporting
        if assessment.individuals_affected >= 500 {
            return true;
        }
        
        // Clinical/financial information is high risk
        matches!(
            assessment.phi_extent,
            PhiExtent::ClinicalInformation |
            PhiExtent::FinancialInformation |
            PhiExtent::FullMedicalRecords
        )
    }
    
    /// Get audit log for specific PHI element
    pub fn get_phi_audit_trail(&self, phi_id: &[u8; 32]) -> Vec<&AccessAuditRecord> {
        self.audit_log
            .iter()
            .filter(|r| r.phi_elements.contains(phi_id))
            .collect()
    }
    
    /// Get all audit records within time range
    pub fn get_audit_records(&self, start: u64, end: u64) -> Vec<&AccessAuditRecord> {
        self.audit_log
            .iter()
            .filter(|r| r.timestamp >= start && r.timestamp <= end)
            .collect()
    }
    
    /// Generate HIPAA compliance report
    pub fn generate_compliance_report(&self) -> HipaaComplianceReport {
        let total_phi_elements = self.phi_tags.len();
        let total_access_events = self.audit_log.len();
        let denied_access_events = self.audit_log.iter().filter(|r| !r.granted).count();
        let reportable_breaches = self.breach_assessments.iter().filter(|b| b.is_reportable).count();
        
        // Calculate sensitivity distribution
        let high_sensitivity_phi = self.phi_tags
            .values()
            .filter(|t| t.is_high_sensitivity())
            .count();
        
        HipaaComplianceReport {
            report_timestamp: current_timestamp(),
            total_phi_elements,
            high_sensitivity_phi,
            total_access_events,
            denied_access_events,
            reportable_breaches,
            audit_retention_days: (self.audit_retention_seconds / 86400) as u32,
        }
    }
}

impl Default for HipaaComplianceEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// HIPAA Compliance Report
#[derive(Debug, Clone)]
pub struct HipaaComplianceReport {
    pub report_timestamp: u64,
    pub total_phi_elements: usize,
    pub high_sensitivity_phi: usize,
    pub total_access_events: usize,
    pub denied_access_events: usize,
    pub reportable_breaches: usize,
    pub audit_retention_days: u32,
}

/// Get current timestamp
fn current_timestamp() -> u64 {
    #[cfg(feature = "std")]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }
    #[cfg(not(feature = "std"))]
    {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_phi_tag_creation() {
        let tag = PhiTag::new(
            [1u8; 32],
            vec![PhiCategory::Names, PhiCategory::MedicalRecordNumbers],
            PhiSensitivity::Medium,
            "Hospital A".into(),
        );
        
        assert_eq!(tag.categories.len(), 2);
        assert!(!tag.is_high_sensitivity());
    }
    
    #[test]
    fn test_access_audit() {
        let mut engine = HipaaComplianceEngine::new();
        
        // Configure role
        engine.configure_role(
            "Physician".into(),
            vec![AccessPurpose::Treatment, AccessPurpose::HealthcareOperations],
        );
        
        // Register PHI
        let phi_id = [1u8; 32];
        let tag = PhiTag::new(
            phi_id,
            vec![PhiCategory::MedicalRecordNumbers],
            PhiSensitivity::Medium,
            "Hospital A".into(),
        );
        engine.register_phi(tag);
        
        // Access request
        let record = AccessAuditRecord::new(
            "Dr. Smith".into(),
            "Physician".into(),
            vec![phi_id],
            AccessPurpose::Treatment,
            AccessAction::Read,
        );
        
        let granted = engine.check_access(record);
        assert!(granted);
    }
    
    #[test]
    fn test_denied_access() {
        let mut engine = HipaaComplianceEngine::new();
        
        // No role configured - should deny
        let record = AccessAuditRecord::new(
            "Unknown User".into(),
            "Unknown".into(),
            vec![[1u8; 32]],
            AccessPurpose::Treatment,
            AccessAction::Read,
        );
        
        let granted = engine.check_access(record);
        assert!(!granted);
    }
    
    #[test]
    fn test_breach_assessment() {
        let mut engine = HipaaComplianceEngine::new();
        
        let assessment = BreachAssessment {
            assessment_id: [1u8; 32],
            timestamp: current_timestamp(),
            incident_description: "Laptop stolen".into(),
            phi_involved: vec![[1u8; 32]],
            individuals_affected: 1000,
            phi_extent: PhiExtent::FullMedicalRecords,
            unauthorized_person: Some("Unknown".into()),
            phi_acquired: true,
            mitigation_measures: vec!["Remote wipe initiated".into()],
            is_reportable: false,
            notification_deadline: None,
        };
        
        let reportable = engine.assess_breach(assessment);
        assert!(reportable); // 1000 individuals affected
    }
}
