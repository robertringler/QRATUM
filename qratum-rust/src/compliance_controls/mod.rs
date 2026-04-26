//! Compliance Controls Module
//!
//! Executable compliance controls for regulatory frameworks:
//! - HIPAA: Healthcare data protection
//! - GDPR: EU data protection with cryptographic tombstoning
//! - CMMC L2: Defense contractor cybersecurity
//!
//! ## Architecture
//!
//! Each compliance engine provides:
//! 1. Data tagging and classification
//! 2. Access control enforcement
//! 3. Audit trail generation
//! 4. Compliance reporting
//!
//! ## Usage
//!
//! ```rust,ignore
//! use qratum::compliance_controls::{HipaaComplianceEngine, GdprComplianceEngine, CmmcComplianceEngine};
//!
//! // HIPAA compliance
//! let hipaa = HipaaComplianceEngine::new();
//!
//! // GDPR compliance with cryptographic tombstoning
//! let gdpr = GdprComplianceEngine::new("Controller".into());
//!
//! // CMMC L2 with role-based enclaves
//! let cmmc = CmmcComplianceEngine::new();
//! ```

pub mod hipaa;
pub mod gdpr;
pub mod cmmc;

pub use hipaa::{
    HipaaComplianceEngine,
    PhiTag,
    PhiCategory,
    PhiSensitivity,
    AccessAuditRecord,
    AccessPurpose,
    AccessAction,
    BreachAssessment,
    PhiExtent,
    HipaaComplianceReport,
};

pub use gdpr::{
    GdprComplianceEngine,
    PersonalDataRecord,
    DataCategory as GdprDataCategory,
    LawfulBasis,
    DataSubjectRight,
    CryptographicTombstone,
    ErasureReason,
    ConsentRecord,
    DataSubjectAccessRequest,
    GdprComplianceReport,
};

pub use cmmc::{
    CmmcComplianceEngine,
    SecurityEnclave,
    ClassificationLevel,
    DataCategory as CmmcDataCategory,
    BoundaryControls,
    UserIdentity,
    AccountStatus,
    AccessControlEntry,
    Permission,
    AccessCondition,
    CmmcAuditEvent,
    AuditEventType,
    ConfigurationBaseline,
    ConfigurationItem,
    Criticality,
    CmmcComplianceReport,
};

/// Unified compliance status across all frameworks
#[derive(Debug, Clone)]
pub struct UnifiedComplianceStatus {
    pub timestamp: u64,
    pub hipaa: Option<HipaaComplianceReport>,
    pub gdpr: Option<GdprComplianceReport>,
    pub cmmc: Option<CmmcComplianceReport>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_module_exports() {
        // Verify all engines can be created
        let _hipaa = HipaaComplianceEngine::new();
        let _gdpr = GdprComplianceEngine::new("Test".into());
        let _cmmc = CmmcComplianceEngine::new();
    }
}
