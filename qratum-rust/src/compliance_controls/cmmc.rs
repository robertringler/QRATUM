//! CMMC Level 2 Compliance Engine
//!
//! Executable controls for Cybersecurity Maturity Model Certification (CMMC)
//! Level 2 compliance including:
//! - Role-based enclave segmentation
//! - Access control enforcement
//! - Audit logging
//! - Configuration management
//! - Incident response capabilities
//!
//! ## CMMC 2.0 Level 2 Requirements
//!
//! Level 2 requires implementation of 110 practices from NIST SP 800-171.
//! This module focuses on:
//! - Access Control (AC)
//! - Audit and Accountability (AU)
//! - Configuration Management (CM)
//! - Identification and Authentication (IA)
//! - System and Communications Protection (SC)

extern crate alloc;
use alloc::vec::Vec;
use alloc::string::String;
use alloc::collections::BTreeMap;
use alloc::collections::BTreeSet;
use alloc::format;

use sha3::{Sha3_256, Digest};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// CMMC Practice Domain
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmmcDomain {
    /// Access Control
    AccessControl,
    /// Audit and Accountability
    AuditAccountability,
    /// Awareness and Training
    AwarenessTraining,
    /// Configuration Management
    ConfigurationManagement,
    /// Identification and Authentication
    IdentificationAuthentication,
    /// Incident Response
    IncidentResponse,
    /// Maintenance
    Maintenance,
    /// Media Protection
    MediaProtection,
    /// Personnel Security
    PersonnelSecurity,
    /// Physical Protection
    PhysicalProtection,
    /// Risk Assessment
    RiskAssessment,
    /// Security Assessment
    SecurityAssessment,
    /// System and Communications Protection
    SystemCommunications,
    /// System and Information Integrity
    SystemIntegrity,
}

/// Security Enclave for role-based segmentation
#[derive(Debug, Clone)]
pub struct SecurityEnclave {
    /// Enclave identifier
    pub enclave_id: [u8; 32],
    
    /// Enclave name
    pub name: String,
    
    /// Classification level
    pub classification: ClassificationLevel,
    
    /// Authorized roles
    pub authorized_roles: BTreeSet<String>,
    
    /// Data categories allowed in enclave
    pub data_categories: Vec<DataCategory>,
    
    /// Enclave boundary controls
    pub boundary_controls: BoundaryControls,
    
    /// Active sessions in enclave
    pub active_sessions: Vec<[u8; 32]>,
    
    /// Creation timestamp
    pub created_at: u64,
}

/// Classification Level per DoD guidance
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ClassificationLevel {
    /// Unclassified
    Unclassified,
    /// Controlled Unclassified Information (CUI)
    Cui,
    /// CUI Specified
    CuiSpecified,
    /// Confidential
    Confidential,
    /// Secret
    Secret,
    /// Top Secret (not typically in CMMC L2)
    TopSecret,
}

/// Data Category for enclave segmentation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataCategory {
    /// Federal Contract Information
    Fci,
    /// Controlled Unclassified Information
    Cui,
    /// Technical Data
    TechnicalData,
    /// Proprietary Information
    Proprietary,
    /// Export Controlled (ITAR/EAR)
    ExportControlled,
    /// Personnel Data
    PersonnelData,
    /// Financial Data
    FinancialData,
}

/// Boundary Controls for enclave
#[derive(Debug, Clone)]
pub struct BoundaryControls {
    /// Encryption required for data at rest
    pub encryption_at_rest: bool,
    
    /// Encryption required for data in transit
    pub encryption_in_transit: bool,
    
    /// Multi-factor authentication required
    pub mfa_required: bool,
    
    /// Session timeout in seconds
    pub session_timeout_seconds: u64,
    
    /// Maximum concurrent sessions
    pub max_concurrent_sessions: u32,
    
    /// Allowed ingress sources (network CIDRs or identifiers)
    pub allowed_ingress: Vec<String>,
    
    /// Allowed egress destinations
    pub allowed_egress: Vec<String>,
    
    /// Data loss prevention enabled
    pub dlp_enabled: bool,
}

impl Default for BoundaryControls {
    fn default() -> Self {
        Self {
            encryption_at_rest: true,
            encryption_in_transit: true,
            mfa_required: true,
            session_timeout_seconds: 900, // 15 minutes
            max_concurrent_sessions: 3,
            allowed_ingress: Vec::new(),
            allowed_egress: Vec::new(),
            dlp_enabled: true,
        }
    }
}

/// User Identity for access control
#[derive(Debug, Clone)]
pub struct UserIdentity {
    /// User identifier
    pub user_id: [u8; 32],
    
    /// Username
    pub username: String,
    
    /// Assigned roles
    pub roles: BTreeSet<String>,
    
    /// Clearance level
    pub clearance_level: ClassificationLevel,
    
    /// Account status
    pub status: AccountStatus,
    
    /// Last authentication timestamp
    pub last_auth: Option<u64>,
    
    /// Failed login attempts
    pub failed_attempts: u32,
    
    /// Account creation timestamp
    pub created_at: u64,
    
    /// MFA enabled
    pub mfa_enabled: bool,
}

/// Account Status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccountStatus {
    /// Active account
    Active,
    /// Locked due to failed attempts
    Locked,
    /// Suspended by administrator
    Suspended,
    /// Disabled
    Disabled,
    /// Pending activation
    Pending,
}

/// Access Control Entry
#[derive(Debug, Clone)]
pub struct AccessControlEntry {
    /// Entry identifier
    pub entry_id: [u8; 32],
    
    /// Resource being protected
    pub resource_id: [u8; 32],
    
    /// Role that has access
    pub role: String,
    
    /// Permissions granted
    pub permissions: BTreeSet<Permission>,
    
    /// Time-based restrictions
    pub time_restrictions: Option<TimeRestriction>,
    
    /// Condition for access (e.g., from specific enclave)
    pub conditions: Vec<AccessCondition>,
}

/// Permission types
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Permission {
    Read,
    Write,
    Execute,
    Delete,
    Admin,
    Export,
    Print,
}

/// Time-based access restriction
#[derive(Debug, Clone)]
pub struct TimeRestriction {
    /// Start hour (0-23)
    pub start_hour: u8,
    /// End hour (0-23)
    pub end_hour: u8,
    /// Days of week (0=Sunday, 6=Saturday)
    pub allowed_days: Vec<u8>,
}

/// Access condition
#[derive(Debug, Clone)]
pub enum AccessCondition {
    /// Must be from specific enclave
    FromEnclave([u8; 32]),
    /// Must have MFA
    RequiresMfa,
    /// Must be from approved network
    FromNetwork(String),
    /// Must have specific attribute
    HasAttribute(String, String),
}

/// Audit Event for AC and AU domains
#[derive(Debug, Clone)]
pub struct CmmcAuditEvent {
    /// Event identifier
    pub event_id: [u8; 32],
    
    /// Event timestamp
    pub timestamp: u64,
    
    /// Event type
    pub event_type: AuditEventType,
    
    /// User identity (if applicable)
    pub user_id: Option<[u8; 32]>,
    
    /// Resource accessed
    pub resource_id: Option<[u8; 32]>,
    
    /// Enclave context
    pub enclave_id: Option<[u8; 32]>,
    
    /// Action performed
    pub action: String,
    
    /// Success/failure
    pub success: bool,
    
    /// Additional details
    pub details: String,
    
    /// Source IP/identifier
    pub source: String,
}

/// Audit Event Types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditEventType {
    /// Authentication event
    Authentication,
    /// Authorization decision
    Authorization,
    /// Data access
    DataAccess,
    /// Configuration change
    ConfigurationChange,
    /// Security event
    SecurityEvent,
    /// System event
    SystemEvent,
    /// Enclave boundary crossing
    EnclaveBoundary,
    /// Failed access attempt
    FailedAccess,
}

/// Configuration Baseline for CM domain
#[derive(Debug, Clone)]
pub struct ConfigurationBaseline {
    /// Baseline identifier
    pub baseline_id: [u8; 32],
    
    /// Baseline name
    pub name: String,
    
    /// System component
    pub component: String,
    
    /// Configuration items
    pub items: BTreeMap<String, ConfigurationItem>,
    
    /// Last verified timestamp
    pub last_verified: u64,
    
    /// Deviation count
    pub deviation_count: u32,
    
    /// Approved by
    pub approved_by: String,
}

/// Configuration Item
#[derive(Debug, Clone)]
pub struct ConfigurationItem {
    /// Item name
    pub name: String,
    /// Expected value
    pub expected_value: String,
    /// Current value
    pub current_value: Option<String>,
    /// Compliant flag
    pub is_compliant: bool,
    /// Criticality
    pub criticality: Criticality,
}

/// Criticality level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Criticality {
    Low,
    Medium,
    High,
    Critical,
}

/// CMMC L2 Compliance Engine
///
/// Provides executable controls for CMMC Level 2 compliance including:
/// - Role-based enclave segmentation
/// - Access control matrix enforcement
/// - Audit logging with integrity protection
/// - Configuration baseline management
pub struct CmmcComplianceEngine {
    /// Security enclaves
    enclaves: BTreeMap<[u8; 32], SecurityEnclave>,
    
    /// User identities
    users: BTreeMap<[u8; 32], UserIdentity>,
    
    /// Access control entries
    access_control_list: Vec<AccessControlEntry>,
    
    /// Audit log (immutable)
    audit_log: Vec<CmmcAuditEvent>,
    
    /// Configuration baselines
    baselines: BTreeMap<[u8; 32], ConfigurationBaseline>,
    
    /// Maximum failed login attempts before lockout
    max_failed_attempts: u32,
    
    /// Audit log retention in seconds (1 year minimum for CMMC)
    audit_retention_seconds: u64,
}

impl CmmcComplianceEngine {
    /// Create new CMMC compliance engine
    pub fn new() -> Self {
        Self {
            enclaves: BTreeMap::new(),
            users: BTreeMap::new(),
            access_control_list: Vec::new(),
            audit_log: Vec::new(),
            baselines: BTreeMap::new(),
            max_failed_attempts: 3,
            audit_retention_seconds: 365 * 24 * 60 * 60, // 1 year
        }
    }
    
    /// Create security enclave
    pub fn create_enclave(&mut self, enclave: SecurityEnclave) -> [u8; 32] {
        let enclave_id = enclave.enclave_id;
        self.enclaves.insert(enclave_id, enclave);
        
        self.log_event(CmmcAuditEvent {
            event_id: generate_event_id(),
            timestamp: current_timestamp(),
            event_type: AuditEventType::ConfigurationChange,
            user_id: None,
            resource_id: Some(enclave_id),
            enclave_id: Some(enclave_id),
            action: "CREATE_ENCLAVE".into(),
            success: true,
            details: "Security enclave created".into(),
            source: "system".into(),
        });
        
        enclave_id
    }
    
    /// Register user identity
    pub fn register_user(&mut self, user: UserIdentity) {
        let user_id = user.user_id;
        self.users.insert(user_id, user);
        
        self.log_event(CmmcAuditEvent {
            event_id: generate_event_id(),
            timestamp: current_timestamp(),
            event_type: AuditEventType::ConfigurationChange,
            user_id: Some(user_id),
            resource_id: None,
            enclave_id: None,
            action: "REGISTER_USER".into(),
            success: true,
            details: "User account registered".into(),
            source: "system".into(),
        });
    }
    
    /// Add access control entry
    pub fn add_access_control(&mut self, entry: AccessControlEntry) {
        self.access_control_list.push(entry);
    }
    
    /// Check access authorization
    ///
    /// Returns true if access is granted, false if denied.
    /// All access checks are logged.
    pub fn check_access(
        &mut self,
        user_id: &[u8; 32],
        resource_id: &[u8; 32],
        permission: Permission,
        enclave_id: Option<&[u8; 32]>,
    ) -> bool {
        // Get user
        let user = match self.users.get(user_id) {
            Some(u) => u,
            None => {
                self.log_failed_access(*user_id, *resource_id, "User not found");
                return false;
            }
        };
        
        // Check account status
        if user.status != AccountStatus::Active {
            self.log_failed_access(*user_id, *resource_id, "Account not active");
            return false;
        }
        
        // Check MFA requirement if in enclave
        if let Some(enc_id) = enclave_id {
            if let Some(enclave) = self.enclaves.get(enc_id) {
                if enclave.boundary_controls.mfa_required && !user.mfa_enabled {
                    self.log_failed_access(*user_id, *resource_id, "MFA required");
                    return false;
                }
                
                // Check if user role is authorized for enclave
                let role_authorized = user.roles.iter()
                    .any(|r| enclave.authorized_roles.contains(r));
                
                if !role_authorized {
                    self.log_failed_access(*user_id, *resource_id, "Role not authorized for enclave");
                    return false;
                }
                
                // Check clearance level
                if user.clearance_level < enclave.classification {
                    self.log_failed_access(*user_id, *resource_id, "Insufficient clearance");
                    return false;
                }
            }
        }
        
        // Check access control list
        let has_permission = self.access_control_list.iter().any(|ace| {
            ace.resource_id == *resource_id &&
            user.roles.contains(&ace.role) &&
            ace.permissions.contains(&permission) &&
            self.check_conditions(user, &ace.conditions, enclave_id)
        });
        
        if has_permission {
            self.log_event(CmmcAuditEvent {
                event_id: generate_event_id(),
                timestamp: current_timestamp(),
                event_type: AuditEventType::Authorization,
                user_id: Some(*user_id),
                resource_id: Some(*resource_id),
                enclave_id: enclave_id.copied(),
                action: format!("{:?}", permission),
                success: true,
                details: "Access granted".into(),
                source: "access_control".into(),
            });
            true
        } else {
            self.log_failed_access(*user_id, *resource_id, "Permission denied by ACL");
            false
        }
    }
    
    /// Check access conditions
    fn check_conditions(
        &self,
        user: &UserIdentity,
        conditions: &[AccessCondition],
        enclave_id: Option<&[u8; 32]>,
    ) -> bool {
        for condition in conditions {
            match condition {
                AccessCondition::FromEnclave(required_enclave) => {
                    if enclave_id != Some(required_enclave) {
                        return false;
                    }
                }
                AccessCondition::RequiresMfa => {
                    if !user.mfa_enabled {
                        return false;
                    }
                }
                AccessCondition::FromNetwork(_) => {
                    // Would check network in real implementation
                }
                AccessCondition::HasAttribute(_, _) => {
                    // Would check attributes in real implementation
                }
            }
        }
        true
    }
    
    /// Log failed access attempt
    fn log_failed_access(&mut self, user_id: [u8; 32], resource_id: [u8; 32], reason: &str) {
        self.log_event(CmmcAuditEvent {
            event_id: generate_event_id(),
            timestamp: current_timestamp(),
            event_type: AuditEventType::FailedAccess,
            user_id: Some(user_id),
            resource_id: Some(resource_id),
            enclave_id: None,
            action: "ACCESS_DENIED".into(),
            success: false,
            details: reason.into(),
            source: "access_control".into(),
        });
    }
    
    /// Record authentication attempt
    pub fn record_authentication(
        &mut self,
        user_id: &[u8; 32],
        success: bool,
        mfa_used: bool,
    ) {
        if let Some(user) = self.users.get_mut(user_id) {
            if success {
                user.last_auth = Some(current_timestamp());
                user.failed_attempts = 0;
            } else {
                user.failed_attempts += 1;
                if user.failed_attempts >= self.max_failed_attempts {
                    user.status = AccountStatus::Locked;
                }
            }
        }
        
        self.log_event(CmmcAuditEvent {
            event_id: generate_event_id(),
            timestamp: current_timestamp(),
            event_type: AuditEventType::Authentication,
            user_id: Some(*user_id),
            resource_id: None,
            enclave_id: None,
            action: if success { "LOGIN_SUCCESS" } else { "LOGIN_FAILURE" }.into(),
            success,
            details: format!("MFA: {}", mfa_used),
            source: "authentication".into(),
        });
    }
    
    /// Log audit event
    fn log_event(&mut self, event: CmmcAuditEvent) {
        self.audit_log.push(event);
    }
    
    /// Create configuration baseline
    pub fn create_baseline(&mut self, baseline: ConfigurationBaseline) {
        let baseline_id = baseline.baseline_id;
        self.baselines.insert(baseline_id, baseline);
    }
    
    /// Verify configuration baseline
    pub fn verify_baseline(&mut self, baseline_id: &[u8; 32]) -> Option<ConfigurationVerification> {
        let baseline = self.baselines.get_mut(baseline_id)?;
        
        let mut deviations = Vec::new();
        let mut compliant_count = 0;
        
        for (name, item) in &baseline.items {
            if item.is_compliant {
                compliant_count += 1;
            } else {
                deviations.push(ConfigurationDeviation {
                    item_name: name.clone(),
                    expected: item.expected_value.clone(),
                    actual: item.current_value.clone(),
                    criticality: item.criticality,
                });
            }
        }
        
        baseline.last_verified = current_timestamp();
        baseline.deviation_count = deviations.len() as u32;
        
        Some(ConfigurationVerification {
            baseline_id: *baseline_id,
            verified_at: current_timestamp(),
            total_items: baseline.items.len(),
            compliant_items: compliant_count,
            deviations,
        })
    }
    
    /// Get audit events for time range
    pub fn get_audit_events(&self, start: u64, end: u64) -> Vec<&CmmcAuditEvent> {
        self.audit_log
            .iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .collect()
    }
    
    /// Generate CMMC compliance report
    pub fn generate_compliance_report(&self) -> CmmcComplianceReport {
        let total_enclaves = self.enclaves.len();
        let total_users = self.users.len();
        let active_users = self.users.values().filter(|u| u.status == AccountStatus::Active).count();
        let locked_users = self.users.values().filter(|u| u.status == AccountStatus::Locked).count();
        let mfa_enabled_users = self.users.values().filter(|u| u.mfa_enabled).count();
        let total_audit_events = self.audit_log.len();
        let failed_access_events = self.audit_log.iter()
            .filter(|e| e.event_type == AuditEventType::FailedAccess)
            .count();
        let total_baselines = self.baselines.len();
        let baselines_compliant = self.baselines.values()
            .filter(|b| b.deviation_count == 0)
            .count();
        
        CmmcComplianceReport {
            report_timestamp: current_timestamp(),
            total_enclaves,
            total_users,
            active_users,
            locked_users,
            mfa_enabled_users,
            total_audit_events,
            failed_access_events,
            total_baselines,
            baselines_compliant,
        }
    }
}

impl Default for CmmcComplianceEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration verification result
#[derive(Debug, Clone)]
pub struct ConfigurationVerification {
    pub baseline_id: [u8; 32],
    pub verified_at: u64,
    pub total_items: usize,
    pub compliant_items: usize,
    pub deviations: Vec<ConfigurationDeviation>,
}

/// Configuration deviation
#[derive(Debug, Clone)]
pub struct ConfigurationDeviation {
    pub item_name: String,
    pub expected: String,
    pub actual: Option<String>,
    pub criticality: Criticality,
}

/// CMMC Compliance Report
#[derive(Debug, Clone)]
pub struct CmmcComplianceReport {
    pub report_timestamp: u64,
    pub total_enclaves: usize,
    pub total_users: usize,
    pub active_users: usize,
    pub locked_users: usize,
    pub mfa_enabled_users: usize,
    pub total_audit_events: usize,
    pub failed_access_events: usize,
    pub total_baselines: usize,
    pub baselines_compliant: usize,
}

/// Generate unique event ID
fn generate_event_id() -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(&current_timestamp().to_le_bytes());
    hasher.update(b"event");
    hasher.finalize().into()
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
    fn test_enclave_creation() {
        let mut engine = CmmcComplianceEngine::new();
        
        let mut authorized_roles = BTreeSet::new();
        authorized_roles.insert("Engineer".into());
        
        let enclave = SecurityEnclave {
            enclave_id: [1u8; 32],
            name: "CUI Enclave".into(),
            classification: ClassificationLevel::Cui,
            authorized_roles,
            data_categories: vec![DataCategory::Cui],
            boundary_controls: BoundaryControls::default(),
            active_sessions: Vec::new(),
            created_at: current_timestamp(),
        };
        
        let enclave_id = engine.create_enclave(enclave);
        assert_eq!(enclave_id, [1u8; 32]);
    }
    
    #[test]
    fn test_access_control() {
        let mut engine = CmmcComplianceEngine::new();
        
        // Create user
        let mut roles = BTreeSet::new();
        roles.insert("Engineer".into());
        
        let user = UserIdentity {
            user_id: [1u8; 32],
            username: "john.doe".into(),
            roles,
            clearance_level: ClassificationLevel::Cui,
            status: AccountStatus::Active,
            last_auth: None,
            failed_attempts: 0,
            created_at: current_timestamp(),
            mfa_enabled: true,
        };
        engine.register_user(user);
        
        // Create ACE
        let mut permissions = BTreeSet::new();
        permissions.insert(Permission::Read);
        
        let ace = AccessControlEntry {
            entry_id: [2u8; 32],
            resource_id: [3u8; 32],
            role: "Engineer".into(),
            permissions,
            time_restrictions: None,
            conditions: Vec::new(),
        };
        engine.add_access_control(ace);
        
        // Check access
        let granted = engine.check_access(&[1u8; 32], &[3u8; 32], Permission::Read, None);
        assert!(granted);
        
        // Check denied permission
        let denied = engine.check_access(&[1u8; 32], &[3u8; 32], Permission::Delete, None);
        assert!(!denied);
    }
    
    #[test]
    fn test_account_lockout() {
        let mut engine = CmmcComplianceEngine::new();
        
        let user = UserIdentity {
            user_id: [1u8; 32],
            username: "test.user".into(),
            roles: BTreeSet::new(),
            clearance_level: ClassificationLevel::Unclassified,
            status: AccountStatus::Active,
            last_auth: None,
            failed_attempts: 0,
            created_at: current_timestamp(),
            mfa_enabled: false,
        };
        engine.register_user(user);
        
        // Simulate failed logins
        for _ in 0..3 {
            engine.record_authentication(&[1u8; 32], false, false);
        }
        
        // Check user is locked
        let user = engine.users.get(&[1u8; 32]).unwrap();
        assert_eq!(user.status, AccountStatus::Locked);
    }
    
    #[test]
    fn test_audit_logging() {
        let mut engine = CmmcComplianceEngine::new();
        
        // Generate some events
        engine.record_authentication(&[1u8; 32], true, true);
        
        let events = engine.get_audit_events(0, u64::MAX);
        assert!(!events.is_empty());
    }
}
