//! Audit and Provenance Module
//!
//! Full audit trail for deterministic execution:
//! - Operation logging
//! - Provenance records
//! - Invariant verification
//! - Rollback support
//!
//! All operations are logged with deterministic timestamps

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

/// Audit entry for operation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Entry sequence number
    pub sequence: u64,
    /// Operation name
    pub operation: String,
    /// Operation count for this type
    pub op_count: usize,
    /// Timestamp (deterministic counter)
    pub timestamp: u64,
    /// Module that performed operation
    pub module: String,
    /// Input hash (for verification)
    pub input_hash: Option<u64>,
    /// Output hash (for verification)
    pub output_hash: Option<u64>,
    /// Success status
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Provenance record for inter-module calls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    /// Source module
    pub source: String,
    /// Target module (if applicable)
    pub target: Option<String>,
    /// Operation type
    pub operation: String,
    /// Timestamp
    pub timestamp: u64,
    /// Duration in microseconds
    pub duration_us: u64,
    /// Memory used
    pub memory_used: usize,
}

/// Invariant check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvariantCheck {
    /// Invariant name
    pub name: String,
    /// Check passed
    pub passed: bool,
    /// Expected value
    pub expected: String,
    /// Actual value
    pub actual: String,
    /// Timestamp
    pub timestamp: u64,
}

/// Rollback point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackPoint {
    /// Point ID
    pub id: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Sequence number at this point
    pub sequence: u64,
    /// State hash
    pub state_hash: u64,
    /// Description
    pub description: String,
}

/// Audit log manager
pub struct AuditLog {
    /// Audit entries
    entries: Vec<AuditEntry>,
    /// Provenance records
    provenance: Vec<ProvenanceRecord>,
    /// Invariant checks
    invariant_checks: Vec<InvariantCheck>,
    /// Rollback points
    rollback_points: Vec<RollbackPoint>,
    /// Current sequence number
    sequence: u64,
    /// Current timestamp
    timestamp: u64,
    /// Is logging enabled
    enabled: bool,
}

impl AuditLog {
    /// Create new audit log
    pub fn new() -> Self {
        AuditLog {
            entries: Vec::new(),
            provenance: Vec::new(),
            invariant_checks: Vec::new(),
            rollback_points: Vec::new(),
            sequence: 0,
            timestamp: 0,
            enabled: true,
        }
    }

    /// Enable/disable logging
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Log an operation
    pub fn log_operation(&mut self, operation: &str, op_count: usize) {
        if !self.enabled {
            return;
        }
        
        self.sequence += 1;
        self.timestamp += 1;
        
        self.entries.push(AuditEntry {
            sequence: self.sequence,
            operation: operation.into(),
            op_count,
            timestamp: self.timestamp,
            module: "qsubstrate".into(),
            input_hash: None,
            output_hash: None,
            success: true,
            error: None,
        });
    }

    /// Log operation with hashes
    pub fn log_operation_with_hash(
        &mut self,
        operation: &str,
        module: &str,
        input_hash: u64,
        output_hash: u64,
        success: bool,
        error: Option<String>,
    ) {
        if !self.enabled {
            return;
        }
        
        self.sequence += 1;
        self.timestamp += 1;
        
        self.entries.push(AuditEntry {
            sequence: self.sequence,
            operation: operation.into(),
            op_count: 1,
            timestamp: self.timestamp,
            module: module.into(),
            input_hash: Some(input_hash),
            output_hash: Some(output_hash),
            success,
            error,
        });
    }

    /// Record provenance
    pub fn record_provenance(
        &mut self,
        source: &str,
        target: Option<&str>,
        operation: &str,
        duration_us: u64,
        memory_used: usize,
    ) {
        if !self.enabled {
            return;
        }
        
        self.timestamp += 1;
        
        self.provenance.push(ProvenanceRecord {
            source: source.into(),
            target: target.map(|s| s.into()),
            operation: operation.into(),
            timestamp: self.timestamp,
            duration_us,
            memory_used,
        });
    }

    /// Check invariant
    pub fn check_invariant(&mut self, name: &str, expected: &str, actual: &str) -> bool {
        self.timestamp += 1;
        
        let passed = expected == actual;
        
        self.invariant_checks.push(InvariantCheck {
            name: name.into(),
            passed,
            expected: expected.into(),
            actual: actual.into(),
            timestamp: self.timestamp,
        });
        
        passed
    }

    /// Create rollback point
    pub fn create_rollback_point(&mut self, description: &str, state_hash: u64) -> u64 {
        self.timestamp += 1;
        
        let id = self.rollback_points.len() as u64;
        
        self.rollback_points.push(RollbackPoint {
            id,
            timestamp: self.timestamp,
            sequence: self.sequence,
            state_hash,
            description: description.into(),
        });
        
        id
    }

    /// Get entries since rollback point
    pub fn get_entries_since_rollback(&self, point_id: u64) -> Vec<&AuditEntry> {
        if let Some(point) = self.rollback_points.get(point_id as usize) {
            self.entries
                .iter()
                .filter(|e| e.sequence > point.sequence)
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get all entries
    pub fn get_entries(&self) -> &[AuditEntry] {
        &self.entries
    }

    /// Get all provenance records
    pub fn get_provenance(&self) -> &[ProvenanceRecord] {
        &self.provenance
    }

    /// Get all invariant checks
    pub fn get_invariant_checks(&self) -> &[InvariantCheck] {
        &self.invariant_checks
    }

    /// Get all rollback points
    pub fn get_rollback_points(&self) -> &[RollbackPoint] {
        &self.rollback_points
    }

    /// Get current sequence number
    pub fn get_sequence(&self) -> u64 {
        self.sequence
    }

    /// Get current timestamp
    pub fn get_timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Get summary statistics
    pub fn get_summary(&self) -> AuditSummary {
        let total_ops = self.entries.len();
        let successful_ops = self.entries.iter().filter(|e| e.success).count();
        let failed_ops = total_ops - successful_ops;
        
        let invariants_passed = self.invariant_checks.iter().filter(|c| c.passed).count();
        let invariants_failed = self.invariant_checks.len() - invariants_passed;
        
        AuditSummary {
            total_entries: total_ops,
            successful_ops,
            failed_ops,
            provenance_records: self.provenance.len(),
            invariants_passed,
            invariants_failed,
            rollback_points: self.rollback_points.len(),
            current_sequence: self.sequence,
            current_timestamp: self.timestamp,
        }
    }

    /// Clear all logs
    pub fn clear(&mut self) {
        self.entries.clear();
        self.provenance.clear();
        self.invariant_checks.clear();
        self.rollback_points.clear();
        self.sequence = 0;
        self.timestamp = 0;
    }

    /// Verify determinism by checking all operations are logged
    pub fn verify_determinism(&self) -> bool {
        // Check sequence is continuous
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.sequence != (i + 1) as u64 {
                return false;
            }
        }
        true
    }
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new()
    }
}

/// Audit summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditSummary {
    /// Total audit entries
    pub total_entries: usize,
    /// Successful operations
    pub successful_ops: usize,
    /// Failed operations
    pub failed_ops: usize,
    /// Provenance records count
    pub provenance_records: usize,
    /// Invariants passed
    pub invariants_passed: usize,
    /// Invariants failed
    pub invariants_failed: usize,
    /// Rollback points count
    pub rollback_points: usize,
    /// Current sequence number
    pub current_sequence: u64,
    /// Current timestamp
    pub current_timestamp: u64,
}

/// 8 Fatal Invariants for Q-Substrate
pub mod invariants {
    /// Invariant 1: ℛ(t) ≥ 0 (Non-negative resources)
    pub const RESOURCE_POSITIVE: &str = "R(t) >= 0";
    
    /// Invariant 2: Deterministic execution
    pub const DETERMINISTIC_EXEC: &str = "deterministic_execution";
    
    /// Invariant 3: No side-channel leaks
    pub const NO_SIDE_CHANNELS: &str = "no_side_channels";
    
    /// Invariant 4: Pod isolation
    pub const POD_ISOLATION: &str = "pod_isolation";
    
    /// Invariant 5: Rollback compatibility
    pub const ROLLBACK_COMPAT: &str = "rollback_compatible";
    
    /// Invariant 6: Audit trail complete
    pub const AUDIT_COMPLETE: &str = "audit_trail_complete";
    
    /// Invariant 7: Memory bounds
    pub const MEMORY_BOUNDS: &str = "memory_bounds";
    
    /// Invariant 8: No goal drift
    pub const NO_GOAL_DRIFT: &str = "no_goal_drift";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_log_creation() {
        let log = AuditLog::new();
        assert_eq!(log.get_sequence(), 0);
        assert!(log.get_entries().is_empty());
    }

    #[test]
    fn test_log_operation() {
        let mut log = AuditLog::new();
        log.log_operation("test_op", 1);
        
        assert_eq!(log.get_entries().len(), 1);
        assert_eq!(log.get_entries()[0].operation, "test_op");
        assert_eq!(log.get_sequence(), 1);
    }

    #[test]
    fn test_provenance_recording() {
        let mut log = AuditLog::new();
        log.record_provenance("ai_pod", Some("quantum_pod"), "inference", 100, 1024);
        
        assert_eq!(log.get_provenance().len(), 1);
        assert_eq!(log.get_provenance()[0].source, "ai_pod");
    }

    #[test]
    fn test_invariant_check() {
        let mut log = AuditLog::new();
        
        let passed = log.check_invariant("test", "expected", "expected");
        assert!(passed);
        
        let failed = log.check_invariant("test2", "expected", "actual");
        assert!(!failed);
    }

    #[test]
    fn test_rollback_point() {
        let mut log = AuditLog::new();
        
        log.log_operation("op1", 1);
        log.log_operation("op2", 1);
        
        let point_id = log.create_rollback_point("checkpoint", 12345);
        
        log.log_operation("op3", 1);
        log.log_operation("op4", 1);
        
        let entries_since = log.get_entries_since_rollback(point_id);
        assert_eq!(entries_since.len(), 2);
    }

    #[test]
    fn test_summary() {
        let mut log = AuditLog::new();
        log.log_operation("op1", 1);
        log.log_operation("op2", 1);
        log.check_invariant("test", "a", "a");
        
        let summary = log.get_summary();
        assert_eq!(summary.total_entries, 2);
        assert_eq!(summary.invariants_passed, 1);
    }

    #[test]
    fn test_verify_determinism() {
        let mut log = AuditLog::new();
        log.log_operation("op1", 1);
        log.log_operation("op2", 1);
        log.log_operation("op3", 1);
        
        assert!(log.verify_determinism());
    }
}
