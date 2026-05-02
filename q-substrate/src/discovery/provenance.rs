//! Provenance Integration Module
//!
//! Integrates with QRADLE for provenance chain tracking
//! Generates and verifies provenance hashes for discoveries

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use super::types::{Discovery, DiscoveryError};
use crate::wasm_pod::ProvenanceEntry;

/// Generate deterministic provenance hash for discovery
///
/// Hash is based on discovery content and seed for determinism
pub fn generate_provenance_hash(discovery: &Discovery) -> String {
    // Simple deterministic hash based on discovery properties
    let mut hash_input = 0u64;
    
    // Hash ID
    for byte in discovery.id.bytes() {
        hash_input = hash_input.wrapping_mul(31).wrapping_add(byte as u64);
    }
    
    // Hash title
    for byte in discovery.title.bytes() {
        hash_input = hash_input.wrapping_mul(31).wrapping_add(byte as u64);
    }
    
    // Hash seed
    hash_input = hash_input.wrapping_mul(31).wrapping_add(discovery.provenance.seed as u64);
    
    // Hash fitness score (scaled to integer)
    let fitness_int = (discovery.fitness_score * 1000000.0) as u64;
    hash_input = hash_input.wrapping_mul(31).wrapping_add(fitness_int);
    
    // Generate final hash with additional mixing
    let hash = hash_input
        .wrapping_mul(0x517cc1b727220a95)
        .wrapping_add(0x63f5d5a6a9e1a3c7);
    
    alloc::format!("QRDL-{:016x}", hash)
}

/// Verify provenance chain integrity
///
/// Checks that all discoveries have valid provenance and form a valid chain
pub fn verify_provenance_chain(discoveries: &[Discovery]) -> Result<bool, DiscoveryError> {
    if discoveries.is_empty() {
        return Ok(true);
    }
    
    // Check each discovery has valid provenance
    for discovery in discoveries {
        // Verify hash format
        if !discovery.provenance.qradle_hash.starts_with("QRDL-") {
            return Err(DiscoveryError::ProvenanceFailed(
                format!("Invalid hash format for {}", discovery.id)
            ));
        }
        
        // Verify hash matches content
        let computed_hash = generate_provenance_hash(discovery);
        if computed_hash != discovery.provenance.qradle_hash {
            return Err(DiscoveryError::ProvenanceFailed(
                format!("Hash mismatch for {}: expected {}, got {}", 
                    discovery.id, discovery.provenance.qradle_hash, computed_hash)
            ));
        }
        
        // Verify ID is valid
        if !discovery.has_valid_id() {
            return Err(DiscoveryError::ProvenanceFailed(
                format!("Invalid ID format: {}", discovery.id)
            ));
        }
        
        // Verify fitness score
        if !discovery.is_valid() {
            return Err(DiscoveryError::ProvenanceFailed(
                format!("Discovery {} has invalid fitness score: {}", 
                    discovery.id, discovery.fitness_score)
            ));
        }
    }
    
    // Check ordering (QRD-001, QRD-002, etc.)
    for (i, discovery) in discoveries.iter().enumerate() {
        let expected_id = alloc::format!("QRD-{:03}", i + 1);
        if discovery.id != expected_id {
            return Err(DiscoveryError::ProvenanceFailed(
                format!("Discovery order violation: expected {}, got {}", 
                    expected_id, discovery.id)
            ));
        }
    }
    
    Ok(true)
}

/// Record discovery to QRADLE ledger
///
/// Creates a provenance entry for audit trail
pub fn record_to_qradle(discovery: &Discovery) -> Result<ProvenanceEntry, DiscoveryError> {
    // Create provenance entry
    let entry = ProvenanceEntry {
        source: "discovery_engine".into(),
        target: Some("qradle_ledger".into()),
        operation: format!("generate_discovery_{}", discovery.id),
        input_hash: hash_string(&discovery.hypothesis),
        output_hash: hash_string(&discovery.provenance.qradle_hash),
        timestamp: parse_timestamp(&discovery.provenance.generated_at),
        duration_us: 0, // Not measured in this context
    };
    
    Ok(entry)
}

/// Simple string hash function
fn hash_string(s: &str) -> u64 {
    let mut hash = 0u64;
    for byte in s.bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
    }
    hash
}

/// Parse ISO 8601 timestamp to simple counter
fn parse_timestamp(timestamp: &str) -> u64 {
    // Simple extraction: count total minutes since epoch
    // Format: YYYY-MM-DDTHH:MM:SSZ
    if timestamp.len() < 19 {
        return 0;
    }
    
    // Extract day, hour, minute
    let day = timestamp[8..10].parse::<u64>().unwrap_or(1);
    let hour = timestamp[11..13].parse::<u64>().unwrap_or(0);
    let minute = timestamp[14..16].parse::<u64>().unwrap_or(0);
    
    day * 1440 + hour * 60 + minute
}

/// Provenance verification report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceReport {
    pub total_discoveries: usize,
    pub valid_hashes: usize,
    pub invalid_hashes: usize,
    pub chain_valid: bool,
    pub errors: Vec<String>,
}

/// Generate provenance report for a set of discoveries
pub fn generate_provenance_report(discoveries: &[Discovery]) -> ProvenanceReport {
    let mut valid_hashes = 0;
    let mut invalid_hashes = 0;
    let mut errors = Vec::new();
    
    for discovery in discoveries {
        let computed = generate_provenance_hash(discovery);
        if computed == discovery.provenance.qradle_hash {
            valid_hashes += 1;
        } else {
            invalid_hashes += 1;
            errors.push(alloc::format!(
                "Hash mismatch for {}: expected {}, got {}",
                discovery.id, discovery.provenance.qradle_hash, computed
            ));
        }
    }
    
    let chain_valid = match verify_provenance_chain(discoveries) {
        Ok(valid) => valid,
        Err(e) => {
            errors.push(alloc::format!("Chain verification failed: {}", e));
            false
        }
    };
    
    ProvenanceReport {
        total_discoveries: discoveries.len(),
        valid_hashes,
        invalid_hashes,
        chain_valid,
        errors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::types::{
        Formulation, IndustrialImpact, Provenance, RiskEnvelope, ValidationMethod, ValidationPath,
    };

    fn create_test_discovery() -> Discovery {
        Discovery {
            id: "QRD-001".into(),
            title: "Test Discovery".into(),
            hypothesis: "Test hypothesis".into(),
            core_mechanism: "Test mechanism".into(),
            formulation: Formulation {
                equations: Vec::new(),
                pseudocode: None,
                formal_spec: None,
            },
            validation: ValidationPath {
                method: ValidationMethod::Simulation,
                test_rig: "Test rig".into(),
                expected_outcome: "Test outcome".into(),
                confidence: 0.9,
            },
            industrial_impact: IndustrialImpact {
                application: "Test app".into(),
                market_sector: "Test sector".into(),
                estimated_value: None,
            },
            risk_envelope: RiskEnvelope {
                failure_modes: Vec::new(),
                safety_constraints: Vec::new(),
                mitigation_strategies: Vec::new(),
            },
            fitness_score: 0.95,
            provenance: Provenance {
                generated_at: "2025-01-01T00:00:00Z".into(),
                qradle_hash: "placeholder".into(),
                seed: 42,
                lattice_node: "test_node".into(),
            },
        }
    }

    #[test]
    fn test_generate_provenance_hash() {
        let discovery = create_test_discovery();
        let hash = generate_provenance_hash(&discovery);
        
        assert!(hash.starts_with("QRDL-"));
        assert_eq!(hash.len(), 21); // "QRDL-" + 16 hex chars
    }

    #[test]
    fn test_hash_determinism() {
        let discovery = create_test_discovery();
        let hash1 = generate_provenance_hash(&discovery);
        let hash2 = generate_provenance_hash(&discovery);
        
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_verify_provenance_chain_empty() {
        let discoveries: Vec<Discovery> = Vec::new();
        let result = verify_provenance_chain(&discoveries);
        
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_verify_provenance_chain_valid() {
        let mut discovery = create_test_discovery();
        discovery.provenance.qradle_hash = generate_provenance_hash(&discovery);
        
        let discoveries = alloc::vec![discovery];
        let result = verify_provenance_chain(&discoveries);
        
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_verify_provenance_chain_invalid_hash() {
        let mut discovery = create_test_discovery();
        discovery.provenance.qradle_hash = "QRDL-invalid".into();
        
        let discoveries = alloc::vec![discovery];
        let result = verify_provenance_chain(&discoveries);
        
        assert!(result.is_err());
    }

    #[test]
    fn test_record_to_qradle() {
        let mut discovery = create_test_discovery();
        discovery.provenance.qradle_hash = generate_provenance_hash(&discovery);
        
        let result = record_to_qradle(&discovery);
        assert!(result.is_ok());
        
        let entry = result.unwrap();
        assert_eq!(entry.source, "discovery_engine");
        assert!(entry.target.is_some());
    }

    #[test]
    fn test_provenance_report() {
        let mut discovery = create_test_discovery();
        discovery.provenance.qradle_hash = generate_provenance_hash(&discovery);
        
        let discoveries = alloc::vec![discovery];
        let report = generate_provenance_report(&discoveries);
        
        assert_eq!(report.total_discoveries, 1);
        assert_eq!(report.valid_hashes, 1);
        assert_eq!(report.invalid_hashes, 0);
        assert!(report.chain_valid);
    }
}
