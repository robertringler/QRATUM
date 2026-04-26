//! Discovery CLI - Command Line Interface
//!
//! Provides command-line interface for running discovery directive

extern crate alloc;

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use super::engine::DiscoveryEngine;
use super::provenance::verify_provenance_chain;
use super::types::{Discovery, DiscoveryError};

/// Discovery execution report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryReport {
    pub total_candidates_evaluated: usize,
    pub discoveries_generated: usize,
    pub discoveries_validated: usize,
    pub average_fitness: f64,
    pub execution_time_ms: u64,
}

/// Run discovery directive from command line
///
/// Generates discoveries and optionally writes to output directory
pub fn run_discovery_directive(
    seed: u32,
    target_count: usize,
    output_dir: Option<&str>,
) -> Result<DiscoveryReport, DiscoveryError> {
    let start_time = get_time_ms();
    
    // Create and run discovery engine
    let mut engine = DiscoveryEngine::with_target(seed, target_count);
    
    let discoveries = engine.run()?;
    
    // Verify provenance chain
    verify_provenance_chain(&discoveries)?;
    
    // Calculate statistics
    let total_candidates = engine.get_discoveries().len();
    let valid_count = discoveries
        .iter()
        .filter(|d| d.fitness_score >= 0.87)
        .count();
    
    let avg_fitness = if !discoveries.is_empty() {
        discoveries.iter().map(|d| d.fitness_score).sum::<f64>() / discoveries.len() as f64
    } else {
        0.0
    };
    
    let end_time = get_time_ms();
    let execution_time = end_time - start_time;
    
    // Write to output directory if specified
    if let Some(dir) = output_dir {
        write_discoveries_to_dir(&discoveries, dir)?;
    }
    
    Ok(DiscoveryReport {
        total_candidates_evaluated: total_candidates,
        discoveries_generated: discoveries.len(),
        discoveries_validated: valid_count,
        average_fitness: avg_fitness,
        execution_time_ms: execution_time,
    })
}

/// Write discoveries to directory structure
///
/// Creates JSON files directly in the base directory (pending)
/// These will be moved to validated/ or rejected/ by the archive phase
#[cfg(feature = "std")]
fn write_discoveries_to_dir(discoveries: &[Discovery], base_dir: &str) -> Result<(), DiscoveryError> {
    use std::fs;
    
    // Create base directory if it doesn't exist
    if let Err(e) = fs::create_dir_all(base_dir) {
        return Err(DiscoveryError::Generic(alloc::format!("Failed to create directory: {}", e)));
    }
    
    for discovery in discoveries {
        let filename = alloc::format!("{}/{}.json", base_dir, discovery.id);
        
        // Serialize discovery to JSON
        let json = serde_json::to_string_pretty(discovery)
            .map_err(|e| DiscoveryError::SerializationError(alloc::format!("{}", e)))?;
        
        // Write to file
        if let Err(e) = fs::write(&filename, &json) {
            return Err(DiscoveryError::Generic(alloc::format!("Failed to write {}: {}", filename, e)));
        }
    }
    
    Ok(())
}

/// Write discoveries to directory structure (no_std version)
///
/// Returns an error since file I/O is not available without std
#[cfg(not(feature = "std"))]
fn write_discoveries_to_dir(_discoveries: &[Discovery], _base_dir: &str) -> Result<(), DiscoveryError> {
    // File I/O is not available in no_std environments
    // Return an error to prevent silent data loss
    Err(DiscoveryError::Generic(
        "File I/O not available in no_std environment - discoveries cannot be written".into()
    ))
}

/// Get current time in milliseconds (simplified for deterministic execution)
fn get_time_ms() -> u64 {
    // In a real implementation, this would use actual time
    // For deterministic execution, we use a counter
    0
}

/// Print discovery report to string
pub fn format_report(report: &DiscoveryReport) -> String {
    alloc::format!(
        "Discovery Report:\n\
         - Total Candidates Evaluated: {}\n\
         - Discoveries Generated: {}\n\
         - Discoveries Validated: {}\n\
         - Average Fitness: {:.3}\n\
         - Execution Time: {} ms\n",
        report.total_candidates_evaluated,
        report.discoveries_generated,
        report.discoveries_validated,
        report.average_fitness,
        report.execution_time_ms
    )
}

/// Validate discovery against schema
pub fn validate_discovery_schema(discovery: &Discovery) -> Result<(), DiscoveryError> {
    // Check ID format
    if !discovery.has_valid_id() {
        return Err(DiscoveryError::ValidationError(
            format!("Invalid ID format: {}", discovery.id)
        ));
    }
    
    // Check title length
    if discovery.title.len() < 10 {
        return Err(DiscoveryError::ValidationError(
            "Title must be at least 10 characters".into()
        ));
    }
    
    // Check fitness score
    if !discovery.is_valid() {
        return Err(DiscoveryError::ValidationError(
            format!("Fitness score {} is out of valid range [0.87, 1.0]", discovery.fitness_score)
        ));
    }
    
    // Check required fields are not empty
    if discovery.hypothesis.is_empty() {
        return Err(DiscoveryError::ValidationError("Hypothesis is empty".into()));
    }
    
    if discovery.core_mechanism.is_empty() {
        return Err(DiscoveryError::ValidationError("Core mechanism is empty".into()));
    }
    
    Ok(())
}

/// Export discoveries to JSON string
pub fn export_discoveries_json(discoveries: &[Discovery]) -> Result<String, DiscoveryError> {
    serde_json::to_string_pretty(discoveries)
        .map_err(|e| DiscoveryError::SerializationError(alloc::format!("{}", e)))
}

/// Import discoveries from JSON string
pub fn import_discoveries_json(json: &str) -> Result<Vec<Discovery>, DiscoveryError> {
    serde_json::from_str(json)
        .map_err(|e| DiscoveryError::SerializationError(alloc::format!("{}", e)))
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
            title: "Test Discovery Title".into(),
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
                qradle_hash: "QRDL-0123456789abcdef".into(),
                seed: 42,
                lattice_node: "test_node".into(),
            },
        }
    }

    #[test]
    fn test_validate_discovery_schema_valid() {
        let discovery = create_test_discovery();
        let result = validate_discovery_schema(&discovery);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_discovery_schema_invalid_id() {
        let mut discovery = create_test_discovery();
        discovery.id = "INVALID".into();
        
        let result = validate_discovery_schema(&discovery);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_discovery_schema_short_title() {
        let mut discovery = create_test_discovery();
        discovery.title = "Short".into();
        
        let result = validate_discovery_schema(&discovery);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_discovery_schema_low_fitness() {
        let mut discovery = create_test_discovery();
        discovery.fitness_score = 0.5;
        
        let result = validate_discovery_schema(&discovery);
        assert!(result.is_err());
    }

    #[test]
    fn test_format_report() {
        let report = DiscoveryReport {
            total_candidates_evaluated: 100,
            discoveries_generated: 150,
            discoveries_validated: 100,
            average_fitness: 0.92,
            execution_time_ms: 1000,
        };
        
        let formatted = format_report(&report);
        assert!(formatted.contains("100"));
        assert!(formatted.contains("150"));
        assert!(formatted.contains("0.92"));
    }

    #[test]
    fn test_export_import_json() {
        let discovery = create_test_discovery();
        let discoveries = alloc::vec![discovery];
        
        let json = export_discoveries_json(&discoveries).unwrap();
        assert!(!json.is_empty());
        
        let imported = import_discoveries_json(&json).unwrap();
        assert_eq!(imported.len(), 1);
        assert_eq!(imported[0].id, "QRD-001");
    }
}
