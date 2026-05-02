//! Discovery Type Definitions
//!
//! Core data structures for the Discovery Directive framework.
//! All discoveries follow the QRD-XXX format and must meet fitness threshold.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

/// A validated scientific/engineering discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Discovery {
    /// Discovery ID in format QRD-XXX
    pub id: String,
    /// Discovery title
    pub title: String,
    /// Formal falsifiable hypothesis
    pub hypothesis: String,
    /// Causal mechanism explanation
    pub core_mechanism: String,
    /// Mathematical/computational formulation
    pub formulation: Formulation,
    /// Validation methodology
    pub validation: ValidationPath,
    /// Industrial application potential
    pub industrial_impact: IndustrialImpact,
    /// Risk analysis and mitigation
    pub risk_envelope: RiskEnvelope,
    /// Fitness score (must be >= 0.87)
    pub fitness_score: f64,
    /// Provenance tracking
    pub provenance: Provenance,
}

/// Mathematical and computational formulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Formulation {
    /// Mathematical equations
    pub equations: Vec<String>,
    /// Pseudocode implementation
    pub pseudocode: Option<String>,
    /// Formal specification
    pub formal_spec: Option<String>,
}

/// Validation methodology and testing approach
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationPath {
    /// Validation method type
    pub method: ValidationMethod,
    /// Test rig description
    pub test_rig: String,
    /// Expected outcome
    pub expected_outcome: String,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
}

/// Validation method types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationMethod {
    /// Computational simulation
    Simulation,
    /// Analytical proof
    Analytic,
    /// Physical experiment
    Experimental,
    /// Combination of methods
    Hybrid,
}

/// Industrial impact assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndustrialImpact {
    /// Application domain
    pub application: String,
    /// Target market sector
    pub market_sector: String,
    /// Estimated economic value
    pub estimated_value: Option<String>,
}

/// Risk envelope for failure analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskEnvelope {
    /// Potential failure modes
    pub failure_modes: Vec<String>,
    /// Safety constraints
    pub safety_constraints: Vec<String>,
    /// Mitigation strategies
    pub mitigation_strategies: Vec<String>,
}

/// Provenance tracking for QRADLE integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    /// Generation timestamp (ISO 8601)
    pub generated_at: String,
    /// QRADLE provenance hash
    pub qradle_hash: String,
    /// Deterministic seed used
    pub seed: u32,
    /// Lattice node identifier
    pub lattice_node: String,
}

impl Discovery {
    /// Check if discovery meets fitness threshold
    pub fn is_valid(&self) -> bool {
        self.fitness_score >= 0.87 && self.fitness_score <= 1.0
    }

    /// Validate ID format (QRD-XXX)
    pub fn has_valid_id(&self) -> bool {
        if self.id.len() != 7 {
            return false;
        }
        if !self.id.starts_with("QRD-") {
            return false;
        }
        let num_part = &self.id[4..];
        num_part.chars().all(|c| c.is_ascii_digit())
    }
}

/// Discovery generation error types
#[derive(Debug, Clone)]
pub enum DiscoveryError {
    /// Failed to meet fitness threshold
    LowFitness(f64),
    /// Invalid lattice node
    InvalidNode(String),
    /// Provenance generation failed
    ProvenanceFailed(String),
    /// Serialization error
    SerializationError(String),
    /// Validation error
    ValidationError(String),
    /// Generic error
    Generic(String),
}

impl core::fmt::Display for DiscoveryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DiscoveryError::LowFitness(score) => {
                write!(f, "Fitness score {} below threshold 0.87", score)
            }
            DiscoveryError::InvalidNode(node) => {
                write!(f, "Invalid lattice node: {}", node)
            }
            DiscoveryError::ProvenanceFailed(msg) => {
                write!(f, "Provenance generation failed: {}", msg)
            }
            DiscoveryError::SerializationError(msg) => {
                write!(f, "Serialization error: {}", msg)
            }
            DiscoveryError::ValidationError(msg) => {
                write!(f, "Validation error: {}", msg)
            }
            DiscoveryError::Generic(msg) => write!(f, "{}", msg),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discovery_id_validation() {
        let mut discovery = Discovery {
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
                qradle_hash: "test_hash".into(),
                seed: 42,
                lattice_node: "test_node".into(),
            },
        };

        assert!(discovery.has_valid_id());
        assert!(discovery.is_valid());

        discovery.id = "INVALID".into();
        assert!(!discovery.has_valid_id());

        discovery.id = "QRD-001".into();
        discovery.fitness_score = 0.5;
        assert!(!discovery.is_valid());
    }

    #[test]
    fn test_fitness_threshold() {
        let discovery = Discovery {
            id: "QRD-001".into(),
            title: "Test".into(),
            hypothesis: "Test".into(),
            core_mechanism: "Test".into(),
            formulation: Formulation {
                equations: Vec::new(),
                pseudocode: None,
                formal_spec: None,
            },
            validation: ValidationPath {
                method: ValidationMethod::Analytic,
                test_rig: "Test".into(),
                expected_outcome: "Test".into(),
                confidence: 0.9,
            },
            industrial_impact: IndustrialImpact {
                application: "Test".into(),
                market_sector: "Test".into(),
                estimated_value: None,
            },
            risk_envelope: RiskEnvelope {
                failure_modes: Vec::new(),
                safety_constraints: Vec::new(),
                mitigation_strategies: Vec::new(),
            },
            fitness_score: 0.87,
            provenance: Provenance {
                generated_at: "2025-01-01T00:00:00Z".into(),
                qradle_hash: "test".into(),
                seed: 42,
                lattice_node: "test".into(),
            },
        };

        assert!(discovery.is_valid());
    }
}
