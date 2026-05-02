//! Discovery Directive Module (QR-DISC-100-V1)
//!
//! Generates, validates, and archives 100 original discoveries across:
//! - Quantum computing
//! - Materials science  
//! - AI systems
//! - Cryptography
//! - Industrial design
//!
//! Using QRATUM ASI + QRADLE substrate for provenance tracking.

pub mod types;
pub mod lattice;
pub mod fitness;
pub mod engine;
pub mod provenance;
pub mod cli;

// Re-exports for convenience
pub use types::{
    Discovery, DiscoveryError, Formulation, IndustrialImpact, Provenance, RiskEnvelope,
    ValidationMethod, ValidationPath,
};

pub use lattice::{
    CandidateNode, ComputationNode, DiscoveryLattice, EconomicsNode, MaterialsNode,
    MutatedNode, PhysicsNode, SymbolicRepresentation, SystemsNode,
};

pub use fitness::{
    compute_feasibility, compute_fitness, compute_novelty, compute_scalability,
    compute_strategic_leverage, FitnessWeights, KnownArchitecture, MarketContext,
};

pub use engine::{DiscoveryEngine, DiscoveryReport};

pub use provenance::{
    generate_provenance_hash, generate_provenance_report, record_to_qradle,
    verify_provenance_chain, ProvenanceReport,
};

pub use cli::{
    export_discoveries_json, format_report, import_discoveries_json, run_discovery_directive,
    validate_discovery_schema,
};

/// Discovery module version
pub const DISCOVERY_VERSION: &str = "1.0.0";

/// Default fitness threshold for valid discoveries
pub const FITNESS_THRESHOLD: f64 = 0.87;

/// Default target count for discovery generation
pub const TARGET_DISCOVERY_COUNT: usize = 100;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all main types are accessible
        let _engine = DiscoveryEngine::new(42);
        let _lattice = DiscoveryLattice::new(42);
        let _weights = FitnessWeights::default();
    }

    #[test]
    fn test_constants() {
        assert_eq!(FITNESS_THRESHOLD, 0.87);
        assert_eq!(TARGET_DISCOVERY_COUNT, 100);
        assert_eq!(DISCOVERY_VERSION, "1.0.0");
    }
}
