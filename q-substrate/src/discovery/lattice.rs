//! Discovery Lattice - Search Space Construction
//!
//! Implements the 5-dimensional discovery search space:
//! - Physics (quantum, materials, energy)
//! - Computation (algorithms, architectures)
//! - Materials (structures, properties)
//! - Systems (integration, deployment)
//! - Economics (value capture, scaling)

extern crate alloc;

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

/// Physics layer discovery nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhysicsNode {
    QuantumErrorSuppression,
    DecoherenceTopology,
    TopologicalQubit,
    AnyonBraiding,
    QuantumMemory,
    CoherentControl,
    AdiabaticEvolution,
    QuantumAnnealing,
}

/// Computation layer discovery nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputationNode {
    UltraLowBitInference,
    ReversibleKernel,
    NeuromorphicCircuit,
    SpikingNetwork,
    QuantumAlgorithm,
    HybridClassicalQuantum,
    DeterministicASI,
    StreamingInference,
}

/// Materials layer discovery nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterialsNode {
    MetastableLattice,
    PhononRoutingCrystal,
    SpinLiquidSubstrate,
    TopologicalInsulator,
    SuperconductingQubit,
    DiamondNV,
    SiliconPhotonics,
    GrapheneComposite,
}

/// Systems layer discovery nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SystemsNode {
    DeterministicASIRuntime,
    DistributedQRADLE,
    EdgeCloudHybrid,
    QuantumInterconnect,
    FaultTolerantArchitecture,
    PodIsolation,
    ProvenanceChain,
    RollbackSystem,
}

/// Economics layer discovery nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EconomicsNode {
    CapitalEfficiencyGradient,
    LearningCurveArbitrage,
    MoatTopology,
    NetworkEffect,
    ScalabilityVector,
    MarketPenetration,
    ValueCapture,
    CompetitiveAdvantage,
}

/// Candidate node combining multiple dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateNode {
    pub physics: Option<PhysicsNode>,
    pub computation: Option<ComputationNode>,
    pub materials: Option<MaterialsNode>,
    pub systems: Option<SystemsNode>,
    pub economics: Option<EconomicsNode>,
    pub interaction_id: String,
}

impl CandidateNode {
    /// Generate unique identifier for this node combination
    pub fn generate_id(&self) -> String {
        let mut parts = Vec::new();
        
        if let Some(p) = self.physics {
            parts.push(format!("P{:?}", p));
        }
        if let Some(c) = self.computation {
            parts.push(format!("C{:?}", c));
        }
        if let Some(m) = self.materials {
            parts.push(format!("M{:?}", m));
        }
        if let Some(s) = self.systems {
            parts.push(format!("S{:?}", s));
        }
        if let Some(e) = self.economics {
            parts.push(format!("E{:?}", e));
        }
        
        parts.join("-")
    }
}

/// Symbolic representation of a collapsed node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicRepresentation {
    pub node: CandidateNode,
    pub symbolic_form: String,
    pub dimensionality: usize,
}

/// Mutated node after constraint-breaking operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutatedNode {
    pub original: SymbolicRepresentation,
    pub mutation_type: String,
    pub mutated_form: String,
    pub novelty_score: f64,
}

/// Discovery search space lattice
pub struct DiscoveryLattice {
    physics_nodes: Vec<PhysicsNode>,
    computation_nodes: Vec<ComputationNode>,
    materials_nodes: Vec<MaterialsNode>,
    systems_nodes: Vec<SystemsNode>,
    economics_nodes: Vec<EconomicsNode>,
    candidate_count: usize,
    seed: u32,
}

impl DiscoveryLattice {
    /// Construct the full search space with deterministic seed
    pub fn new(seed: u32) -> Self {
        let physics_nodes = vec![
            PhysicsNode::QuantumErrorSuppression,
            PhysicsNode::DecoherenceTopology,
            PhysicsNode::TopologicalQubit,
            PhysicsNode::AnyonBraiding,
            PhysicsNode::QuantumMemory,
            PhysicsNode::CoherentControl,
            PhysicsNode::AdiabaticEvolution,
            PhysicsNode::QuantumAnnealing,
        ];
        
        let computation_nodes = vec![
            ComputationNode::UltraLowBitInference,
            ComputationNode::ReversibleKernel,
            ComputationNode::NeuromorphicCircuit,
            ComputationNode::SpikingNetwork,
            ComputationNode::QuantumAlgorithm,
            ComputationNode::HybridClassicalQuantum,
            ComputationNode::DeterministicASI,
            ComputationNode::StreamingInference,
        ];
        
        let materials_nodes = vec![
            MaterialsNode::MetastableLattice,
            MaterialsNode::PhononRoutingCrystal,
            MaterialsNode::SpinLiquidSubstrate,
            MaterialsNode::TopologicalInsulator,
            MaterialsNode::SuperconductingQubit,
            MaterialsNode::DiamondNV,
            MaterialsNode::SiliconPhotonics,
            MaterialsNode::GrapheneComposite,
        ];
        
        let systems_nodes = vec![
            SystemsNode::DeterministicASIRuntime,
            SystemsNode::DistributedQRADLE,
            SystemsNode::EdgeCloudHybrid,
            SystemsNode::QuantumInterconnect,
            SystemsNode::FaultTolerantArchitecture,
            SystemsNode::PodIsolation,
            SystemsNode::ProvenanceChain,
            SystemsNode::RollbackSystem,
        ];
        
        let economics_nodes = vec![
            EconomicsNode::CapitalEfficiencyGradient,
            EconomicsNode::LearningCurveArbitrage,
            EconomicsNode::MoatTopology,
            EconomicsNode::NetworkEffect,
            EconomicsNode::ScalabilityVector,
            EconomicsNode::MarketPenetration,
            EconomicsNode::ValueCapture,
            EconomicsNode::CompetitiveAdvantage,
        ];
        
        // Candidate count: combinations of 2-3 nodes from different dimensions
        let candidate_count = physics_nodes.len() * computation_nodes.len() 
            + physics_nodes.len() * materials_nodes.len()
            + computation_nodes.len() * systems_nodes.len()
            + systems_nodes.len() * economics_nodes.len()
            + materials_nodes.len() * computation_nodes.len();
        
        DiscoveryLattice {
            physics_nodes,
            computation_nodes,
            materials_nodes,
            systems_nodes,
            economics_nodes,
            candidate_count,
            seed,
        }
    }

    /// Enumerate all candidate interaction nodes
    pub fn enumerate_candidates(&self) -> Vec<CandidateNode> {
        let mut candidates = Vec::new();
        
        // Physics + Computation pairs
        for p in &self.physics_nodes {
            for c in &self.computation_nodes {
                let mut node = CandidateNode {
                    physics: Some(*p),
                    computation: Some(*c),
                    materials: None,
                    systems: None,
                    economics: None,
                    interaction_id: String::new(),
                };
                node.interaction_id = node.generate_id();
                candidates.push(node);
            }
        }
        
        // Physics + Materials pairs
        for p in &self.physics_nodes {
            for m in &self.materials_nodes {
                let mut node = CandidateNode {
                    physics: Some(*p),
                    computation: None,
                    materials: Some(*m),
                    systems: None,
                    economics: None,
                    interaction_id: String::new(),
                };
                node.interaction_id = node.generate_id();
                candidates.push(node);
            }
        }
        
        // Computation + Systems pairs
        for c in &self.computation_nodes {
            for s in &self.systems_nodes {
                let mut node = CandidateNode {
                    physics: None,
                    computation: Some(*c),
                    materials: None,
                    systems: Some(*s),
                    economics: None,
                    interaction_id: String::new(),
                };
                node.interaction_id = node.generate_id();
                candidates.push(node);
            }
        }
        
        // Systems + Economics pairs
        for s in &self.systems_nodes {
            for e in &self.economics_nodes {
                let mut node = CandidateNode {
                    physics: None,
                    computation: None,
                    materials: None,
                    systems: Some(*s),
                    economics: Some(*e),
                    interaction_id: String::new(),
                };
                node.interaction_id = node.generate_id();
                candidates.push(node);
            }
        }
        
        // Materials + Computation pairs
        for m in &self.materials_nodes {
            for c in &self.computation_nodes {
                let mut node = CandidateNode {
                    physics: None,
                    computation: Some(*c),
                    materials: Some(*m),
                    systems: None,
                    economics: None,
                    interaction_id: String::new(),
                };
                node.interaction_id = node.generate_id();
                candidates.push(node);
            }
        }
        
        candidates
    }

    /// Collapse node into symbolic representation
    pub fn collapse_node(&self, node: &CandidateNode) -> SymbolicRepresentation {
        let mut symbolic_parts = Vec::new();
        let mut dims = 0;
        
        if let Some(p) = node.physics {
            symbolic_parts.push(format!("Φ({:?})", p));
            dims += 1;
        }
        if let Some(c) = node.computation {
            symbolic_parts.push(format!("Ψ({:?})", c));
            dims += 1;
        }
        if let Some(m) = node.materials {
            symbolic_parts.push(format!("Μ({:?})", m));
            dims += 1;
        }
        if let Some(s) = node.systems {
            symbolic_parts.push(format!("Σ({:?})", s));
            dims += 1;
        }
        if let Some(e) = node.economics {
            symbolic_parts.push(format!("Ε({:?})", e));
            dims += 1;
        }
        
        SymbolicRepresentation {
            node: node.clone(),
            symbolic_form: symbolic_parts.join(" ⊗ "),
            dimensionality: dims,
        }
    }

    /// Get total candidate count
    pub fn get_candidate_count(&self) -> usize {
        self.candidate_count
    }

    /// Get seed
    pub fn get_seed(&self) -> u32 {
        self.seed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lattice_creation() {
        let lattice = DiscoveryLattice::new(42);
        assert_eq!(lattice.get_seed(), 42);
        assert!(lattice.get_candidate_count() > 0);
    }

    #[test]
    fn test_enumerate_candidates() {
        let lattice = DiscoveryLattice::new(42);
        let candidates = lattice.enumerate_candidates();
        
        assert!(!candidates.is_empty());
        assert_eq!(candidates.len(), lattice.get_candidate_count());
    }

    #[test]
    fn test_collapse_node() {
        let lattice = DiscoveryLattice::new(42);
        let node = CandidateNode {
            physics: Some(PhysicsNode::TopologicalQubit),
            computation: Some(ComputationNode::QuantumAlgorithm),
            materials: None,
            systems: None,
            economics: None,
            interaction_id: "test".into(),
        };
        
        let symbolic = lattice.collapse_node(&node);
        assert_eq!(symbolic.dimensionality, 2);
        assert!(symbolic.symbolic_form.contains("Φ"));
        assert!(symbolic.symbolic_form.contains("Ψ"));
    }

    #[test]
    fn test_candidate_id_generation() {
        let node = CandidateNode {
            physics: Some(PhysicsNode::QuantumErrorSuppression),
            computation: Some(ComputationNode::DeterministicASI),
            materials: None,
            systems: None,
            economics: None,
            interaction_id: String::new(),
        };
        
        let id = node.generate_id();
        assert!(id.contains("PQuantumErrorSuppression"));
        assert!(id.contains("CDeterministicASI"));
    }
}
