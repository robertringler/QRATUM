//! Configuration for SENTINEL

use serde::{Deserialize, Serialize};

/// SENTINEL configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentinelConfig {
    /// Operational mode
    pub mode: OperationalMode,
    /// Stratum A (Cognitive) configuration
    pub stratum_a: StratumAConfig,
    /// Stratum B (Cryptographic) configuration
    pub stratum_b: StratumBConfig,
    /// Stratum C (Perception) configuration
    pub stratum_c: StratumCConfig,
}

impl Default for SentinelConfig {
    fn default() -> Self {
        Self {
            mode: OperationalMode::Active,
            stratum_a: StratumAConfig::default(),
            stratum_b: StratumBConfig::default(),
            stratum_c: StratumCConfig::default(),
        }
    }
}

/// Operational mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationalMode {
    /// Active monitoring and intervention
    Active,
    /// Passive monitoring only
    Passive,
    /// Simulation mode
    Simulation,
}

/// Stratum A configuration (Cognitive layer)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratumAConfig {
    /// Prediction window in seconds
    pub prediction_window_s: u64,
    /// Evaluation threshold
    pub evaluation_threshold: f64,
    /// Intervention enabled
    pub intervention_enabled: bool,
}

impl Default for StratumAConfig {
    fn default() -> Self {
        Self {
            prediction_window_s: 60,
            evaluation_threshold: 0.8,
            intervention_enabled: true,
        }
    }
}

/// Stratum B configuration (Cryptographic layer)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratumBConfig {
    /// Onion routing layers
    pub onion_layers: usize,
    /// Signature verification enabled
    pub signature_verification: bool,
}

impl Default for StratumBConfig {
    fn default() -> Self {
        Self {
            onion_layers: 3,
            signature_verification: true,
        }
    }
}

/// Stratum C configuration (Perception layer)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratumCConfig {
    /// Sensor fusion enabled
    pub fusion_enabled: bool,
    /// Temporal coherence window in milliseconds
    pub coherence_window_ms: u64,
}

impl Default for StratumCConfig {
    fn default() -> Self {
        Self {
            fusion_enabled: true,
            coherence_window_ms: 1000,
        }
    }
}
