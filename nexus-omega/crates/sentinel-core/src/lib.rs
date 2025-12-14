//! SENTINEL Zero Layer Meta-Substrate
//!
//! Multi-stratum cognitive defense system with shadow identity management.

pub mod anomaly;
pub mod config;
pub mod error;
pub mod fusion;
pub mod interface;
pub mod shadow;
pub mod stratum_a;
pub mod stratum_b;
pub mod stratum_c;

pub use anomaly::{Anomaly, AnomalyClass, AnomalyDetector, ThreatLevel};
pub use config::{OperationalMode, SentinelConfig};
pub use error::{Result, SentinelError};
pub use interface::SentinelInterface;
pub use shadow::{ObfuscationLevel, ShadowIdentity, ShadowManager};
