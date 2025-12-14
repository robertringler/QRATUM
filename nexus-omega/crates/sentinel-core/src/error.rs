//! Error types for sentinel-core

use thiserror::Error;

/// SENTINEL errors
#[derive(Error, Debug)]
pub enum SentinelError {
    #[error("Anomaly detection error: {0}")]
    AnomalyDetection(String),

    #[error("Shadow identity error: {0}")]
    ShadowIdentity(String),

    #[error("Stratum error: {0}")]
    Stratum(String),

    #[error("Fusion error: {0}")]
    Fusion(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Cryptographic error: {0}")]
    Crypto(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type alias for sentinel-core
pub type Result<T> = std::result::Result<T, SentinelError>;
