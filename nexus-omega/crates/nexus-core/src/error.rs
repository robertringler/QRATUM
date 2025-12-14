//! Error types for nexus-core

use thiserror::Error;

/// Core runtime errors
#[derive(Error, Debug)]
pub enum CoreError {
    #[error("Scheduler error: {0}")]
    Scheduler(String),

    #[error("Executor error: {0}")]
    Executor(String),

    #[error("Time error: {0}")]
    Time(String),

    #[error("Determinism violation: {0}")]
    DeterminismViolation(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type alias for nexus-core
pub type Result<T> = std::result::Result<T, CoreError>;
