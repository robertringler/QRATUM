//! Input error types.

use thiserror::Error;

/// Result type for input operations.
pub type InputResult<T> = Result<T, InputError>;

/// Errors that can occur during input operations.
#[derive(Debug, Error)]
pub enum InputError {
    /// Invalid input device.
    #[error("invalid input device: {reason}")]
    InvalidDevice {
        /// Reason for the error.
        reason: String,
    },

    /// Input queue full.
    #[error("input queue full")]
    QueueFull,

    /// Device not found.
    #[error("device not found: {name}")]
    DeviceNotFound {
        /// Device name.
        name: String,
    },
}
