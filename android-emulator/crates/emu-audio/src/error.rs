//! Audio error types.

use thiserror::Error;

/// Result type for audio operations.
pub type AudioResult<T> = Result<T, AudioError>;

/// Errors that can occur during audio operations.
#[derive(Debug, Error)]
pub enum AudioError {
    /// Failed to initialize audio.
    #[error("audio initialization failed: {reason}")]
    InitializationFailed {
        /// Reason for the failure.
        reason: String,
    },

    /// Invalid audio format.
    #[error("invalid audio format: {reason}")]
    InvalidFormat {
        /// Reason for the invalid format.
        reason: String,
    },

    /// Buffer underrun.
    #[error("audio buffer underrun")]
    BufferUnderrun,

    /// Buffer overrun.
    #[error("audio buffer overrun")]
    BufferOverrun,

    /// Stream not running.
    #[error("audio stream not running")]
    StreamNotRunning,

    /// Device error.
    #[error("audio device error: {reason}")]
    DeviceError {
        /// Reason for the error.
        reason: String,
    },
}
