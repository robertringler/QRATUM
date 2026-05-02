//! Graphics error types.

use thiserror::Error;

/// Result type for graphics operations.
pub type GraphicsResult<T> = Result<T, GraphicsError>;

/// Errors that can occur during graphics operations.
#[derive(Debug, Error)]
pub enum GraphicsError {
    /// Failed to initialize graphics.
    #[error("graphics initialization failed: {reason}")]
    InitializationFailed {
        /// Reason for the failure.
        reason: String,
    },

    /// Invalid framebuffer configuration.
    #[error("invalid framebuffer configuration: {reason}")]
    InvalidFramebufferConfig {
        /// Reason for the invalid configuration.
        reason: String,
    },

    /// Unsupported pixel format.
    #[error("unsupported pixel format: {format:?}")]
    UnsupportedPixelFormat {
        /// The unsupported format.
        format: crate::PixelFormat,
    },

    /// Shader compilation failed.
    #[error("shader compilation failed: {reason}")]
    ShaderCompilationFailed {
        /// Reason for the failure.
        reason: String,
    },

    /// Out of GPU memory.
    #[error("out of GPU memory")]
    OutOfMemory,

    /// Invalid GL operation.
    #[error("invalid GL operation: {reason}")]
    InvalidOperation {
        /// Reason for the invalid operation.
        reason: String,
    },

    /// Memory error.
    #[error("memory error: {0}")]
    Memory(#[from] emu_memory::MemoryError),
}
