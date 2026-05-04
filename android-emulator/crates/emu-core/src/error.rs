//! Emulator error types.

use thiserror::Error;

/// Result type for emulator operations.
pub type EmulatorResult<T> = Result<T, EmulatorError>;

/// Errors that can occur during emulator operations.
#[derive(Debug, Error)]
pub enum EmulatorError {
    /// Failed to initialize the emulator.
    #[error("initialization failed: {reason}")]
    InitializationFailed {
        /// Reason for the failure.
        reason: String,
    },

    /// Configuration error.
    #[error("configuration error: {reason}")]
    ConfigurationError {
        /// Reason for the error.
        reason: String,
    },

    /// Boot failure.
    #[error("boot failed: {reason}")]
    BootFailed {
        /// Reason for the failure.
        reason: String,
    },

    /// CPU error.
    #[error("CPU error: {0}")]
    Cpu(#[from] emu_cpu::CpuError),

    /// Memory error.
    #[error("memory error: {0}")]
    Memory(#[from] emu_memory::MemoryError),

    /// Android runtime error.
    #[error("runtime error: {0}")]
    Runtime(#[from] emu_android_runtime::RuntimeError),

    /// Graphics error.
    #[error("graphics error: {0}")]
    Graphics(#[from] emu_graphics::GraphicsError),

    /// Audio error.
    #[error("audio error: {0}")]
    Audio(#[from] emu_audio::AudioError),

    /// Filesystem error.
    #[error("filesystem error: {0}")]
    Filesystem(#[from] emu_filesystem::FsError),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
