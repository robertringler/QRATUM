//! PC compatibility error types.

use thiserror::Error;

/// Result type for PC compatibility operations.
pub type PcCompatResult<T> = Result<T, PcCompatError>;

/// Errors that can occur during PC compatibility operations.
#[derive(Debug, Error)]
pub enum PcCompatError {
    /// Unsupported syscall.
    #[error("unsupported syscall: {number}")]
    UnsupportedSyscall {
        /// Syscall number.
        number: u32,
    },

    /// Unsupported D3D feature.
    #[error("unsupported D3D feature: {feature}")]
    UnsupportedD3DFeature {
        /// Feature name.
        feature: String,
    },

    /// Shader compilation failed.
    #[error("shader compilation failed: {reason}")]
    ShaderCompilationFailed {
        /// Reason for the failure.
        reason: String,
    },

    /// Timing error.
    #[error("timing error: {reason}")]
    TimingError {
        /// Error reason.
        reason: String,
    },

    /// CPU error.
    #[error("CPU error: {0}")]
    Cpu(#[from] emu_cpu::CpuError),

    /// Memory error.
    #[error("memory error: {0}")]
    Memory(#[from] emu_memory::MemoryError),

    /// Graphics error.
    #[error("graphics error: {0}")]
    Graphics(#[from] emu_graphics::GraphicsError),
}
