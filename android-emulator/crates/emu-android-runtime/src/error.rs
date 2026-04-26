//! Android runtime error types.

use thiserror::Error;

/// Result type for runtime operations.
pub type RuntimeResult<T> = Result<T, RuntimeError>;

/// Errors that can occur during Android runtime emulation.
#[derive(Debug, Error)]
pub enum RuntimeError {
    /// Failed to initialize the runtime.
    #[error("runtime initialization failed: {reason}")]
    InitializationFailed {
        /// Reason for the failure.
        reason: String,
    },

    /// Invalid DEX file format.
    #[error("invalid DEX file: {reason}")]
    InvalidDex {
        /// Reason for the invalid format.
        reason: String,
    },

    /// Class not found.
    #[error("class not found: {name}")]
    ClassNotFound {
        /// Name of the missing class.
        name: String,
    },

    /// Method not found.
    #[error("method not found: {class}.{method}")]
    MethodNotFound {
        /// Class name.
        class: String,
        /// Method name.
        method: String,
    },

    /// Field not found.
    #[error("field not found: {class}.{field}")]
    FieldNotFound {
        /// Class name.
        class: String,
        /// Field name.
        field: String,
    },

    /// Null pointer exception.
    #[error("null pointer exception")]
    NullPointerException,

    /// Array index out of bounds.
    #[error("array index out of bounds: index {index}, length {length}")]
    ArrayIndexOutOfBounds {
        /// The invalid index.
        index: i32,
        /// The array length.
        length: i32,
    },

    /// Out of memory error.
    #[error("out of memory: requested {requested} bytes")]
    OutOfMemory {
        /// Requested allocation size.
        requested: usize,
    },

    /// Binder transaction failed.
    #[error("binder transaction failed: {reason}")]
    BinderError {
        /// Reason for the failure.
        reason: String,
    },

    /// Syscall error.
    #[error("syscall error: {syscall} returned {errno}")]
    SyscallError {
        /// Syscall name.
        syscall: String,
        /// Error number.
        errno: i32,
    },

    /// CPU error.
    #[error("CPU error: {0}")]
    Cpu(#[from] emu_cpu::CpuError),

    /// Memory error.
    #[error("memory error: {0}")]
    Memory(#[from] emu_memory::MemoryError),
}
