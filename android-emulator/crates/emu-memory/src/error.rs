//! Memory subsystem error types.

use thiserror::Error;

/// Result type for memory operations.
pub type MemoryResult<T> = Result<T, MemoryError>;

/// Errors that can occur during memory operations.
#[derive(Debug, Error)]
pub enum MemoryError {
    /// Attempted to access an unmapped address.
    #[error("unmapped address: {addr:#x}")]
    UnmappedAddress {
        /// The address that was accessed.
        addr: u64,
    },

    /// Attempted to write to a read-only region.
    #[error("write to read-only region at {addr:#x}")]
    ReadOnlyViolation {
        /// The address of the write attempt.
        addr: u64,
    },

    /// Attempted to execute from a non-executable region.
    #[error("execute from non-executable region at {addr:#x}")]
    ExecuteViolation {
        /// The address of the execute attempt.
        addr: u64,
    },

    /// Address alignment error.
    #[error("misaligned address: {addr:#x} (required alignment: {alignment})")]
    MisalignedAddress {
        /// The misaligned address.
        addr: u64,
        /// The required alignment.
        alignment: usize,
    },

    /// Memory region overlap error.
    #[error("region overlap: {start:#x}-{end:#x} overlaps with existing region")]
    RegionOverlap {
        /// Start of the overlapping region.
        start: u64,
        /// End of the overlapping region.
        end: u64,
    },

    /// Out of memory.
    #[error("out of memory: requested {requested} bytes, available {available} bytes")]
    OutOfMemory {
        /// Requested allocation size.
        requested: usize,
        /// Available memory.
        available: usize,
    },

    /// Snapshot creation failed.
    #[error("snapshot creation failed: {reason}")]
    SnapshotFailed {
        /// Reason for the failure.
        reason: String,
    },

    /// Snapshot restore failed.
    #[error("snapshot restore failed: {reason}")]
    RestoreFailed {
        /// Reason for the failure.
        reason: String,
    },

    /// Invalid page table entry.
    #[error("invalid page table entry at {addr:#x}")]
    InvalidPageEntry {
        /// Address of the invalid entry.
        addr: u64,
    },

    /// Host memory mapping failed.
    #[error("host memory mapping failed: {0}")]
    HostMappingFailed(#[from] std::io::Error),
}
