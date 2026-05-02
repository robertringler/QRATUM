//! Filesystem error types.

use thiserror::Error;

/// Result type for filesystem operations.
pub type FsResult<T> = Result<T, FsError>;

/// Errors that can occur during filesystem operations.
#[derive(Debug, Error)]
pub enum FsError {
    /// File not found.
    #[error("file not found: {path}")]
    NotFound {
        /// Path that was not found.
        path: String,
    },

    /// Permission denied.
    #[error("permission denied: {path}")]
    PermissionDenied {
        /// Path with denied access.
        path: String,
    },

    /// File already exists.
    #[error("file already exists: {path}")]
    AlreadyExists {
        /// Path that already exists.
        path: String,
    },

    /// Not a directory.
    #[error("not a directory: {path}")]
    NotDirectory {
        /// Path that is not a directory.
        path: String,
    },

    /// Not a file.
    #[error("not a file: {path}")]
    NotFile {
        /// Path that is not a file.
        path: String,
    },

    /// Directory not empty.
    #[error("directory not empty: {path}")]
    DirectoryNotEmpty {
        /// Non-empty directory path.
        path: String,
    },

    /// Invalid path.
    #[error("invalid path: {reason}")]
    InvalidPath {
        /// Reason the path is invalid.
        reason: String,
    },

    /// Disk full.
    #[error("disk full")]
    DiskFull,

    /// Too many open files.
    #[error("too many open files")]
    TooManyOpenFiles,

    /// Invalid file handle.
    #[error("invalid file handle")]
    InvalidHandle,

    /// I/O error.
    #[error("I/O error: {reason}")]
    IoError {
        /// Error reason.
        reason: String,
    },

    /// Mount error.
    #[error("mount error: {reason}")]
    MountError {
        /// Error reason.
        reason: String,
    },

    /// Filesystem corruption.
    #[error("filesystem corruption detected: {reason}")]
    Corruption {
        /// Details about the corruption.
        reason: String,
    },
}
