//! Error types for Q-Core Git Object Store

use thiserror::Error;

/// Result type for Q-Core operations
pub type Result<T> = std::result::Result<T, GitError>;

/// Error types for Git object store operations
#[derive(Error, Debug)]
pub enum GitError {
    #[error("Object not found: {0}")]
    ObjectNotFound(String),

    #[error("Invalid object type: {0}")]
    InvalidObjectType(String),

    #[error("Invalid SHA-1 hash: {0}")]
    InvalidSha(String),

    #[error("Invalid ref name: {0}")]
    InvalidRefName(String),

    #[error("Ref not found: {0}")]
    RefNotFound(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Compression error: {0}")]
    CompressionError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("UTF-8 error: {0}")]
    Utf8Error(#[from] std::string::FromUtf8Error),

    #[error("Hex decode error: {0}")]
    HexError(#[from] hex::FromHexError),
}
