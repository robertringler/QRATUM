//! QCore VCS - Universal Version Control System Engine
//!
//! A foundational Rust engine for multi-VCS (Git, Mercurial, SVN) with CRDT overlays,
//! cross-repo diff, fast object pool, and live state streaming.
//!
//! # Features
//!
//! - Git-compatible object store
//! - CRDT-based conflict-free collaboration
//! - Multi-VCS adapter interface
//! - High-performance object deduplication

pub mod crdt;
pub mod git;
pub mod mercurial;
pub mod object_store;
pub mod svn;
pub mod vcs;

pub use crate::vcs::VcsAdapter;

#[derive(Debug, thiserror::Error)]
pub enum QCoreError {
    #[error("Invalid object ID: {0}")]
    InvalidObjectId(String),

    #[error("Object not found: {0}")]
    ObjectNotFound(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("CRDT merge conflict: {0}")]
    CrdtMergeConflict(String),

    #[error("VCS adapter error: {0}")]
    VcsAdapterError(String),
}

pub type Result<T> = std::result::Result<T, QCoreError>;
