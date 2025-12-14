//! Q-Core: Rust-based Git Object Store
//!
//! This library provides a bare Git object store implementation with support for:
//! - Storing and retrieving Git objects (blobs, trees, commits) by SHA-1
//! - Managing refs (branches, tags, HEAD) in memory
//! - Pack/unpack operations
//! - In-memory repository loading

pub mod objects;
pub mod refs;
pub mod store;
pub mod error;
pub mod pack;

pub use objects::{GitObject, Blob, Tree, Commit, TreeEntry};
pub use refs::{RefManager, RefType};
pub use store::ObjectStore;
pub use error::{GitError, Result};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_exports() {
        // Verify that all public APIs are accessible
        let _store = ObjectStore::new();
        let _refs = RefManager::new();
    }
}
