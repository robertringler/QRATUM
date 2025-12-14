//! In-memory Git object store

use crate::error::{GitError, Result};
use crate::objects::{Blob, Commit, GitObject, Tree};
use crate::refs::RefManager;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// In-memory object store for Git objects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectStore {
    objects: HashMap<String, GitObject>,
    refs: RefManager,
}

impl ObjectStore {
    /// Create a new object store
    pub fn new() -> Self {
        Self {
            objects: HashMap::new(),
            refs: RefManager::new(),
        }
    }

    /// Store a Git object and return its SHA
    pub fn store_object(&mut self, object: GitObject) -> String {
        let sha = object.sha();
        self.objects.insert(sha.clone(), object);
        sha
    }

    /// Store a blob and return its SHA
    pub fn store_blob(&mut self, data: Vec<u8>) -> String {
        let blob = Blob::new(data);
        self.store_object(GitObject::Blob(blob))
    }

    /// Store a tree and return its SHA
    pub fn store_tree(&mut self, tree: Tree) -> String {
        self.store_object(GitObject::Tree(tree))
    }

    /// Store a commit and return its SHA
    pub fn store_commit(&mut self, commit: Commit) -> String {
        self.store_object(GitObject::Commit(commit))
    }

    /// Retrieve an object by SHA
    pub fn get_object(&self, sha: &str) -> Result<&GitObject> {
        self.objects
            .get(sha)
            .ok_or_else(|| GitError::ObjectNotFound(sha.to_string()))
    }

    /// Retrieve a blob by SHA
    pub fn get_blob(&self, sha: &str) -> Result<&Blob> {
        match self.get_object(sha)? {
            GitObject::Blob(blob) => Ok(blob),
            _ => Err(GitError::InvalidObjectType("Expected blob".to_string())),
        }
    }

    /// Retrieve a tree by SHA
    pub fn get_tree(&self, sha: &str) -> Result<&Tree> {
        match self.get_object(sha)? {
            GitObject::Tree(tree) => Ok(tree),
            _ => Err(GitError::InvalidObjectType("Expected tree".to_string())),
        }
    }

    /// Retrieve a commit by SHA
    pub fn get_commit(&self, sha: &str) -> Result<&Commit> {
        match self.get_object(sha)? {
            GitObject::Commit(commit) => Ok(commit),
            _ => Err(GitError::InvalidObjectType("Expected commit".to_string())),
        }
    }

    /// Check if an object exists
    pub fn has_object(&self, sha: &str) -> bool {
        self.objects.contains_key(sha)
    }

    /// List all object SHAs
    pub fn list_objects(&self) -> Vec<String> {
        self.objects.keys().cloned().collect()
    }

    /// Get the number of objects
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Get a mutable reference to the ref manager
    pub fn refs_mut(&mut self) -> &mut RefManager {
        &mut self.refs
    }

    /// Get a reference to the ref manager
    pub fn refs(&self) -> &RefManager {
        &self.refs
    }

    /// Export all objects as a map (for serialization/pack)
    pub fn export_objects(&self) -> &HashMap<String, GitObject> {
        &self.objects
    }

    /// Import objects from a map (for deserialization/unpack)
    pub fn import_objects(&mut self, objects: HashMap<String, GitObject>) {
        self.objects.extend(objects);
    }

    /// Clear all objects (useful for testing)
    pub fn clear(&mut self) {
        self.objects.clear();
        self.refs = RefManager::new();
    }
}

impl Default for ObjectStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::TreeEntry;

    #[test]
    fn test_store_and_retrieve_blob() {
        let mut store = ObjectStore::new();
        let data = b"Hello, World!".to_vec();
        let sha = store.store_blob(data.clone());

        let blob = store.get_blob(&sha).unwrap();
        assert_eq!(blob.data, data);
    }

    #[test]
    fn test_store_and_retrieve_tree() {
        let mut store = ObjectStore::new();
        let entries = vec![TreeEntry::new(
            "100644".to_string(),
            "file.txt".to_string(),
            "a".repeat(40),
        )];
        let tree = Tree::new(entries);
        let sha = store.store_tree(tree.clone());

        let retrieved = store.get_tree(&sha).unwrap();
        assert_eq!(retrieved.entries.len(), 1);
        assert_eq!(retrieved.entries[0].name, "file.txt");
    }

    #[test]
    fn test_store_and_retrieve_commit() {
        let mut store = ObjectStore::new();
        let commit = Commit::new(
            "a".repeat(40),
            vec![],
            "Author <author@example.com>".to_string(),
            "Committer <committer@example.com>".to_string(),
            "Initial commit".to_string(),
        );
        let sha = store.store_commit(commit.clone());

        let retrieved = store.get_commit(&sha).unwrap();
        assert_eq!(retrieved.message, "Initial commit");
    }

    #[test]
    fn test_has_object() {
        let mut store = ObjectStore::new();
        let sha = store.store_blob(b"test".to_vec());
        assert!(store.has_object(&sha));
        assert!(!store.has_object("invalid_sha"));
    }

    #[test]
    fn test_list_objects() {
        let mut store = ObjectStore::new();
        let sha1 = store.store_blob(b"test1".to_vec());
        let sha2 = store.store_blob(b"test2".to_vec());
        let objects = store.list_objects();
        assert_eq!(objects.len(), 2);
        assert!(objects.contains(&sha1));
        assert!(objects.contains(&sha2));
    }

    #[test]
    fn test_object_count() {
        let mut store = ObjectStore::new();
        assert_eq!(store.object_count(), 0);
        store.store_blob(b"test".to_vec());
        assert_eq!(store.object_count(), 1);
    }

    #[test]
    fn test_get_nonexistent_object() {
        let store = ObjectStore::new();
        let result = store.get_object("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_wrong_object_type() {
        let mut store = ObjectStore::new();
        let sha = store.store_blob(b"test".to_vec());
        let result = store.get_tree(&sha);
        assert!(result.is_err());
    }

    #[test]
    fn test_refs_integration() {
        let mut store = ObjectStore::new();
        let commit_sha = store.store_commit(Commit::new(
            "a".repeat(40),
            vec![],
            "Author".to_string(),
            "Committer".to_string(),
            "Message".to_string(),
        ));

        store
            .refs_mut()
            .create_branch("main".to_string(), commit_sha.clone())
            .unwrap();
        let branch = store.refs().get_branch("main").unwrap();
        assert_eq!(branch.sha, commit_sha);
    }

    #[test]
    fn test_clear() {
        let mut store = ObjectStore::new();
        store.store_blob(b"test".to_vec());
        assert_eq!(store.object_count(), 1);
        store.clear();
        assert_eq!(store.object_count(), 0);
    }

    #[test]
    fn test_export_import_objects() {
        let mut store1 = ObjectStore::new();
        let sha = store1.store_blob(b"test".to_vec());
        
        let exported = store1.export_objects().clone();
        
        let mut store2 = ObjectStore::new();
        store2.import_objects(exported);
        
        assert!(store2.has_object(&sha));
        assert_eq!(store2.object_count(), 1);
    }
}
