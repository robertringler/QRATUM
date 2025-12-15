//! Object store for Git-compatible objects (blobs, trees, commits, tags)

use crate::{QCoreError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Object ID - SHA-256 hash of the object content
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectId(String);

impl ObjectId {
    pub fn new(hash: String) -> Self {
        ObjectId(hash)
    }

    pub fn from_bytes(data: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        ObjectId(hex::encode(result))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ObjectId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Git object types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GitObject {
    Blob(Blob),
    Tree(Tree),
    Commit(Commit),
    Tag(Tag),
}

/// Blob object - stores file content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Blob {
    pub data: Vec<u8>,
}

impl Blob {
    pub fn new(data: Vec<u8>) -> Self {
        Blob { data }
    }

    pub fn compute_id(&self) -> ObjectId {
        let content = format!("blob {}\0", self.data.len());
        let mut full_data = content.into_bytes();
        full_data.extend_from_slice(&self.data);
        ObjectId::from_bytes(&full_data)
    }
}

/// Tree entry - represents a file or subdirectory in a tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeEntry {
    pub mode: String,
    pub name: String,
    pub object_id: ObjectId,
}

/// Tree object - stores directory structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tree {
    pub entries: Vec<TreeEntry>,
}

impl Tree {
    pub fn new(entries: Vec<TreeEntry>) -> Self {
        Tree { entries }
    }

    pub fn compute_id(&self) -> ObjectId {
        let mut content = String::from("tree ");
        // Use a simpler format for tree entries that doesn't require JSON serialization
        let mut entries_str = String::new();
        for entry in &self.entries {
            entries_str.push_str(&format!("{} {} {}\n", entry.mode, entry.name, entry.object_id));
        }
        content.push_str(&entries_str.len().to_string());
        content.push('\0');
        content.push_str(&entries_str);
        ObjectId::from_bytes(content.as_bytes())
    }
}

/// Commit object - represents a snapshot in history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Commit {
    pub tree: ObjectId,
    pub parents: Vec<ObjectId>,
    pub author: String,
    pub committer: String,
    pub message: String,
    pub timestamp: i64,
}

impl Commit {
    pub fn new(
        tree: ObjectId,
        parents: Vec<ObjectId>,
        author: String,
        committer: String,
        message: String,
        timestamp: i64,
    ) -> Self {
        Commit {
            tree,
            parents,
            author,
            committer,
            message,
            timestamp,
        }
    }

    pub fn compute_id(&self) -> ObjectId {
        // Create a deterministic commit format
        let mut content = String::new();
        content.push_str(&format!("tree {}\n", self.tree));
        for parent in &self.parents {
            content.push_str(&format!("parent {}\n", parent));
        }
        content.push_str(&format!("author {} {}\n", self.author, self.timestamp));
        content.push_str(&format!("committer {} {}\n", self.committer, self.timestamp));
        content.push('\n');
        content.push_str(&self.message);

        let full_content = format!("commit {}\0{}", content.len(), content);
        ObjectId::from_bytes(full_content.as_bytes())
    }
}

/// Tag object - lightweight or annotated tag
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tag {
    pub object: ObjectId,
    pub tag_type: String,
    pub tag_name: String,
    pub tagger: String,
    pub message: String,
}

/// In-memory object store with deduplication
pub struct ObjectStore {
    objects: HashMap<ObjectId, GitObject>,
}

impl ObjectStore {
    pub fn new() -> Self {
        ObjectStore {
            objects: HashMap::new(),
        }
    }

    /// Store an object and return its ID
    pub fn store(&mut self, object: GitObject) -> Result<ObjectId> {
        let id = match &object {
            GitObject::Blob(blob) => blob.compute_id(),
            GitObject::Tree(tree) => tree.compute_id(),
            GitObject::Commit(commit) => commit.compute_id(),
            GitObject::Tag(tag) => {
                // For tags, we'll use a simple hash of the serialized tag
                let tag_data = serde_json::to_string(tag)?;
                ObjectId::from_bytes(tag_data.as_bytes())
            }
        };

        self.objects.insert(id.clone(), object);
        Ok(id)
    }

    /// Retrieve an object by its ID
    pub fn get(&self, id: &ObjectId) -> Result<&GitObject> {
        self.objects
            .get(id)
            .ok_or_else(|| QCoreError::ObjectNotFound(id.to_string()))
    }

    /// Check if an object exists
    pub fn contains(&self, id: &ObjectId) -> bool {
        self.objects.contains_key(id)
    }

    /// Get the number of objects in the store
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
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

    #[test]
    fn test_blob_creation_and_storage() {
        let mut store = ObjectStore::new();
        let data = b"Hello, World!".to_vec();
        let blob = Blob::new(data.clone());
        let id = store.store(GitObject::Blob(blob)).unwrap();

        assert!(store.contains(&id));
        match store.get(&id).unwrap() {
            GitObject::Blob(retrieved_blob) => {
                assert_eq!(retrieved_blob.data, data);
            }
            _ => panic!("Expected Blob object"),
        }
    }

    #[test]
    fn test_object_deduplication() {
        let mut store = ObjectStore::new();
        let data = b"Test data".to_vec();
        let blob1 = Blob::new(data.clone());
        let blob2 = Blob::new(data);

        let id1 = store.store(GitObject::Blob(blob1)).unwrap();
        let id2 = store.store(GitObject::Blob(blob2)).unwrap();

        assert_eq!(id1, id2);
        assert_eq!(store.len(), 1); // Should deduplicate
    }

    #[test]
    fn test_tree_creation() {
        let mut store = ObjectStore::new();

        // Create a blob
        let blob = Blob::new(b"file content".to_vec());
        let blob_id = store.store(GitObject::Blob(blob)).unwrap();

        // Create a tree with the blob
        let entry = TreeEntry {
            mode: "100644".to_string(),
            name: "file.txt".to_string(),
            object_id: blob_id,
        };
        let tree = Tree::new(vec![entry]);
        let tree_id = store.store(GitObject::Tree(tree)).unwrap();

        assert!(store.contains(&tree_id));
    }

    #[test]
    fn test_commit_creation() {
        let mut store = ObjectStore::new();

        // Create a tree
        let tree = Tree::new(vec![]);
        let tree_id = store.store(GitObject::Tree(tree)).unwrap();

        // Create a commit
        let commit = Commit::new(
            tree_id,
            vec![],
            "Test Author <test@example.com>".to_string(),
            "Test Committer <test@example.com>".to_string(),
            "Initial commit".to_string(),
            chrono::Utc::now().timestamp(),
        );
        let commit_id = store.store(GitObject::Commit(commit)).unwrap();

        assert!(store.contains(&commit_id));
    }
}
