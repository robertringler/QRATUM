//! Git object types (Blob, Tree, Commit)

use crate::error::{GitError, Result};
use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};
use std::fmt;

/// Git object types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GitObject {
    Blob(Blob),
    Tree(Tree),
    Commit(Commit),
}

impl GitObject {
    /// Calculate the SHA-1 hash of this object
    pub fn sha(&self) -> String {
        match self {
            GitObject::Blob(blob) => blob.sha(),
            GitObject::Tree(tree) => tree.sha(),
            GitObject::Commit(commit) => commit.sha(),
        }
    }

    /// Get the object type as a string
    pub fn object_type(&self) -> &str {
        match self {
            GitObject::Blob(_) => "blob",
            GitObject::Tree(_) => "tree",
            GitObject::Commit(_) => "commit",
        }
    }

    /// Serialize the object to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            GitObject::Blob(blob) => blob.to_bytes(),
            GitObject::Tree(tree) => tree.to_bytes(),
            GitObject::Commit(commit) => commit.to_bytes(),
        }
    }

    /// Parse a Git object from bytes
    pub fn from_bytes(obj_type: &str, data: &[u8]) -> Result<Self> {
        match obj_type {
            "blob" => Ok(GitObject::Blob(Blob::from_bytes(data)?)),
            "tree" => Ok(GitObject::Tree(Tree::from_bytes(data)?)),
            "commit" => Ok(GitObject::Commit(Commit::from_bytes(data)?)),
            _ => Err(GitError::InvalidObjectType(obj_type.to_string())),
        }
    }
}

/// A Git blob object (file content)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Blob {
    pub data: Vec<u8>,
}

impl Blob {
    /// Create a new blob
    pub fn new(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Calculate the SHA-1 hash of this blob
    pub fn sha(&self) -> String {
        let header = format!("blob {}\0", self.data.len());
        let mut hasher = Sha1::new();
        hasher.update(header.as_bytes());
        hasher.update(&self.data);
        hex::encode(hasher.finalize())
    }

    /// Serialize the blob to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        self.data.clone()
    }

    /// Parse a blob from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        Ok(Self {
            data: data.to_vec(),
        })
    }
}

/// A Git tree entry (file or subdirectory)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreeEntry {
    pub mode: String,
    pub name: String,
    pub sha: String,
}

impl TreeEntry {
    /// Create a new tree entry
    pub fn new(mode: String, name: String, sha: String) -> Self {
        Self { mode, name, sha }
    }
}

impl fmt::Display for TreeEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.mode, self.name, self.sha)
    }
}

/// A Git tree object (directory listing)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tree {
    pub entries: Vec<TreeEntry>,
}

impl Tree {
    /// Create a new tree
    pub fn new(entries: Vec<TreeEntry>) -> Self {
        Self { entries }
    }

    /// Calculate the SHA-1 hash of this tree
    pub fn sha(&self) -> String {
        let content = self.to_bytes();
        let header = format!("tree {}\0", content.len());
        let mut hasher = Sha1::new();
        hasher.update(header.as_bytes());
        hasher.update(&content);
        hex::encode(hasher.finalize())
    }

    /// Serialize the tree to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = Vec::new();
        for entry in &self.entries {
            result.extend(entry.mode.as_bytes());
            result.push(b' ');
            result.extend(entry.name.as_bytes());
            result.push(b'\0');
            // SHA stored as binary (20 bytes)
            if let Ok(sha_bytes) = hex::decode(&entry.sha) {
                result.extend(&sha_bytes);
            }
        }
        result
    }

    /// Parse a tree from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut entries = Vec::new();
        let mut pos = 0;

        while pos < data.len() {
            // Read mode
            let space_pos = data[pos..]
                .iter()
                .position(|&b| b == b' ')
                .ok_or_else(|| GitError::ParseError("Invalid tree format".to_string()))?;
            let mode = String::from_utf8(data[pos..pos + space_pos].to_vec())?;
            pos += space_pos + 1;

            // Read name
            let null_pos = data[pos..]
                .iter()
                .position(|&b| b == b'\0')
                .ok_or_else(|| GitError::ParseError("Invalid tree format".to_string()))?;
            let name = String::from_utf8(data[pos..pos + null_pos].to_vec())?;
            pos += null_pos + 1;

            // Read SHA (20 bytes)
            if pos + 20 > data.len() {
                return Err(GitError::ParseError("Invalid tree format".to_string()));
            }
            let sha = hex::encode(&data[pos..pos + 20]);
            pos += 20;

            entries.push(TreeEntry { mode, name, sha });
        }

        Ok(Self { entries })
    }
}

/// A Git commit object
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Commit {
    pub tree: String,
    pub parents: Vec<String>,
    pub author: String,
    pub committer: String,
    pub message: String,
}

impl Commit {
    /// Create a new commit
    pub fn new(
        tree: String,
        parents: Vec<String>,
        author: String,
        committer: String,
        message: String,
    ) -> Self {
        Self {
            tree,
            parents,
            author,
            committer,
            message,
        }
    }

    /// Calculate the SHA-1 hash of this commit
    pub fn sha(&self) -> String {
        let content = self.to_bytes();
        let header = format!("commit {}\0", content.len());
        let mut hasher = Sha1::new();
        hasher.update(header.as_bytes());
        hasher.update(&content);
        hex::encode(hasher.finalize())
    }

    /// Serialize the commit to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = String::new();
        result.push_str(&format!("tree {}\n", self.tree));
        for parent in &self.parents {
            result.push_str(&format!("parent {}\n", parent));
        }
        result.push_str(&format!("author {}\n", self.author));
        result.push_str(&format!("committer {}\n", self.committer));
        result.push_str(&format!("\n{}", self.message));
        result.into_bytes()
    }

    /// Parse a commit from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let content = String::from_utf8(data.to_vec())?;
        let lines = content.lines();

        let mut tree = String::new();
        let mut parents = Vec::new();
        let mut author = String::new();
        let mut committer = String::new();
        let mut message_lines = Vec::new();
        let mut in_message = false;

        for line in lines {
            if in_message {
                message_lines.push(line);
            } else if line.is_empty() {
                in_message = true;
            } else if let Some(rest) = line.strip_prefix("tree ") {
                tree = rest.to_string();
            } else if let Some(rest) = line.strip_prefix("parent ") {
                parents.push(rest.to_string());
            } else if let Some(rest) = line.strip_prefix("author ") {
                author = rest.to_string();
            } else if let Some(rest) = line.strip_prefix("committer ") {
                committer = rest.to_string();
            }
        }

        let message = message_lines.join("\n");

        Ok(Self {
            tree,
            parents,
            author,
            committer,
            message,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blob_creation() {
        let data = b"Hello, World!".to_vec();
        let blob = Blob::new(data.clone());
        assert_eq!(blob.data, data);
    }

    #[test]
    fn test_blob_sha() {
        let blob = Blob::new(b"Hello, World!".to_vec());
        let sha = blob.sha();
        // Pre-calculated SHA-1 for "Hello, World!"
        assert_eq!(sha, "8ab686eafeb1f44702738c8b0f24f2567c36da6d");
    }

    #[test]
    fn test_blob_serialization() {
        let data = b"test data".to_vec();
        let blob = Blob::new(data.clone());
        let bytes = blob.to_bytes();
        let parsed = Blob::from_bytes(&bytes).unwrap();
        assert_eq!(blob, parsed);
    }

    #[test]
    fn test_tree_entry_creation() {
        let entry = TreeEntry::new(
            "100644".to_string(),
            "test.txt".to_string(),
            "abc123".to_string(),
        );
        assert_eq!(entry.mode, "100644");
        assert_eq!(entry.name, "test.txt");
        assert_eq!(entry.sha, "abc123");
    }

    #[test]
    fn test_tree_creation() {
        let entries = vec![
            TreeEntry::new(
                "100644".to_string(),
                "file1.txt".to_string(),
                "a".repeat(40),
            ),
            TreeEntry::new(
                "100644".to_string(),
                "file2.txt".to_string(),
                "b".repeat(40),
            ),
        ];
        let tree = Tree::new(entries.clone());
        assert_eq!(tree.entries.len(), 2);
        assert_eq!(tree.entries, entries);
    }

    #[test]
    fn test_commit_creation() {
        let commit = Commit::new(
            "a".repeat(40),
            vec!["b".repeat(40)],
            "Author <author@example.com>".to_string(),
            "Committer <committer@example.com>".to_string(),
            "Initial commit".to_string(),
        );
        assert_eq!(commit.tree.len(), 40);
        assert_eq!(commit.parents.len(), 1);
        assert_eq!(commit.message, "Initial commit");
    }

    #[test]
    fn test_commit_serialization() {
        let commit = Commit::new(
            "a".repeat(40),
            vec!["b".repeat(40)],
            "Author <author@example.com>".to_string(),
            "Committer <committer@example.com>".to_string(),
            "Test message".to_string(),
        );
        let bytes = commit.to_bytes();
        let parsed = Commit::from_bytes(&bytes).unwrap();
        assert_eq!(commit, parsed);
    }

    #[test]
    fn test_git_object_type() {
        let blob = GitObject::Blob(Blob::new(vec![1, 2, 3]));
        assert_eq!(blob.object_type(), "blob");

        let tree = GitObject::Tree(Tree::new(vec![]));
        assert_eq!(tree.object_type(), "tree");

        let commit = GitObject::Commit(Commit::new(
            "a".repeat(40),
            vec![],
            "author".to_string(),
            "committer".to_string(),
            "msg".to_string(),
        ));
        assert_eq!(commit.object_type(), "commit");
    }
}
