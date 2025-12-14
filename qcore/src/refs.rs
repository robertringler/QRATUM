//! Ref management (branches, tags, HEAD)

use crate::error::{GitError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Type of Git reference
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RefType {
    Branch,
    Tag,
    Head,
}

/// A Git reference (branch, tag, or HEAD)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ref {
    pub name: String,
    pub sha: String,
    pub ref_type: RefType,
}

impl Ref {
    /// Create a new reference
    pub fn new(name: String, sha: String, ref_type: RefType) -> Self {
        Self {
            name,
            sha,
            ref_type,
        }
    }
}

/// In-memory ref manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefManager {
    refs: HashMap<String, Ref>,
    head: Option<String>,
}

impl RefManager {
    /// Create a new ref manager
    pub fn new() -> Self {
        Self {
            refs: HashMap::new(),
            head: None,
        }
    }

    /// Create a branch
    pub fn create_branch(&mut self, name: String, sha: String) -> Result<()> {
        let full_name = format!("refs/heads/{}", name);
        if self.refs.contains_key(&full_name) {
            return Err(GitError::InvalidRefName(format!(
                "Branch already exists: {}",
                name
            )));
        }
        let ref_obj = Ref::new(full_name.clone(), sha, RefType::Branch);
        self.refs.insert(full_name, ref_obj);
        Ok(())
    }

    /// Update a branch
    pub fn update_branch(&mut self, name: String, sha: String) -> Result<()> {
        let full_name = format!("refs/heads/{}", name);
        if !self.refs.contains_key(&full_name) {
            return Err(GitError::RefNotFound(name));
        }
        let ref_obj = Ref::new(full_name.clone(), sha, RefType::Branch);
        self.refs.insert(full_name, ref_obj);
        Ok(())
    }

    /// Get a branch
    pub fn get_branch(&self, name: &str) -> Result<&Ref> {
        let full_name = format!("refs/heads/{}", name);
        self.refs
            .get(&full_name)
            .ok_or_else(|| GitError::RefNotFound(name.to_string()))
    }

    /// Delete a branch
    pub fn delete_branch(&mut self, name: &str) -> Result<()> {
        let full_name = format!("refs/heads/{}", name);
        self.refs
            .remove(&full_name)
            .ok_or_else(|| GitError::RefNotFound(name.to_string()))?;
        Ok(())
    }

    /// List all branches
    pub fn list_branches(&self) -> Vec<&Ref> {
        self.refs
            .values()
            .filter(|r| r.ref_type == RefType::Branch)
            .collect()
    }

    /// Create a tag
    pub fn create_tag(&mut self, name: String, sha: String) -> Result<()> {
        let full_name = format!("refs/tags/{}", name);
        if self.refs.contains_key(&full_name) {
            return Err(GitError::InvalidRefName(format!(
                "Tag already exists: {}",
                name
            )));
        }
        let ref_obj = Ref::new(full_name.clone(), sha, RefType::Tag);
        self.refs.insert(full_name, ref_obj);
        Ok(())
    }

    /// Get a tag
    pub fn get_tag(&self, name: &str) -> Result<&Ref> {
        let full_name = format!("refs/tags/{}", name);
        self.refs
            .get(&full_name)
            .ok_or_else(|| GitError::RefNotFound(name.to_string()))
    }

    /// Delete a tag
    pub fn delete_tag(&mut self, name: &str) -> Result<()> {
        let full_name = format!("refs/tags/{}", name);
        self.refs
            .remove(&full_name)
            .ok_or_else(|| GitError::RefNotFound(name.to_string()))?;
        Ok(())
    }

    /// List all tags
    pub fn list_tags(&self) -> Vec<&Ref> {
        self.refs
            .values()
            .filter(|r| r.ref_type == RefType::Tag)
            .collect()
    }

    /// Set HEAD to point to a ref
    pub fn set_head(&mut self, ref_name: String) -> Result<()> {
        if !self.refs.contains_key(&ref_name) {
            return Err(GitError::RefNotFound(ref_name));
        }
        self.head = Some(ref_name);
        Ok(())
    }

    /// Get the current HEAD
    pub fn get_head(&self) -> Option<&String> {
        self.head.as_ref()
    }

    /// Get the SHA that HEAD points to
    pub fn get_head_sha(&self) -> Result<String> {
        let head_ref = self
            .head
            .as_ref()
            .ok_or_else(|| GitError::RefNotFound("HEAD".to_string()))?;
        let ref_obj = self
            .refs
            .get(head_ref)
            .ok_or_else(|| GitError::RefNotFound(head_ref.clone()))?;
        Ok(ref_obj.sha.clone())
    }

    /// List all refs
    pub fn list_all_refs(&self) -> Vec<&Ref> {
        self.refs.values().collect()
    }

    /// Get a ref by full name
    pub fn get_ref(&self, name: &str) -> Result<&Ref> {
        self.refs
            .get(name)
            .ok_or_else(|| GitError::RefNotFound(name.to_string()))
    }
}

impl Default for RefManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_branch() {
        let mut manager = RefManager::new();
        let sha = "a".repeat(40);
        manager.create_branch("main".to_string(), sha.clone()).unwrap();
        let branch = manager.get_branch("main").unwrap();
        assert_eq!(branch.sha, sha);
        assert_eq!(branch.ref_type, RefType::Branch);
    }

    #[test]
    fn test_update_branch() {
        let mut manager = RefManager::new();
        let sha1 = "a".repeat(40);
        let sha2 = "b".repeat(40);
        manager.create_branch("main".to_string(), sha1).unwrap();
        manager.update_branch("main".to_string(), sha2.clone()).unwrap();
        let branch = manager.get_branch("main").unwrap();
        assert_eq!(branch.sha, sha2);
    }

    #[test]
    fn test_delete_branch() {
        let mut manager = RefManager::new();
        let sha = "a".repeat(40);
        manager.create_branch("temp".to_string(), sha).unwrap();
        assert!(manager.get_branch("temp").is_ok());
        manager.delete_branch("temp").unwrap();
        assert!(manager.get_branch("temp").is_err());
    }

    #[test]
    fn test_list_branches() {
        let mut manager = RefManager::new();
        manager.create_branch("main".to_string(), "a".repeat(40)).unwrap();
        manager.create_branch("dev".to_string(), "b".repeat(40)).unwrap();
        let branches = manager.list_branches();
        assert_eq!(branches.len(), 2);
    }

    #[test]
    fn test_create_tag() {
        let mut manager = RefManager::new();
        let sha = "a".repeat(40);
        manager.create_tag("v1.0.0".to_string(), sha.clone()).unwrap();
        let tag = manager.get_tag("v1.0.0").unwrap();
        assert_eq!(tag.sha, sha);
        assert_eq!(tag.ref_type, RefType::Tag);
    }

    #[test]
    fn test_delete_tag() {
        let mut manager = RefManager::new();
        let sha = "a".repeat(40);
        manager.create_tag("v1.0.0".to_string(), sha).unwrap();
        assert!(manager.get_tag("v1.0.0").is_ok());
        manager.delete_tag("v1.0.0").unwrap();
        assert!(manager.get_tag("v1.0.0").is_err());
    }

    #[test]
    fn test_list_tags() {
        let mut manager = RefManager::new();
        manager.create_tag("v1.0.0".to_string(), "a".repeat(40)).unwrap();
        manager.create_tag("v1.1.0".to_string(), "b".repeat(40)).unwrap();
        let tags = manager.list_tags();
        assert_eq!(tags.len(), 2);
    }

    #[test]
    fn test_set_head() {
        let mut manager = RefManager::new();
        let sha = "a".repeat(40);
        manager.create_branch("main".to_string(), sha.clone()).unwrap();
        manager.set_head("refs/heads/main".to_string()).unwrap();
        assert_eq!(manager.get_head(), Some(&"refs/heads/main".to_string()));
        assert_eq!(manager.get_head_sha().unwrap(), sha);
    }

    #[test]
    fn test_duplicate_branch_error() {
        let mut manager = RefManager::new();
        manager.create_branch("main".to_string(), "a".repeat(40)).unwrap();
        let result = manager.create_branch("main".to_string(), "b".repeat(40));
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_tag_error() {
        let mut manager = RefManager::new();
        manager.create_tag("v1.0.0".to_string(), "a".repeat(40)).unwrap();
        let result = manager.create_tag("v1.0.0".to_string(), "b".repeat(40));
        assert!(result.is_err());
    }
}
