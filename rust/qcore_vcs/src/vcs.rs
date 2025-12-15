//! VCS adapter interface for multi-VCS support

use crate::{object_store::ObjectId, Result};

/// Universal VCS adapter trait
///
/// Provides a common interface for different VCS backends (Git, Mercurial, SVN)
pub trait VcsAdapter {
    /// Get the name of the VCS system
    fn name(&self) -> &str;

    /// Initialize a new repository
    fn init(&mut self) -> Result<()>;

    /// Clone a repository from a URL
    fn clone(&mut self, url: &str) -> Result<()>;

    /// Commit changes with a message
    fn commit(&mut self, message: &str) -> Result<ObjectId>;

    /// Get the current HEAD commit
    fn get_head(&self) -> Result<ObjectId>;

    /// List all branches
    fn list_branches(&self) -> Result<Vec<String>>;

    /// Create a new branch
    fn create_branch(&mut self, name: &str) -> Result<()>;

    /// Checkout a branch or commit
    fn checkout(&mut self, target: &str) -> Result<()>;

    /// Merge a branch into current branch
    fn merge(&mut self, branch: &str) -> Result<ObjectId>;
}

/// Configuration for VCS repository
#[derive(Debug, Clone)]
pub struct VcsConfig {
    pub repo_path: String,
    pub vcs_type: VcsType,
}

/// Supported VCS types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VcsType {
    Git,
    Mercurial,
    Svn,
}

impl std::str::FromStr for VcsType {
    type Err = crate::QCoreError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "git" => Ok(VcsType::Git),
            "hg" | "mercurial" => Ok(VcsType::Mercurial),
            "svn" | "subversion" => Ok(VcsType::Svn),
            _ => Err(crate::QCoreError::VcsAdapterError(format!(
                "Unknown VCS type: {}",
                s
            ))),
        }
    }
}

impl VcsType {
    pub fn as_str(&self) -> &str {
        match self {
            VcsType::Git => "git",
            VcsType::Mercurial => "mercurial",
            VcsType::Svn => "svn",
        }
    }
}
