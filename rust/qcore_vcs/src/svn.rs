//! SVN (Subversion) adapter implementation (stub)

use crate::{
    object_store::ObjectId,
    vcs::VcsAdapter,
    QCoreError, Result,
};

/// SVN adapter
///
/// This is a stub implementation. Full SVN support will be added in Phase 2.
pub struct SvnAdapter {
    current_branch: String,
    head: Option<ObjectId>,
    revision: u64,
}

impl SvnAdapter {
    pub fn new() -> Self {
        SvnAdapter {
            current_branch: "trunk".to_string(),
            head: None,
            revision: 0,
        }
    }

    /// Get the current SVN revision number
    pub fn revision(&self) -> u64 {
        self.revision
    }
}

impl Default for SvnAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl VcsAdapter for SvnAdapter {
    fn name(&self) -> &str {
        "svn"
    }

    fn init(&mut self) -> Result<()> {
        self.head = None;
        self.current_branch = "trunk".to_string();
        self.revision = 0;
        Ok(())
    }

    fn clone(&mut self, _url: &str) -> Result<()> {
        Err(QCoreError::VcsAdapterError(
            "SVN checkout not yet implemented".to_string(),
        ))
    }

    fn commit(&mut self, _message: &str) -> Result<ObjectId> {
        Err(QCoreError::VcsAdapterError(
            "SVN commit not yet implemented".to_string(),
        ))
    }

    fn get_head(&self) -> Result<ObjectId> {
        self.head
            .clone()
            .ok_or_else(|| QCoreError::VcsAdapterError("No HEAD revision".to_string()))
    }

    fn list_branches(&self) -> Result<Vec<String>> {
        // SVN uses trunk/branches/tags structure
        Ok(vec!["trunk".to_string()])
    }

    fn create_branch(&mut self, name: &str) -> Result<()> {
        if name.is_empty() {
            return Err(QCoreError::VcsAdapterError(
                "Branch name cannot be empty".to_string(),
            ));
        }
        Ok(())
    }

    fn checkout(&mut self, target: &str) -> Result<()> {
        self.current_branch = target.to_string();
        Ok(())
    }

    fn merge(&mut self, _branch: &str) -> Result<ObjectId> {
        Err(QCoreError::VcsAdapterError(
            "SVN merge not yet implemented".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svn_adapter_init() {
        let mut adapter = SvnAdapter::new();
        adapter.init().unwrap();
        assert_eq!(adapter.name(), "svn");
        assert_eq!(adapter.revision(), 0);
        assert!(adapter.get_head().is_err());
    }

    #[test]
    fn test_svn_adapter_branches() {
        let adapter = SvnAdapter::new();
        let branches = adapter.list_branches().unwrap();
        assert_eq!(branches.len(), 1);
        assert_eq!(branches[0], "trunk");
    }

    #[test]
    fn test_svn_adapter_checkout() {
        let mut adapter = SvnAdapter::new();
        adapter.checkout("branches/release-1.0").unwrap();
        let branches = adapter.list_branches().unwrap();
        assert_eq!(branches[0], "trunk"); // list_branches always returns trunk
    }

    #[test]
    fn test_svn_operations_not_implemented() {
        let mut adapter = SvnAdapter::new();
        adapter.init().unwrap();

        assert!(adapter.commit("test").is_err());
        assert!(adapter.merge("branch").is_err());
        assert!(adapter.clone("http://example.com/repo").is_err());
    }
}
