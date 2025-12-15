//! Mercurial adapter implementation (stub)

use crate::{
    object_store::ObjectId,
    vcs::VcsAdapter,
    QCoreError, Result,
};

/// Mercurial adapter
///
/// This is a stub implementation. Full Mercurial support will be added in Phase 2.
pub struct MercurialAdapter {
    current_branch: String,
    head: Option<ObjectId>,
}

impl MercurialAdapter {
    pub fn new() -> Self {
        MercurialAdapter {
            current_branch: "default".to_string(),
            head: None,
        }
    }
}

impl Default for MercurialAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl VcsAdapter for MercurialAdapter {
    fn name(&self) -> &str {
        "mercurial"
    }

    fn init(&mut self) -> Result<()> {
        self.head = None;
        self.current_branch = "default".to_string();
        Ok(())
    }

    fn clone(&mut self, _url: &str) -> Result<()> {
        Err(QCoreError::VcsAdapterError(
            "Mercurial clone not yet implemented".to_string(),
        ))
    }

    fn commit(&mut self, _message: &str) -> Result<ObjectId> {
        Err(QCoreError::VcsAdapterError(
            "Mercurial commit not yet implemented".to_string(),
        ))
    }

    fn get_head(&self) -> Result<ObjectId> {
        self.head
            .clone()
            .ok_or_else(|| QCoreError::VcsAdapterError("No HEAD commit".to_string()))
    }

    fn list_branches(&self) -> Result<Vec<String>> {
        Ok(vec![self.current_branch.clone()])
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
            "Mercurial merge not yet implemented".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mercurial_adapter_init() {
        let mut adapter = MercurialAdapter::new();
        adapter.init().unwrap();
        assert_eq!(adapter.name(), "mercurial");
        assert!(adapter.get_head().is_err());
    }

    #[test]
    fn test_mercurial_adapter_branches() {
        let adapter = MercurialAdapter::new();
        let branches = adapter.list_branches().unwrap();
        assert_eq!(branches.len(), 1);
        assert_eq!(branches[0], "default");
    }

    #[test]
    fn test_mercurial_adapter_checkout() {
        let mut adapter = MercurialAdapter::new();
        adapter.checkout("stable").unwrap();
        let branches = adapter.list_branches().unwrap();
        assert_eq!(branches[0], "stable");
    }

    #[test]
    fn test_mercurial_operations_not_implemented() {
        let mut adapter = MercurialAdapter::new();
        adapter.init().unwrap();

        assert!(adapter.commit("test").is_err());
        assert!(adapter.merge("branch").is_err());
        assert!(adapter.clone("http://example.com/repo").is_err());
    }
}
