//! Git adapter implementation

use crate::{
    object_store::{Commit, ObjectId, ObjectStore},
    vcs::VcsAdapter,
    QCoreError, Result,
};

/// Git adapter using our object store
pub struct GitAdapter {
    store: ObjectStore,
    head: Option<ObjectId>,
    current_branch: String,
}

impl GitAdapter {
    pub fn new() -> Self {
        GitAdapter {
            store: ObjectStore::new(),
            head: None,
            current_branch: "main".to_string(),
        }
    }

    pub fn store(&self) -> &ObjectStore {
        &self.store
    }

    pub fn store_mut(&mut self) -> &mut ObjectStore {
        &mut self.store
    }
}

impl Default for GitAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl VcsAdapter for GitAdapter {
    fn name(&self) -> &str {
        "git"
    }

    fn init(&mut self) -> Result<()> {
        // Initialize an empty repository
        self.head = None;
        self.current_branch = "main".to_string();
        Ok(())
    }

    fn clone(&mut self, _url: &str) -> Result<()> {
        // Stub: In a real implementation, this would clone from a remote
        Err(QCoreError::VcsAdapterError(
            "Clone not yet implemented".to_string(),
        ))
    }

    fn commit(&mut self, message: &str) -> Result<ObjectId> {
        // Create a simple commit with a placeholder tree
        // In a full implementation, this would reference an actual tree object
        const EMPTY_TREE_PLACEHOLDER: &str = 
            "0000000000000000000000000000000000000000000000000000000000000000";
        let tree_id = ObjectId::new(EMPTY_TREE_PLACEHOLDER.to_string());

        let parents = if let Some(head) = &self.head {
            vec![head.clone()]
        } else {
            vec![]
        };

        let commit = Commit::new(
            tree_id,
            parents,
            "QuASIM <quasim@example.com>".to_string(),
            "QuASIM <quasim@example.com>".to_string(),
            message.to_string(),
            chrono::Utc::now().timestamp(),
        );

        let commit_id = commit.compute_id();
        self.store
            .store(crate::object_store::GitObject::Commit(commit))?;
        self.head = Some(commit_id.clone());

        Ok(commit_id)
    }

    fn get_head(&self) -> Result<ObjectId> {
        self.head
            .clone()
            .ok_or_else(|| QCoreError::VcsAdapterError("No HEAD commit".to_string()))
    }

    fn list_branches(&self) -> Result<Vec<String>> {
        // Stub: In a real implementation, this would list all branches
        Ok(vec![self.current_branch.clone()])
    }

    fn create_branch(&mut self, name: &str) -> Result<()> {
        // Stub: In a real implementation, this would create a new branch
        if name.is_empty() {
            return Err(QCoreError::VcsAdapterError(
                "Branch name cannot be empty".to_string(),
            ));
        }
        Ok(())
    }

    fn checkout(&mut self, target: &str) -> Result<()> {
        // Stub: In a real implementation, this would checkout a branch or commit
        self.current_branch = target.to_string();
        Ok(())
    }

    fn merge(&mut self, _branch: &str) -> Result<ObjectId> {
        // Stub: In a real implementation, this would merge branches
        Err(QCoreError::VcsAdapterError(
            "Merge not yet implemented".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_git_adapter_init() {
        let mut adapter = GitAdapter::new();
        adapter.init().unwrap();
        assert_eq!(adapter.name(), "git");
        assert!(adapter.get_head().is_err());
    }

    #[test]
    fn test_git_adapter_commit() {
        let mut adapter = GitAdapter::new();
        adapter.init().unwrap();

        let commit_id = adapter.commit("Initial commit").unwrap();
        assert_eq!(adapter.get_head().unwrap(), commit_id);
    }

    #[test]
    fn test_git_adapter_branches() {
        let mut adapter = GitAdapter::new();
        adapter.init().unwrap();

        let branches = adapter.list_branches().unwrap();
        assert_eq!(branches.len(), 1);
        assert_eq!(branches[0], "main");
    }

    #[test]
    fn test_git_adapter_checkout() {
        let mut adapter = GitAdapter::new();
        adapter.init().unwrap();

        adapter.checkout("develop").unwrap();
        let branches = adapter.list_branches().unwrap();
        assert_eq!(branches[0], "develop");
    }
}
