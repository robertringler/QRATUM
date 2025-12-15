//! CRDT (Conflict-free Replicated Data Type) implementation for timeline

use crate::{object_store::ObjectId, QCoreError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// CRDT timeline for conflict-free collaboration
///
/// Uses a DAG (Directed Acyclic Graph) to track operations and their dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrdtTimeline {
    /// Operations indexed by their ID
    operations: HashMap<OperationId, Operation>,
    /// Dependencies between operations (operation -> its dependencies)
    dependencies: HashMap<OperationId, HashSet<OperationId>>,
    /// Current heads (operations with no dependents)
    heads: HashSet<OperationId>,
}

/// Unique operation ID
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OperationId(String);

impl OperationId {
    pub fn new(id: String) -> Self {
        OperationId(id)
    }

    pub fn from_commit(commit_id: &ObjectId) -> Self {
        OperationId(commit_id.as_str().to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for OperationId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Operation in the CRDT timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    pub id: OperationId,
    pub commit: ObjectId,
    pub author: String,
    pub timestamp: i64,
    pub parents: Vec<OperationId>,
}

impl Operation {
    pub fn new(
        commit: ObjectId,
        author: String,
        timestamp: i64,
        parents: Vec<OperationId>,
    ) -> Self {
        let id = OperationId::from_commit(&commit);
        Operation {
            id,
            commit,
            author,
            timestamp,
            parents,
        }
    }
}

impl CrdtTimeline {
    pub fn new() -> Self {
        CrdtTimeline {
            operations: HashMap::new(),
            dependencies: HashMap::new(),
            heads: HashSet::new(),
        }
    }

    /// Add a new operation to the timeline
    pub fn add_operation(&mut self, operation: Operation) -> Result<()> {
        let op_id = operation.id.clone();

        // Validate that parent operations exist
        for parent in &operation.parents {
            if !self.operations.contains_key(parent) {
                return Err(QCoreError::CrdtMergeConflict(format!(
                    "Parent operation {} not found",
                    parent
                )));
            }
            // Remove parent from heads since it now has a dependent
            self.heads.remove(parent);
        }

        // Store dependencies
        let deps: HashSet<OperationId> = operation.parents.iter().cloned().collect();
        self.dependencies.insert(op_id.clone(), deps);

        // Add to heads (will be removed if another operation depends on this)
        self.heads.insert(op_id.clone());

        // Store the operation
        self.operations.insert(op_id, operation);

        Ok(())
    }

    /// Get an operation by its ID
    pub fn get_operation(&self, id: &OperationId) -> Option<&Operation> {
        self.operations.get(id)
    }

    /// Get all current heads (operations with no dependents)
    pub fn get_heads(&self) -> Vec<&Operation> {
        self.heads
            .iter()
            .filter_map(|id| self.operations.get(id))
            .collect()
    }

    /// Check if an operation exists
    pub fn contains(&self, id: &OperationId) -> bool {
        self.operations.contains_key(id)
    }

    /// Get the number of operations
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Check if the timeline is empty
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Merge two timelines (conflict-free)
    pub fn merge(&mut self, other: &CrdtTimeline) -> Result<()> {
        // Collect operations in topological order to handle dependencies correctly
        let mut to_add = Vec::new();
        for (id, operation) in &other.operations {
            if !self.operations.contains_key(id) {
                to_add.push(operation.clone());
            }
        }

        // Sort by timestamp to add in chronological order
        to_add.sort_by_key(|op| op.timestamp);

        // Add operations, skipping those whose dependencies aren't met yet
        let mut added = true;
        while added && !to_add.is_empty() {
            added = false;
            to_add.retain(|op| {
                // Check if all dependencies exist
                let deps_met = op.parents.iter().all(|parent| self.operations.contains_key(parent));
                if deps_met {
                    // Safe to unwrap since we know the operation is valid
                    if self.add_operation(op.clone()).is_ok() {
                        added = true;
                        return false; // Remove from to_add
                    }
                }
                true // Keep in to_add
            });
        }

        // If there are operations left, their dependencies weren't in either timeline
        if !to_add.is_empty() {
            return Err(QCoreError::CrdtMergeConflict(format!(
                "Cannot merge: {} operation(s) have unmet dependencies",
                to_add.len()
            )));
        }

        Ok(())
    }

    /// Get operations in topological order
    pub fn topological_sort(&self) -> Result<Vec<&Operation>> {
        let mut sorted = Vec::new();
        let mut visited = HashSet::new();
        let mut temp_mark = HashSet::new();

        // Helper function for DFS
        fn visit<'a>(
            id: &OperationId,
            operations: &'a HashMap<OperationId, Operation>,
            dependencies: &HashMap<OperationId, HashSet<OperationId>>,
            visited: &mut HashSet<OperationId>,
            temp_mark: &mut HashSet<OperationId>,
            sorted: &mut Vec<&'a Operation>,
        ) -> Result<()> {
            if visited.contains(id) {
                return Ok(());
            }
            if temp_mark.contains(id) {
                return Err(QCoreError::CrdtMergeConflict("Cycle detected".to_string()));
            }

            temp_mark.insert(id.clone());

            if let Some(deps) = dependencies.get(id) {
                for dep in deps {
                    visit(dep, operations, dependencies, visited, temp_mark, sorted)?;
                }
            }

            temp_mark.remove(id);
            visited.insert(id.clone());
            if let Some(op) = operations.get(id) {
                sorted.push(op);
            }

            Ok(())
        }

        // Visit all operations
        for id in self.operations.keys() {
            if !visited.contains(id) {
                visit(
                    id,
                    &self.operations,
                    &self.dependencies,
                    &mut visited,
                    &mut temp_mark,
                    &mut sorted,
                )?;
            }
        }

        Ok(sorted)
    }
}

impl Default for CrdtTimeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crdt_timeline_creation() {
        let timeline = CrdtTimeline::new();
        assert!(timeline.is_empty());
        assert_eq!(timeline.len(), 0);
    }

    #[test]
    fn test_add_operation() {
        let mut timeline = CrdtTimeline::new();

        let commit_id = ObjectId::new("abc123".to_string());
        let op = Operation::new(
            commit_id,
            "test@example.com".to_string(),
            chrono::Utc::now().timestamp(),
            vec![],
        );

        timeline.add_operation(op.clone()).unwrap();

        assert_eq!(timeline.len(), 1);
        assert!(timeline.contains(&op.id));
    }

    #[test]
    fn test_add_dependent_operation() {
        let mut timeline = CrdtTimeline::new();

        // Add first operation
        let commit1 = ObjectId::new("commit1".to_string());
        let op1 = Operation::new(
            commit1,
            "test@example.com".to_string(),
            chrono::Utc::now().timestamp(),
            vec![],
        );
        let op1_id = op1.id.clone();
        timeline.add_operation(op1).unwrap();

        // Add second operation that depends on first
        let commit2 = ObjectId::new("commit2".to_string());
        let op2 = Operation::new(
            commit2,
            "test@example.com".to_string(),
            chrono::Utc::now().timestamp(),
            vec![op1_id.clone()],
        );
        timeline.add_operation(op2.clone()).unwrap();

        assert_eq!(timeline.len(), 2);

        // Check that op2 is now the only head
        let heads = timeline.get_heads();
        assert_eq!(heads.len(), 1);
        assert_eq!(heads[0].id, op2.id);
    }

    #[test]
    fn test_merge_timelines() {
        let mut timeline1 = CrdtTimeline::new();
        let mut timeline2 = CrdtTimeline::new();

        // Add operation to timeline1
        let commit1 = ObjectId::new("commit1".to_string());
        let op1 = Operation::new(
            commit1,
            "user1@example.com".to_string(),
            chrono::Utc::now().timestamp(),
            vec![],
        );
        timeline1.add_operation(op1.clone()).unwrap();

        // Add operation to timeline2
        let commit2 = ObjectId::new("commit2".to_string());
        let op2 = Operation::new(
            commit2,
            "user2@example.com".to_string(),
            chrono::Utc::now().timestamp(),
            vec![],
        );
        timeline2.add_operation(op2).unwrap();

        // Merge timeline2 into timeline1
        timeline1.merge(&timeline2).unwrap();

        assert_eq!(timeline1.len(), 2);
    }

    #[test]
    fn test_topological_sort() {
        let mut timeline = CrdtTimeline::new();

        // Create a simple linear history: op1 -> op2 -> op3
        let commit1 = ObjectId::new("commit1".to_string());
        let op1 = Operation::new(
            commit1,
            "test@example.com".to_string(),
            1,
            vec![],
        );
        let op1_id = op1.id.clone();
        timeline.add_operation(op1).unwrap();

        let commit2 = ObjectId::new("commit2".to_string());
        let op2 = Operation::new(
            commit2,
            "test@example.com".to_string(),
            2,
            vec![op1_id.clone()],
        );
        let op2_id = op2.id.clone();
        timeline.add_operation(op2).unwrap();

        let commit3 = ObjectId::new("commit3".to_string());
        let op3 = Operation::new(
            commit3,
            "test@example.com".to_string(),
            3,
            vec![op2_id.clone()],
        );
        timeline.add_operation(op3).unwrap();

        let sorted = timeline.topological_sort().unwrap();
        assert_eq!(sorted.len(), 3);

        // Verify order: op1 should come before op2, op2 before op3
        let timestamps: Vec<i64> = sorted.iter().map(|op| op.timestamp).collect();
        assert_eq!(timestamps, vec![1, 2, 3]);
    }
}
