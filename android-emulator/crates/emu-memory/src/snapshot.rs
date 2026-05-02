//! Memory snapshot support for record/replay and debugging.
//!
//! Provides copy-on-write snapshots for efficient state saving.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;

use crate::error::{MemoryError, MemoryResult};
use crate::page_table::PageEntry;
use crate::PAGE_SIZE;

/// Configuration for memory snapshots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotConfig {
    /// Whether to use copy-on-write optimization.
    pub use_cow: bool,
    /// Whether to compress snapshot data.
    pub compress: bool,
    /// Maximum number of snapshots to retain.
    pub max_snapshots: usize,
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self {
            use_cow: true,
            compress: false,
            max_snapshots: 10,
        }
    }
}

/// A memory snapshot capturing the state at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    /// Unique snapshot ID.
    pub id: u64,
    /// Human-readable name.
    pub name: String,
    /// Timestamp when the snapshot was created.
    pub timestamp: SystemTime,
    /// Page table entries at snapshot time.
    pub page_entries: HashMap<u64, PageEntry>,
    /// Dirty pages that have been copied (for CoW).
    pub copied_pages: HashMap<u64, Vec<u8>>,
    /// Original memory size.
    pub memory_size: usize,
}

impl MemorySnapshot {
    /// Create a new snapshot.
    pub fn new(id: u64, name: impl Into<String>, memory_size: usize) -> Self {
        Self {
            id,
            name: name.into(),
            timestamp: SystemTime::now(),
            page_entries: HashMap::new(),
            copied_pages: HashMap::new(),
            memory_size,
        }
    }

    /// Add a page entry to the snapshot.
    pub fn add_page(&mut self, vpn: u64, entry: PageEntry) {
        self.page_entries.insert(vpn, entry);
    }

    /// Copy a page's data into the snapshot.
    pub fn copy_page(&mut self, vpn: u64, data: &[u8]) {
        assert_eq!(data.len(), PAGE_SIZE);
        self.copied_pages.insert(vpn, data.to_vec());
    }

    /// Get the size of the snapshot in bytes.
    pub fn size_bytes(&self) -> usize {
        self.copied_pages.values().map(|v| v.len()).sum::<usize>()
            + self.page_entries.len() * std::mem::size_of::<(u64, PageEntry)>()
    }

    /// Get the number of pages in the snapshot.
    pub fn page_count(&self) -> usize {
        self.page_entries.len()
    }

    /// Get the number of copied (dirty) pages.
    pub fn copied_page_count(&self) -> usize {
        self.copied_pages.len()
    }
}

/// Manager for memory snapshots.
#[derive(Debug)]
pub struct SnapshotManager {
    /// Configuration.
    config: SnapshotConfig,
    /// Stored snapshots.
    snapshots: RwLock<Vec<Arc<MemorySnapshot>>>,
    /// Next snapshot ID.
    next_id: RwLock<u64>,
}

impl SnapshotManager {
    /// Create a new snapshot manager.
    pub fn new(config: SnapshotConfig) -> Self {
        Self {
            config,
            snapshots: RwLock::new(Vec::new()),
            next_id: RwLock::new(0),
        }
    }

    /// Create a new snapshot with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SnapshotConfig::default())
    }

    /// Store a snapshot.
    pub fn store(&self, mut snapshot: MemorySnapshot) -> MemoryResult<u64> {
        let id = {
            let mut next = self.next_id.write();
            let id = *next;
            *next += 1;
            id
        };
        
        snapshot.id = id;
        
        let mut snapshots = self.snapshots.write();
        
        // Enforce max snapshots limit
        while snapshots.len() >= self.config.max_snapshots {
            snapshots.remove(0);
        }
        
        snapshots.push(Arc::new(snapshot));
        
        Ok(id)
    }

    /// Get a snapshot by ID.
    pub fn get(&self, id: u64) -> Option<Arc<MemorySnapshot>> {
        let snapshots = self.snapshots.read();
        snapshots.iter().find(|s| s.id == id).cloned()
    }

    /// Get the latest snapshot.
    pub fn latest(&self) -> Option<Arc<MemorySnapshot>> {
        let snapshots = self.snapshots.read();
        snapshots.last().cloned()
    }

    /// Delete a snapshot by ID.
    pub fn delete(&self, id: u64) -> MemoryResult<bool> {
        let mut snapshots = self.snapshots.write();
        let len_before = snapshots.len();
        snapshots.retain(|s| s.id != id);
        Ok(snapshots.len() < len_before)
    }

    /// List all snapshots.
    pub fn list(&self) -> Vec<Arc<MemorySnapshot>> {
        self.snapshots.read().clone()
    }

    /// Get the total number of snapshots.
    pub fn count(&self) -> usize {
        self.snapshots.read().len()
    }

    /// Get the total size of all snapshots.
    pub fn total_size(&self) -> usize {
        self.snapshots
            .read()
            .iter()
            .map(|s| s.size_bytes())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page_table::PageFlags;

    #[test]
    fn test_snapshot_creation() {
        let mut snapshot = MemorySnapshot::new(0, "test", 1024 * 1024);
        snapshot.add_page(0, PageEntry::new(0, PageFlags::PRESENT | PageFlags::READ));
        snapshot.copy_page(1, &[0u8; PAGE_SIZE]);
        
        assert_eq!(snapshot.page_count(), 1);
        assert_eq!(snapshot.copied_page_count(), 1);
    }

    #[test]
    fn test_snapshot_manager() {
        let manager = SnapshotManager::new(SnapshotConfig {
            max_snapshots: 3,
            ..Default::default()
        });
        
        for i in 0..5 {
            let snapshot = MemorySnapshot::new(0, format!("snapshot_{}", i), 1024 * 1024);
            manager.store(snapshot).unwrap();
        }
        
        // Should only have 3 snapshots due to limit
        assert_eq!(manager.count(), 3);
        
        // Oldest snapshots should have been removed
        assert!(manager.get(0).is_none());
        assert!(manager.get(1).is_none());
        assert!(manager.get(2).is_some());
    }
}
