//! Page table management for virtual memory translation.
//!
//! This module implements a software page table structure compatible with
//! both ARMv8 and x86-64 memory models.

use bitflags::bitflags;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{MemoryError, MemoryResult};
use crate::{PAGE_SHIFT, PAGE_SIZE};

bitflags! {
    /// Page entry flags representing permissions and state.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PageFlags: u64 {
        /// Page is present/valid.
        const PRESENT = 1 << 0;
        /// Page is readable.
        const READ = 1 << 1;
        /// Page is writable.
        const WRITE = 1 << 2;
        /// Page is executable.
        const EXECUTE = 1 << 3;
        /// Page is user-accessible (vs kernel-only).
        const USER = 1 << 4;
        /// Page has been accessed.
        const ACCESSED = 1 << 5;
        /// Page has been modified (dirty).
        const DIRTY = 1 << 6;
        /// Page is copy-on-write.
        const COW = 1 << 7;
        /// Page is a huge page (2MB/1GB).
        const HUGE = 1 << 8;
        /// Page is for device/MMIO (non-cacheable).
        const DEVICE = 1 << 9;
        /// Page is shared between processes.
        const SHARED = 1 << 10;
    }
}

impl Default for PageFlags {
    fn default() -> Self {
        Self::PRESENT | Self::READ
    }
}

// Manual Serialize/Deserialize for bitflags
impl Serialize for PageFlags {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.bits().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PageFlags {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bits = u64::deserialize(deserializer)?;
        Ok(Self::from_bits_truncate(bits))
    }
}

/// A single page table entry.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PageEntry {
    /// Physical page frame number (host memory offset / PAGE_SIZE).
    pub pfn: u64,
    /// Page flags.
    pub flags: PageFlags,
}

impl PageEntry {
    /// Create a new page entry.
    pub fn new(pfn: u64, flags: PageFlags) -> Self {
        Self { pfn, flags }
    }

    /// Create an invalid/empty page entry.
    pub fn invalid() -> Self {
        Self {
            pfn: 0,
            flags: PageFlags::empty(),
        }
    }

    /// Check if the page is present.
    pub fn is_present(&self) -> bool {
        self.flags.contains(PageFlags::PRESENT)
    }

    /// Check if the page is writable.
    pub fn is_writable(&self) -> bool {
        self.flags.contains(PageFlags::WRITE)
    }

    /// Check if the page is executable.
    pub fn is_executable(&self) -> bool {
        self.flags.contains(PageFlags::EXECUTE)
    }

    /// Check if the page is copy-on-write.
    pub fn is_cow(&self) -> bool {
        self.flags.contains(PageFlags::COW)
    }

    /// Get the physical address for this page.
    pub fn physical_address(&self) -> u64 {
        self.pfn << PAGE_SHIFT
    }
}

/// Page table structure for virtual-to-physical address translation.
///
/// Uses a flat hash map for simplicity and flexibility, supporting
/// sparse address spaces efficiently.
#[derive(Debug)]
pub struct PageTable {
    /// Virtual page number → PageEntry mapping.
    entries: RwLock<HashMap<u64, PageEntry>>,
    /// Base address of guest physical memory in host memory.
    host_base: usize,
    /// Total size of guest physical memory.
    guest_memory_size: usize,
}

impl PageTable {
    /// Create a new empty page table.
    pub fn new(host_base: usize, guest_memory_size: usize) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            host_base,
            guest_memory_size,
        }
    }

    /// Map a virtual page to a physical page.
    pub fn map(&self, virtual_page: u64, entry: PageEntry) -> MemoryResult<()> {
        let mut entries = self.entries.write();
        entries.insert(virtual_page, entry);
        Ok(())
    }

    /// Unmap a virtual page.
    pub fn unmap(&self, virtual_page: u64) -> MemoryResult<Option<PageEntry>> {
        let mut entries = self.entries.write();
        Ok(entries.remove(&virtual_page))
    }

    /// Look up a virtual page.
    pub fn lookup(&self, virtual_page: u64) -> MemoryResult<Option<PageEntry>> {
        let entries = self.entries.read();
        Ok(entries.get(&virtual_page).copied())
    }

    /// Translate a virtual address to a physical address.
    pub fn translate(&self, virtual_addr: u64) -> MemoryResult<u64> {
        let virtual_page = virtual_addr >> PAGE_SHIFT;
        let offset = virtual_addr & ((PAGE_SIZE - 1) as u64);

        let entries = self.entries.read();
        match entries.get(&virtual_page) {
            Some(entry) if entry.is_present() => {
                Ok(entry.physical_address() + offset)
            }
            _ => Err(MemoryError::UnmappedAddress { addr: virtual_addr }),
        }
    }

    /// Translate a virtual address to a host pointer.
    ///
    /// # Safety
    ///
    /// The returned pointer is only valid as long as the memory region
    /// remains mapped and the page table is not modified.
    pub fn translate_to_host(&self, virtual_addr: u64) -> MemoryResult<*mut u8> {
        let physical_addr = self.translate(virtual_addr)?;
        if physical_addr as usize >= self.guest_memory_size {
            return Err(MemoryError::UnmappedAddress { addr: virtual_addr });
        }
        Ok((self.host_base + physical_addr as usize) as *mut u8)
    }

    /// Mark a page as dirty.
    pub fn mark_dirty(&self, virtual_page: u64) -> MemoryResult<()> {
        let mut entries = self.entries.write();
        if let Some(entry) = entries.get_mut(&virtual_page) {
            entry.flags |= PageFlags::DIRTY;
        }
        Ok(())
    }

    /// Mark a page as accessed.
    pub fn mark_accessed(&self, virtual_page: u64) -> MemoryResult<()> {
        let mut entries = self.entries.write();
        if let Some(entry) = entries.get_mut(&virtual_page) {
            entry.flags |= PageFlags::ACCESSED;
        }
        Ok(())
    }

    /// Get all mapped pages.
    pub fn all_pages(&self) -> Vec<(u64, PageEntry)> {
        let entries = self.entries.read();
        entries.iter().map(|(&k, &v)| (k, v)).collect()
    }

    /// Get the number of mapped pages.
    pub fn page_count(&self) -> usize {
        self.entries.read().len()
    }

    /// Clone the page table for copy-on-write.
    pub fn clone_for_cow(&self) -> Arc<Self> {
        let entries = self.entries.read();
        let mut new_entries = HashMap::new();
        
        for (&vpn, &entry) in entries.iter() {
            // Mark both original and new entries as COW
            let cow_entry = PageEntry {
                pfn: entry.pfn,
                flags: (entry.flags | PageFlags::COW) - PageFlags::WRITE,
            };
            new_entries.insert(vpn, cow_entry);
        }
        
        Arc::new(Self {
            entries: RwLock::new(new_entries),
            host_base: self.host_base,
            guest_memory_size: self.guest_memory_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_entry_creation() {
        let entry = PageEntry::new(42, PageFlags::PRESENT | PageFlags::READ | PageFlags::WRITE);
        assert!(entry.is_present());
        assert!(entry.is_writable());
        assert!(!entry.is_executable());
        assert_eq!(entry.physical_address(), 42 << PAGE_SHIFT);
    }

    #[test]
    fn test_page_table_basic_operations() {
        let pt = PageTable::new(0x1000_0000, 0x1000_0000);
        
        // Map a page
        let entry = PageEntry::new(100, PageFlags::PRESENT | PageFlags::READ);
        pt.map(0, entry).unwrap();
        
        // Look it up
        let found = pt.lookup(0).unwrap().unwrap();
        assert_eq!(found.pfn, 100);
        assert!(found.is_present());
        
        // Translate an address
        let phys = pt.translate(0x100).unwrap();
        assert_eq!(phys, (100 << PAGE_SHIFT) + 0x100);
    }

    #[test]
    fn test_unmapped_address() {
        let pt = PageTable::new(0x1000_0000, 0x1000_0000);
        let result = pt.translate(0x1000);
        assert!(matches!(result, Err(MemoryError::UnmappedAddress { .. })));
    }
}
