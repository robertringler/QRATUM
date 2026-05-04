//! Memory Management Unit (MMU) emulation.
//!
//! Provides the main interface for virtual memory operations.

use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::error::{MemoryError, MemoryResult};
use crate::page_table::{PageEntry, PageFlags, PageTable};
use crate::region::{MemoryRegion, RegionPermissions};
use crate::{PAGE_SHIFT, PAGE_SIZE};

/// Memory Management Unit.
///
/// Handles virtual-to-physical address translation and memory access control.
#[derive(Debug)]
pub struct Mmu {
    /// Page table for address translation.
    page_table: Arc<PageTable>,
    /// Memory regions sorted by start address.
    regions: RwLock<BTreeMap<u64, MemoryRegion>>,
    /// Host memory backing buffer.
    host_memory: Vec<u8>,
    /// Total guest memory size.
    guest_memory_size: usize,
    /// Next free physical page.
    next_free_page: RwLock<u64>,
}

impl Mmu {
    /// Create a new MMU with the specified guest memory size.
    pub fn new(guest_memory_size: usize) -> Self {
        let host_memory = vec![0u8; guest_memory_size];
        let host_base = host_memory.as_ptr() as usize;
        
        Self {
            page_table: Arc::new(PageTable::new(host_base, guest_memory_size)),
            regions: RwLock::new(BTreeMap::new()),
            host_memory,
            guest_memory_size,
            next_free_page: RwLock::new(0),
        }
    }

    /// Allocate a physical page.
    fn alloc_physical_page(&self) -> MemoryResult<u64> {
        let mut next = self.next_free_page.write();
        let pfn = *next;
        let phys_addr = (pfn as usize) << PAGE_SHIFT;
        
        if phys_addr + PAGE_SIZE > self.guest_memory_size {
            return Err(MemoryError::OutOfMemory {
                requested: PAGE_SIZE,
                available: self.guest_memory_size - phys_addr,
            });
        }
        
        *next += 1;
        Ok(pfn)
    }

    /// Map a memory region into the guest address space.
    pub fn map_region(&self, region: MemoryRegion) -> MemoryResult<()> {
        // Check for overlaps
        {
            let regions = self.regions.read();
            for existing in regions.values() {
                if region.overlaps(existing) {
                    return Err(MemoryError::RegionOverlap {
                        start: region.start,
                        end: region.end(),
                    });
                }
            }
        }

        // Calculate page flags from permissions
        let mut flags = PageFlags::PRESENT;
        if region.permissions.contains(RegionPermissions::READ) {
            flags |= PageFlags::READ;
        }
        if region.permissions.contains(RegionPermissions::WRITE) {
            flags |= PageFlags::WRITE;
        }
        if region.permissions.contains(RegionPermissions::EXECUTE) {
            flags |= PageFlags::EXECUTE;
        }

        // Map all pages in the region
        let num_pages = (region.size as usize + PAGE_SIZE - 1) / PAGE_SIZE;
        for i in 0..num_pages {
            let virtual_page = (region.start >> PAGE_SHIFT) + i as u64;
            let pfn = self.alloc_physical_page()?;
            let entry = PageEntry::new(pfn, flags);
            self.page_table.map(virtual_page, entry)?;
        }

        // Store the region
        self.regions.write().insert(region.start, region);

        Ok(())
    }

    /// Unmap a memory region.
    pub fn unmap_region(&self, start: u64) -> MemoryResult<Option<MemoryRegion>> {
        let region = self.regions.write().remove(&start);
        
        if let Some(ref r) = region {
            let num_pages = (r.size as usize + PAGE_SIZE - 1) / PAGE_SIZE;
            for i in 0..num_pages {
                let virtual_page = (r.start >> PAGE_SHIFT) + i as u64;
                self.page_table.unmap(virtual_page)?;
            }
        }
        
        Ok(region)
    }

    /// Read bytes from guest memory.
    pub fn read(&self, addr: u64, buf: &mut [u8]) -> MemoryResult<()> {
        self.check_read_permission(addr, buf.len())?;
        
        for (i, byte) in buf.iter_mut().enumerate() {
            let vaddr = addr + i as u64;
            let paddr = self.page_table.translate(vaddr)?;
            *byte = self.host_memory[paddr as usize];
        }
        
        Ok(())
    }

    /// Write bytes to guest memory.
    pub fn write(&self, addr: u64, data: &[u8]) -> MemoryResult<()> {
        self.check_write_permission(addr, data.len())?;
        
        // Safety: We have exclusive access through the borrow checker
        let host_memory = self.host_memory.as_ptr() as *mut u8;
        
        for (i, &byte) in data.iter().enumerate() {
            let vaddr = addr + i as u64;
            let paddr = self.page_table.translate(vaddr)?;
            // Safety: paddr is within bounds (checked by translate)
            unsafe {
                *host_memory.add(paddr as usize) = byte;
            }
            // Mark page as dirty
            let vpn = vaddr >> PAGE_SHIFT;
            self.page_table.mark_dirty(vpn)?;
        }
        
        Ok(())
    }

    /// Read a u8 from guest memory.
    pub fn read_u8(&self, addr: u64) -> MemoryResult<u8> {
        let mut buf = [0u8; 1];
        self.read(addr, &mut buf)?;
        Ok(buf[0])
    }

    /// Read a u16 from guest memory (little-endian).
    pub fn read_u16(&self, addr: u64) -> MemoryResult<u16> {
        let mut buf = [0u8; 2];
        self.read(addr, &mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    /// Read a u32 from guest memory (little-endian).
    pub fn read_u32(&self, addr: u64) -> MemoryResult<u32> {
        let mut buf = [0u8; 4];
        self.read(addr, &mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    /// Read a u64 from guest memory (little-endian).
    pub fn read_u64(&self, addr: u64) -> MemoryResult<u64> {
        let mut buf = [0u8; 8];
        self.read(addr, &mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    /// Write a u8 to guest memory.
    pub fn write_u8(&self, addr: u64, value: u8) -> MemoryResult<()> {
        self.write(addr, &[value])
    }

    /// Write a u16 to guest memory (little-endian).
    pub fn write_u16(&self, addr: u64, value: u16) -> MemoryResult<()> {
        self.write(addr, &value.to_le_bytes())
    }

    /// Write a u32 to guest memory (little-endian).
    pub fn write_u32(&self, addr: u64, value: u32) -> MemoryResult<()> {
        self.write(addr, &value.to_le_bytes())
    }

    /// Write a u64 to guest memory (little-endian).
    pub fn write_u64(&self, addr: u64, value: u64) -> MemoryResult<()> {
        self.write(addr, &value.to_le_bytes())
    }

    /// Get the region containing an address.
    pub fn get_region(&self, addr: u64) -> Option<MemoryRegion> {
        let regions = self.regions.read();
        for region in regions.values() {
            if region.contains(addr) {
                return Some(region.clone());
            }
        }
        None
    }

    /// Check read permission for an address range.
    fn check_read_permission(&self, addr: u64, len: usize) -> MemoryResult<()> {
        let end = addr.saturating_add(len as u64);
        let regions = self.regions.read();
        
        for region in regions.values() {
            if region.contains(addr) || region.contains(end.saturating_sub(1)) {
                if !region.is_readable() {
                    return Err(MemoryError::UnmappedAddress { addr });
                }
            }
        }
        
        Ok(())
    }

    /// Check write permission for an address range.
    fn check_write_permission(&self, addr: u64, len: usize) -> MemoryResult<()> {
        let end = addr.saturating_add(len as u64);
        let regions = self.regions.read();
        
        for region in regions.values() {
            if region.contains(addr) || region.contains(end.saturating_sub(1)) {
                if !region.is_writable() {
                    return Err(MemoryError::ReadOnlyViolation { addr });
                }
            }
        }
        
        Ok(())
    }

    /// Get total guest memory size.
    pub fn guest_memory_size(&self) -> usize {
        self.guest_memory_size
    }

    /// Get the page table.
    pub fn page_table(&self) -> &Arc<PageTable> {
        &self.page_table
    }

    /// Get all mapped regions.
    pub fn regions(&self) -> Vec<MemoryRegion> {
        self.regions.read().values().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::region::RegionType;

    #[test]
    fn test_mmu_creation() {
        let mmu = Mmu::new(16 * 1024 * 1024); // 16 MB
        assert_eq!(mmu.guest_memory_size(), 16 * 1024 * 1024);
    }

    #[test]
    fn test_map_and_access() {
        let mmu = Mmu::new(16 * 1024 * 1024);
        
        let region = MemoryRegion::new(
            "test_ram",
            0x1000,
            0x2000,
            RegionType::Ram,
            RegionPermissions::RW,
        );
        mmu.map_region(region).unwrap();
        
        // Write and read back
        mmu.write_u32(0x1000, 0xDEADBEEF).unwrap();
        let value = mmu.read_u32(0x1000).unwrap();
        assert_eq!(value, 0xDEADBEEF);
    }

    #[test]
    fn test_unmapped_access() {
        let mmu = Mmu::new(16 * 1024 * 1024);
        let result = mmu.read_u32(0x1000);
        assert!(result.is_err());
    }
}
