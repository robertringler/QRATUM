//! Memory region management.
//!
//! Defines different types of memory regions and their permissions.

use bitflags::bitflags;
use serde::{Deserialize, Serialize};

bitflags! {
    /// Permissions for a memory region.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct RegionPermissions: u32 {
        /// Region is readable.
        const READ = 1 << 0;
        /// Region is writable.
        const WRITE = 1 << 1;
        /// Region is executable.
        const EXECUTE = 1 << 2;
        /// Read-write shorthand.
        const RW = Self::READ.bits() | Self::WRITE.bits();
        /// Read-execute shorthand.
        const RX = Self::READ.bits() | Self::EXECUTE.bits();
        /// Read-write-execute shorthand.
        const RWX = Self::READ.bits() | Self::WRITE.bits() | Self::EXECUTE.bits();
    }
}

// Manual Serialize/Deserialize for bitflags
impl Serialize for RegionPermissions {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.bits().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for RegionPermissions {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bits = u32::deserialize(deserializer)?;
        Ok(Self::from_bits_truncate(bits))
    }
}

impl Default for RegionPermissions {
    fn default() -> Self {
        Self::READ
    }
}

/// Type of memory region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegionType {
    /// Regular RAM.
    Ram,
    /// Read-only memory (e.g., ROM, firmware).
    Rom,
    /// Memory-mapped I/O device.
    Mmio,
    /// Framebuffer memory.
    Framebuffer,
    /// DMA buffer.
    Dma,
    /// Stack memory.
    Stack,
    /// Heap memory.
    Heap,
    /// Shared memory region.
    Shared,
    /// Guard page (inaccessible, for detecting overflows).
    Guard,
}

/// A contiguous memory region with associated metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRegion {
    /// Human-readable name for the region.
    pub name: String,
    /// Start address of the region.
    pub start: u64,
    /// Size of the region in bytes.
    pub size: u64,
    /// Region type.
    pub region_type: RegionType,
    /// Access permissions.
    pub permissions: RegionPermissions,
    /// Whether the region is backed by host memory.
    pub host_backed: bool,
    /// Optional file path for file-backed regions.
    pub backing_file: Option<String>,
}

impl MemoryRegion {
    /// Create a new memory region.
    pub fn new(
        name: impl Into<String>,
        start: u64,
        size: u64,
        region_type: RegionType,
        permissions: RegionPermissions,
    ) -> Self {
        Self {
            name: name.into(),
            start,
            size,
            region_type,
            permissions,
            host_backed: true,
            backing_file: None,
        }
    }

    /// Create a RAM region.
    pub fn ram(name: impl Into<String>, start: u64, size: u64) -> Self {
        Self::new(name, start, size, RegionType::Ram, RegionPermissions::RW)
    }

    /// Create a ROM region.
    pub fn rom(name: impl Into<String>, start: u64, size: u64) -> Self {
        Self::new(name, start, size, RegionType::Rom, RegionPermissions::READ)
    }

    /// Create an MMIO region.
    pub fn mmio(name: impl Into<String>, start: u64, size: u64) -> Self {
        Self::new(name, start, size, RegionType::Mmio, RegionPermissions::RW)
    }

    /// End address (exclusive) of the region.
    pub fn end(&self) -> u64 {
        self.start.saturating_add(self.size)
    }

    /// Check if an address is within this region.
    pub fn contains(&self, addr: u64) -> bool {
        addr >= self.start && addr < self.end()
    }

    /// Check if this region overlaps with another.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start < other.end() && other.start < self.end()
    }

    /// Check if the region is readable.
    pub fn is_readable(&self) -> bool {
        self.permissions.contains(RegionPermissions::READ)
    }

    /// Check if the region is writable.
    pub fn is_writable(&self) -> bool {
        self.permissions.contains(RegionPermissions::WRITE)
    }

    /// Check if the region is executable.
    pub fn is_executable(&self) -> bool {
        self.permissions.contains(RegionPermissions::EXECUTE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_creation() {
        let region = MemoryRegion::ram("test_ram", 0x1000, 0x2000);
        assert_eq!(region.start, 0x1000);
        assert_eq!(region.size, 0x2000);
        assert_eq!(region.end(), 0x3000);
        assert!(region.is_readable());
        assert!(region.is_writable());
        assert!(!region.is_executable());
    }

    #[test]
    fn test_region_contains() {
        let region = MemoryRegion::ram("test", 0x1000, 0x1000);
        assert!(!region.contains(0x0FFF));
        assert!(region.contains(0x1000));
        assert!(region.contains(0x1FFF));
        assert!(!region.contains(0x2000));
    }

    #[test]
    fn test_region_overlap() {
        let region1 = MemoryRegion::ram("r1", 0x1000, 0x1000);
        let region2 = MemoryRegion::ram("r2", 0x1800, 0x1000);
        let region3 = MemoryRegion::ram("r3", 0x3000, 0x1000);
        
        assert!(region1.overlaps(&region2));
        assert!(region2.overlaps(&region1));
        assert!(!region1.overlaps(&region3));
    }
}
