//! # emu-memory
//!
//! Virtual memory management for the Android emulator.
//!
//! This crate provides:
//! - Virtual Memory Unit (MMU) emulation
//! - Page table management
//! - Copy-on-write (CoW) snapshot support
//! - Memory-mapped I/O regions
//!
//! ## Architecture
//!
//! The memory subsystem uses a two-level page table structure:
//! - L1: 4KB pages (standard ARM/x86 page size)
//! - L2: 2MB huge pages (optional, for performance)
//!
//! ## Legal Notice
//!
//! This implementation is original work, written from scratch without
//! reference to proprietary code. Licensed under Apache 2.0.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

mod error;
mod mmu;
mod page_table;
mod region;
mod snapshot;

pub use error::{MemoryError, MemoryResult};
pub use mmu::Mmu;
pub use page_table::{PageEntry, PageFlags, PageTable};
pub use region::{MemoryRegion, RegionPermissions, RegionType};
pub use snapshot::{MemorySnapshot, SnapshotConfig};

/// Default page size (4KB)
pub const PAGE_SIZE: usize = 4096;

/// Page size shift (log2 of PAGE_SIZE)
pub const PAGE_SHIFT: usize = 12;

/// Huge page size (2MB)
pub const HUGE_PAGE_SIZE: usize = 2 * 1024 * 1024;

/// Huge page shift (log2 of HUGE_PAGE_SIZE)
pub const HUGE_PAGE_SHIFT: usize = 21;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_constants() {
        assert_eq!(PAGE_SIZE, 1 << PAGE_SHIFT);
        assert_eq!(HUGE_PAGE_SIZE, 1 << HUGE_PAGE_SHIFT);
    }
}
