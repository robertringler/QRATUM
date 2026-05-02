//! EXT4 filesystem emulation.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::error::{FsError, FsResult};
use crate::vfs::VirtualFs;

/// EXT4 superblock information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ext4Superblock {
    /// Total inode count.
    pub inode_count: u32,
    /// Total block count.
    pub block_count: u64,
    /// Free block count.
    pub free_blocks: u64,
    /// Free inode count.
    pub free_inodes: u32,
    /// Block size (typically 4096).
    pub block_size: u32,
    /// Inode size.
    pub inode_size: u16,
    /// Volume name.
    pub volume_name: String,
    /// Last mount time.
    pub last_mount_time: u64,
    /// Last write time.
    pub last_write_time: u64,
    /// Mount count.
    pub mount_count: u16,
    /// Maximum mount count before fsck.
    pub max_mount_count: u16,
    /// Filesystem state.
    pub state: u16,
}

impl Default for Ext4Superblock {
    fn default() -> Self {
        Self {
            inode_count: 1_000_000,
            block_count: 10_000_000,
            free_blocks: 9_000_000,
            free_inodes: 999_900,
            block_size: 4096,
            inode_size: 256,
            volume_name: "Android Data".to_string(),
            last_mount_time: 0,
            last_write_time: 0,
            mount_count: 0,
            max_mount_count: 100,
            state: 1, // Clean
        }
    }
}

/// EXT4 filesystem emulator.
pub struct Ext4Emulator {
    /// Superblock.
    superblock: RwLock<Ext4Superblock>,
    /// Virtual filesystem backend.
    vfs: VirtualFs,
}

impl Ext4Emulator {
    /// Create a new EXT4 emulator.
    pub fn new() -> Self {
        Self {
            superblock: RwLock::new(Ext4Superblock::default()),
            vfs: VirtualFs::new(),
        }
    }

    /// Get the superblock.
    pub fn superblock(&self) -> Ext4Superblock {
        self.superblock.read().clone()
    }

    /// Get the virtual filesystem.
    pub fn vfs(&self) -> &VirtualFs {
        &self.vfs
    }

    /// Initialize standard Android directory structure.
    pub fn init_android_structure(&self) -> FsResult<()> {
        use crate::vfs::FilePermissions;

        // Create standard Android directories
        let dirs = [
            "/data",
            "/data/app",
            "/data/data",
            "/data/local",
            "/data/local/tmp",
            "/data/user",
            "/data/user/0",
            "/system",
            "/system/bin",
            "/system/lib",
            "/system/lib64",
            "/vendor",
            "/vendor/lib",
            "/vendor/lib64",
            "/sdcard",
            "/sdcard/Download",
            "/sdcard/DCIM",
            "/sdcard/Pictures",
            "/sdcard/Documents",
            "/storage",
            "/storage/emulated",
            "/storage/emulated/0",
            "/cache",
            "/proc",
            "/dev",
        ];

        for dir in &dirs {
            if !self.vfs.exists(dir) {
                self.vfs.mkdir(dir, FilePermissions::default())?;
            }
        }

        Ok(())
    }

    /// Get filesystem statistics.
    pub fn statfs(&self) -> FsResult<FsStats> {
        let sb = self.superblock.read();
        Ok(FsStats {
            total_blocks: sb.block_count,
            free_blocks: sb.free_blocks,
            available_blocks: sb.free_blocks,
            total_inodes: sb.inode_count as u64,
            free_inodes: sb.free_inodes as u64,
            block_size: sb.block_size as u64,
            name_max: 255,
        })
    }

    /// Sync filesystem (no-op for emulator).
    pub fn sync(&self) -> FsResult<()> {
        let mut sb = self.superblock.write();
        sb.last_write_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Ok(())
    }
}

impl Default for Ext4Emulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Filesystem statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FsStats {
    /// Total data blocks.
    pub total_blocks: u64,
    /// Free blocks.
    pub free_blocks: u64,
    /// Available blocks (may differ due to reserved space).
    pub available_blocks: u64,
    /// Total inodes.
    pub total_inodes: u64,
    /// Free inodes.
    pub free_inodes: u64,
    /// Block size in bytes.
    pub block_size: u64,
    /// Maximum filename length.
    pub name_max: u64,
}

impl FsStats {
    /// Get total size in bytes.
    pub fn total_size(&self) -> u64 {
        self.total_blocks * self.block_size
    }

    /// Get free size in bytes.
    pub fn free_size(&self) -> u64 {
        self.free_blocks * self.block_size
    }

    /// Get used size in bytes.
    pub fn used_size(&self) -> u64 {
        (self.total_blocks - self.free_blocks) * self.block_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ext4_creation() {
        let ext4 = Ext4Emulator::new();
        let sb = ext4.superblock();
        assert_eq!(sb.block_size, 4096);
    }

    #[test]
    fn test_android_structure() {
        let ext4 = Ext4Emulator::new();
        ext4.init_android_structure().unwrap();
        
        assert!(ext4.vfs().exists("/data"));
        assert!(ext4.vfs().exists("/system"));
        assert!(ext4.vfs().exists("/sdcard"));
    }

    #[test]
    fn test_statfs() {
        let ext4 = Ext4Emulator::new();
        let stats = ext4.statfs().unwrap();
        
        assert!(stats.total_size() > 0);
        assert!(stats.free_size() > 0);
    }
}
