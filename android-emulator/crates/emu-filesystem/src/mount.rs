//! Mount management.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::error::{FsError, FsResult};

/// Mount options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MountOptions {
    /// Read-only mount.
    pub read_only: bool,
    /// No execute permission.
    pub no_exec: bool,
    /// No setuid.
    pub no_suid: bool,
    /// No device files.
    pub no_dev: bool,
    /// Synchronous I/O.
    pub sync: bool,
}

impl Default for MountOptions {
    fn default() -> Self {
        Self {
            read_only: false,
            no_exec: false,
            no_suid: true,
            no_dev: true,
            sync: false,
        }
    }
}

/// Mount point information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MountPoint {
    /// Mount path in the virtual filesystem.
    pub target: String,
    /// Source (device or host path).
    pub source: String,
    /// Filesystem type.
    pub fs_type: String,
    /// Mount options.
    pub options: MountOptions,
}

impl MountPoint {
    /// Create a new mount point.
    pub fn new(target: &str, source: &str, fs_type: &str) -> Self {
        Self {
            target: target.to_string(),
            source: source.to_string(),
            fs_type: fs_type.to_string(),
            options: MountOptions::default(),
        }
    }

    /// Set mount options.
    pub fn with_options(mut self, options: MountOptions) -> Self {
        self.options = options;
        self
    }
}

/// Mount manager for the emulator.
pub struct MountManager {
    /// Active mounts by target path.
    mounts: RwLock<HashMap<String, MountPoint>>,
    /// Host paths that are allowed for passthrough.
    allowed_host_paths: RwLock<Vec<PathBuf>>,
}

impl MountManager {
    /// Create a new mount manager.
    pub fn new() -> Self {
        Self {
            mounts: RwLock::new(HashMap::new()),
            allowed_host_paths: RwLock::new(Vec::new()),
        }
    }

    /// Allow a host path for passthrough mounts.
    pub fn allow_host_path(&self, path: impl Into<PathBuf>) {
        self.allowed_host_paths.write().push(path.into());
    }

    /// Check if a host path is allowed.
    pub fn is_host_path_allowed(&self, path: &PathBuf) -> bool {
        let allowed = self.allowed_host_paths.read();
        for allowed_path in allowed.iter() {
            if path.starts_with(allowed_path) {
                return true;
            }
        }
        false
    }

    /// Mount a filesystem.
    pub fn mount(&self, mount_point: MountPoint) -> FsResult<()> {
        let target = mount_point.target.clone();
        
        // Check for existing mount
        if self.mounts.read().contains_key(&target) {
            return Err(FsError::MountError {
                reason: format!("{} is already mounted", target),
            });
        }

        // For passthrough mounts, verify the host path is allowed
        if mount_point.fs_type == "passthrough" {
            let host_path = PathBuf::from(&mount_point.source);
            if !self.is_host_path_allowed(&host_path) {
                return Err(FsError::PermissionDenied {
                    path: mount_point.source.clone(),
                });
            }
        }

        self.mounts.write().insert(target, mount_point);
        Ok(())
    }

    /// Unmount a filesystem.
    pub fn umount(&self, target: &str) -> FsResult<()> {
        if self.mounts.write().remove(target).is_none() {
            return Err(FsError::MountError {
                reason: format!("{} is not mounted", target),
            });
        }
        Ok(())
    }

    /// Get mount point for a path.
    pub fn find_mount(&self, path: &str) -> Option<MountPoint> {
        let mounts = self.mounts.read();
        
        // Find the longest matching mount point
        let mut best_match: Option<&MountPoint> = None;
        let mut best_len = 0;
        
        for (target, mount) in mounts.iter() {
            if path.starts_with(target) && target.len() > best_len {
                best_match = Some(mount);
                best_len = target.len();
            }
        }
        
        best_match.cloned()
    }

    /// List all mounts.
    pub fn list_mounts(&self) -> Vec<MountPoint> {
        self.mounts.read().values().cloned().collect()
    }

    /// Setup standard Android mounts.
    pub fn setup_android_mounts(&self) -> FsResult<()> {
        // Virtual filesystems
        self.mount(MountPoint::new("/proc", "proc", "procfs"))?;
        self.mount(MountPoint::new("/sys", "sysfs", "sysfs"))?;
        self.mount(MountPoint::new("/dev", "devfs", "devfs"))?;
        
        // Data partition
        self.mount(MountPoint::new("/data", "userdata", "ext4"))?;
        
        // System partition (read-only)
        self.mount(MountPoint::new("/system", "system", "ext4")
            .with_options(MountOptions { read_only: true, ..Default::default() }))?;
        
        // Vendor partition (read-only)
        self.mount(MountPoint::new("/vendor", "vendor", "ext4")
            .with_options(MountOptions { read_only: true, ..Default::default() }))?;
        
        Ok(())
    }
}

impl Default for MountManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mount_umount() {
        let manager = MountManager::new();
        
        let mp = MountPoint::new("/test", "test_device", "ext4");
        manager.mount(mp).unwrap();
        
        let found = manager.find_mount("/test/file.txt").unwrap();
        assert_eq!(found.target, "/test");
        
        manager.umount("/test").unwrap();
        assert!(manager.find_mount("/test/file.txt").is_none());
    }

    #[test]
    fn test_android_mounts() {
        let manager = MountManager::new();
        manager.setup_android_mounts().unwrap();
        
        assert!(manager.find_mount("/data/app").is_some());
        assert!(manager.find_mount("/system/bin").is_some());
    }

    #[test]
    fn test_passthrough_security() {
        let manager = MountManager::new();
        
        // Without allowing the path, mount should fail
        let mp = MountPoint::new("/mnt/host", "/home/user", "passthrough");
        let result = manager.mount(mp);
        assert!(result.is_err());
        
        // After allowing, it should work
        manager.allow_host_path("/home/user");
        let mp = MountPoint::new("/mnt/host", "/home/user", "passthrough");
        manager.mount(mp).unwrap();
    }
}
