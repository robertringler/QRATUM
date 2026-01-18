//! Virtual filesystem abstraction.

use bitflags::bitflags;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::error::{FsError, FsResult};

/// File handle type.
pub type FileHandle = u64;

/// Invalid file handle.
pub const INVALID_HANDLE: FileHandle = 0;

bitflags! {
    /// File permissions.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct FilePermissions: u32 {
        /// Owner read.
        const OWNER_READ = 0o400;
        /// Owner write.
        const OWNER_WRITE = 0o200;
        /// Owner execute.
        const OWNER_EXEC = 0o100;
        /// Group read.
        const GROUP_READ = 0o040;
        /// Group write.
        const GROUP_WRITE = 0o020;
        /// Group execute.
        const GROUP_EXEC = 0o010;
        /// Other read.
        const OTHER_READ = 0o004;
        /// Other write.
        const OTHER_WRITE = 0o002;
        /// Other execute.
        const OTHER_EXEC = 0o001;
    }
}

// Manual Serialize/Deserialize for bitflags
impl Serialize for FilePermissions {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.bits().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FilePermissions {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bits = u32::deserialize(deserializer)?;
        Ok(Self::from_bits_truncate(bits))
    }
}

impl Default for FilePermissions {
    fn default() -> Self {
        Self::OWNER_READ | Self::OWNER_WRITE | Self::GROUP_READ | Self::OTHER_READ
    }
}

/// File type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileType {
    /// Regular file.
    Regular,
    /// Directory.
    Directory,
    /// Symbolic link.
    Symlink,
    /// Block device.
    BlockDevice,
    /// Character device.
    CharDevice,
    /// FIFO (named pipe).
    Fifo,
    /// Socket.
    Socket,
}

/// File open flags.
bitflags! {
    /// Flags for opening files.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct OpenFlags: u32 {
        /// Read only.
        const READ = 0x0001;
        /// Write only.
        const WRITE = 0x0002;
        /// Create if not exists.
        const CREATE = 0x0004;
        /// Truncate if exists.
        const TRUNCATE = 0x0008;
        /// Append mode.
        const APPEND = 0x0010;
        /// Exclusive creation.
        const EXCLUSIVE = 0x0020;
    }
}

// Manual Serialize/Deserialize for bitflags
impl Serialize for OpenFlags {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.bits().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for OpenFlags {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bits = u32::deserialize(deserializer)?;
        Ok(Self::from_bits_truncate(bits))
    }
}

/// Inode data.
#[derive(Debug, Clone)]
struct Inode {
    /// Inode number.
    ino: u64,
    /// File type.
    file_type: FileType,
    /// Permissions.
    permissions: FilePermissions,
    /// Owner UID.
    uid: u32,
    /// Owner GID.
    gid: u32,
    /// Size in bytes.
    size: u64,
    /// Creation time.
    ctime: u64,
    /// Modification time.
    mtime: u64,
    /// Access time.
    atime: u64,
    /// Link count.
    nlink: u32,
    /// Data (for regular files).
    data: Vec<u8>,
    /// Children (for directories).
    children: HashMap<String, u64>,
    /// Symlink target.
    symlink_target: Option<String>,
}

impl Inode {
    fn new_file(ino: u64) -> Self {
        Self {
            ino,
            file_type: FileType::Regular,
            permissions: FilePermissions::default(),
            uid: 0,
            gid: 0,
            size: 0,
            ctime: 0,
            mtime: 0,
            atime: 0,
            nlink: 1,
            data: Vec::new(),
            children: HashMap::new(),
            symlink_target: None,
        }
    }

    fn new_dir(ino: u64) -> Self {
        Self {
            ino,
            file_type: FileType::Directory,
            permissions: FilePermissions::default() | FilePermissions::OWNER_EXEC | FilePermissions::GROUP_EXEC | FilePermissions::OTHER_EXEC,
            uid: 0,
            gid: 0,
            size: 0,
            ctime: 0,
            mtime: 0,
            atime: 0,
            nlink: 2,
            data: Vec::new(),
            children: HashMap::new(),
            symlink_target: None,
        }
    }
}

/// Open file descriptor.
#[derive(Debug)]
struct OpenFile {
    /// Inode number.
    ino: u64,
    /// Current position.
    position: u64,
    /// Open flags.
    flags: OpenFlags,
}

/// Virtual filesystem.
pub struct VirtualFs {
    /// Inodes.
    inodes: RwLock<HashMap<u64, Inode>>,
    /// Next inode number.
    next_ino: AtomicU64,
    /// Open files.
    open_files: RwLock<HashMap<FileHandle, OpenFile>>,
    /// Next file handle.
    next_handle: AtomicU64,
    /// Root inode number.
    root_ino: u64,
}

impl VirtualFs {
    /// Create a new virtual filesystem.
    pub fn new() -> Self {
        let mut inodes = HashMap::new();
        
        // Create root directory
        let root = Inode::new_dir(1);
        inodes.insert(1, root);

        Self {
            inodes: RwLock::new(inodes),
            next_ino: AtomicU64::new(2),
            open_files: RwLock::new(HashMap::new()),
            next_handle: AtomicU64::new(1),
            root_ino: 1,
        }
    }

    fn alloc_ino(&self) -> u64 {
        self.next_ino.fetch_add(1, Ordering::SeqCst)
    }

    fn alloc_handle(&self) -> FileHandle {
        self.next_handle.fetch_add(1, Ordering::SeqCst)
    }

    /// Look up a path and return the inode number.
    fn lookup_path(&self, path: &str) -> FsResult<u64> {
        let inodes = self.inodes.read();
        
        if path == "/" {
            return Ok(self.root_ino);
        }

        let path = path.trim_start_matches('/');
        let mut current_ino = self.root_ino;

        for component in path.split('/') {
            if component.is_empty() || component == "." {
                continue;
            }

            let inode = inodes.get(&current_ino).ok_or_else(|| FsError::NotFound {
                path: path.to_string(),
            })?;

            if inode.file_type != FileType::Directory {
                return Err(FsError::NotDirectory {
                    path: path.to_string(),
                });
            }

            current_ino = *inode.children.get(component).ok_or_else(|| FsError::NotFound {
                path: path.to_string(),
            })?;
        }

        Ok(current_ino)
    }

    /// Create a directory.
    pub fn mkdir(&self, path: &str, permissions: FilePermissions) -> FsResult<()> {
        let (parent_path, name) = self.split_path(path)?;
        let parent_ino = self.lookup_path(&parent_path)?;

        let mut inodes = self.inodes.write();
        let parent = inodes.get_mut(&parent_ino).ok_or_else(|| FsError::NotFound {
            path: parent_path.clone(),
        })?;

        if parent.file_type != FileType::Directory {
            return Err(FsError::NotDirectory { path: parent_path });
        }

        if parent.children.contains_key(&name) {
            return Err(FsError::AlreadyExists { path: path.to_string() });
        }

        let ino = self.alloc_ino();
        let mut dir = Inode::new_dir(ino);
        dir.permissions = permissions;
        
        parent.children.insert(name, ino);
        parent.nlink += 1;
        
        inodes.insert(ino, dir);

        Ok(())
    }

    /// Create a file.
    pub fn create(&self, path: &str, flags: OpenFlags) -> FsResult<FileHandle> {
        let (parent_path, name) = self.split_path(path)?;
        let parent_ino = self.lookup_path(&parent_path)?;

        let mut inodes = self.inodes.write();
        let parent = inodes.get_mut(&parent_ino).ok_or_else(|| FsError::NotFound {
            path: parent_path.clone(),
        })?;

        if parent.file_type != FileType::Directory {
            return Err(FsError::NotDirectory { path: parent_path });
        }

        if let Some(&existing_ino) = parent.children.get(&name) {
            if flags.contains(OpenFlags::EXCLUSIVE) {
                return Err(FsError::AlreadyExists { path: path.to_string() });
            }
            
            // Open existing file
            let handle = self.alloc_handle();
            let open_file = OpenFile {
                ino: existing_ino,
                position: 0,
                flags,
            };
            
            if flags.contains(OpenFlags::TRUNCATE) {
                if let Some(inode) = inodes.get_mut(&existing_ino) {
                    inode.data.clear();
                    inode.size = 0;
                }
            }
            
            self.open_files.write().insert(handle, open_file);
            return Ok(handle);
        }

        let ino = self.alloc_ino();
        let file = Inode::new_file(ino);
        
        parent.children.insert(name, ino);
        inodes.insert(ino, file);

        let handle = self.alloc_handle();
        let open_file = OpenFile {
            ino,
            position: 0,
            flags,
        };
        self.open_files.write().insert(handle, open_file);

        Ok(handle)
    }

    /// Open a file.
    pub fn open(&self, path: &str, flags: OpenFlags) -> FsResult<FileHandle> {
        if flags.contains(OpenFlags::CREATE) {
            return self.create(path, flags);
        }

        let ino = self.lookup_path(path)?;
        
        let inodes = self.inodes.read();
        let inode = inodes.get(&ino).ok_or_else(|| FsError::NotFound {
            path: path.to_string(),
        })?;

        if inode.file_type != FileType::Regular {
            return Err(FsError::NotFile { path: path.to_string() });
        }

        let handle = self.alloc_handle();
        let open_file = OpenFile {
            ino,
            position: if flags.contains(OpenFlags::APPEND) { inode.size } else { 0 },
            flags,
        };
        self.open_files.write().insert(handle, open_file);

        Ok(handle)
    }

    /// Close a file.
    pub fn close(&self, handle: FileHandle) -> FsResult<()> {
        if self.open_files.write().remove(&handle).is_none() {
            return Err(FsError::InvalidHandle);
        }
        Ok(())
    }

    /// Read from a file.
    pub fn read(&self, handle: FileHandle, buf: &mut [u8]) -> FsResult<usize> {
        let mut open_files = self.open_files.write();
        let open_file = open_files.get_mut(&handle).ok_or(FsError::InvalidHandle)?;

        if !open_file.flags.contains(OpenFlags::READ) {
            return Err(FsError::PermissionDenied { path: "".to_string() });
        }

        let inodes = self.inodes.read();
        let inode = inodes.get(&open_file.ino).ok_or(FsError::InvalidHandle)?;

        let pos = open_file.position as usize;
        let available = inode.data.len().saturating_sub(pos);
        let to_read = buf.len().min(available);

        buf[..to_read].copy_from_slice(&inode.data[pos..pos + to_read]);
        open_file.position += to_read as u64;

        Ok(to_read)
    }

    /// Write to a file.
    pub fn write(&self, handle: FileHandle, data: &[u8]) -> FsResult<usize> {
        let mut open_files = self.open_files.write();
        let open_file = open_files.get_mut(&handle).ok_or(FsError::InvalidHandle)?;

        if !open_file.flags.contains(OpenFlags::WRITE) {
            return Err(FsError::PermissionDenied { path: "".to_string() });
        }

        let mut inodes = self.inodes.write();
        let inode = inodes.get_mut(&open_file.ino).ok_or(FsError::InvalidHandle)?;

        let pos = if open_file.flags.contains(OpenFlags::APPEND) {
            inode.data.len()
        } else {
            open_file.position as usize
        };

        // Extend if necessary
        if pos + data.len() > inode.data.len() {
            inode.data.resize(pos + data.len(), 0);
        }

        inode.data[pos..pos + data.len()].copy_from_slice(data);
        open_file.position = (pos + data.len()) as u64;
        inode.size = inode.data.len() as u64;

        Ok(data.len())
    }

    /// Seek in a file.
    pub fn seek(&self, handle: FileHandle, offset: i64, whence: i32) -> FsResult<u64> {
        let mut open_files = self.open_files.write();
        let open_file = open_files.get_mut(&handle).ok_or(FsError::InvalidHandle)?;

        let inodes = self.inodes.read();
        let inode = inodes.get(&open_file.ino).ok_or(FsError::InvalidHandle)?;

        let new_pos = match whence {
            0 => offset as u64, // SEEK_SET
            1 => (open_file.position as i64 + offset) as u64, // SEEK_CUR
            2 => (inode.size as i64 + offset) as u64, // SEEK_END
            _ => return Err(FsError::InvalidPath { reason: "invalid whence".to_string() }),
        };

        open_file.position = new_pos;
        Ok(new_pos)
    }

    /// Get file size.
    pub fn file_size(&self, path: &str) -> FsResult<u64> {
        let ino = self.lookup_path(path)?;
        let inodes = self.inodes.read();
        let inode = inodes.get(&ino).ok_or_else(|| FsError::NotFound { path: path.to_string() })?;
        Ok(inode.size)
    }

    /// Check if a path exists.
    pub fn exists(&self, path: &str) -> bool {
        self.lookup_path(path).is_ok()
    }

    /// Check if a path is a directory.
    pub fn is_dir(&self, path: &str) -> bool {
        if let Ok(ino) = self.lookup_path(path) {
            if let Some(inode) = self.inodes.read().get(&ino) {
                return inode.file_type == FileType::Directory;
            }
        }
        false
    }

    /// List directory contents.
    pub fn readdir(&self, path: &str) -> FsResult<Vec<String>> {
        let ino = self.lookup_path(path)?;
        let inodes = self.inodes.read();
        let inode = inodes.get(&ino).ok_or_else(|| FsError::NotFound { path: path.to_string() })?;

        if inode.file_type != FileType::Directory {
            return Err(FsError::NotDirectory { path: path.to_string() });
        }

        Ok(inode.children.keys().cloned().collect())
    }

    /// Delete a file.
    pub fn unlink(&self, path: &str) -> FsResult<()> {
        let (parent_path, name) = self.split_path(path)?;
        let parent_ino = self.lookup_path(&parent_path)?;

        let mut inodes = self.inodes.write();
        
        // First get the child inode number from parent
        let child_ino = {
            let parent = inodes.get(&parent_ino).ok_or_else(|| FsError::NotFound {
                path: parent_path.clone(),
            })?;
            *parent.children.get(&name).ok_or_else(|| FsError::NotFound {
                path: path.to_string(),
            })?
        };

        // Check if it's a directory
        if let Some(child) = inodes.get(&child_ino) {
            if child.file_type == FileType::Directory {
                return Err(FsError::NotFile { path: path.to_string() });
            }
        }

        // Now remove from parent
        if let Some(parent) = inodes.get_mut(&parent_ino) {
            parent.children.remove(&name);
        }

        inodes.remove(&child_ino);
        Ok(())
    }

    /// Remove a directory.
    pub fn rmdir(&self, path: &str) -> FsResult<()> {
        let (parent_path, name) = self.split_path(path)?;
        let parent_ino = self.lookup_path(&parent_path)?;

        let mut inodes = self.inodes.write();
        
        // Get the directory inode
        let dir_ino = {
            let parent = inodes.get(&parent_ino).ok_or_else(|| FsError::NotFound {
                path: parent_path.clone(),
            })?;
            *parent.children.get(&name).ok_or_else(|| FsError::NotFound {
                path: path.to_string(),
            })?
        };

        // Check if it's a directory and if it's empty
        {
            let dir = inodes.get(&dir_ino).ok_or_else(|| FsError::NotFound {
                path: path.to_string(),
            })?;

            if dir.file_type != FileType::Directory {
                return Err(FsError::NotDirectory { path: path.to_string() });
            }

            if !dir.children.is_empty() {
                return Err(FsError::DirectoryNotEmpty { path: path.to_string() });
            }
        }

        // Remove from parent
        let parent = inodes.get_mut(&parent_ino).unwrap();
        parent.children.remove(&name);
        parent.nlink -= 1;

        inodes.remove(&dir_ino);
        Ok(())
    }

    fn split_path(&self, path: &str) -> FsResult<(String, String)> {
        let path = path.trim_end_matches('/');
        if path.is_empty() || path == "/" {
            return Err(FsError::InvalidPath { reason: "cannot split root path".to_string() });
        }

        if let Some(pos) = path.rfind('/') {
            let parent = if pos == 0 { "/".to_string() } else { path[..pos].to_string() };
            let name = path[pos + 1..].to_string();
            Ok((parent, name))
        } else {
            Ok(("/".to_string(), path.to_string()))
        }
    }
}

impl Default for VirtualFs {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mkdir() {
        let fs = VirtualFs::new();
        fs.mkdir("/test", FilePermissions::default()).unwrap();
        assert!(fs.exists("/test"));
        assert!(fs.is_dir("/test"));
    }

    #[test]
    fn test_create_file() {
        let fs = VirtualFs::new();
        fs.mkdir("/test", FilePermissions::default()).unwrap();
        
        let handle = fs.create("/test/file.txt", OpenFlags::CREATE | OpenFlags::WRITE).unwrap();
        fs.close(handle).unwrap();
        
        assert!(fs.exists("/test/file.txt"));
        assert!(!fs.is_dir("/test/file.txt"));
    }

    #[test]
    fn test_read_write() {
        let fs = VirtualFs::new();
        
        let handle = fs.create("/test.txt", OpenFlags::CREATE | OpenFlags::WRITE | OpenFlags::READ).unwrap();
        
        let data = b"Hello, World!";
        let written = fs.write(handle, data).unwrap();
        assert_eq!(written, data.len());
        
        fs.seek(handle, 0, 0).unwrap(); // Seek to start
        
        let mut buf = vec![0u8; 20];
        let read = fs.read(handle, &mut buf).unwrap();
        assert_eq!(read, data.len());
        assert_eq!(&buf[..read], data);
        
        fs.close(handle).unwrap();
    }

    #[test]
    fn test_readdir() {
        let fs = VirtualFs::new();
        fs.mkdir("/test", FilePermissions::default()).unwrap();
        fs.create("/test/a.txt", OpenFlags::CREATE | OpenFlags::WRITE).unwrap();
        fs.create("/test/b.txt", OpenFlags::CREATE | OpenFlags::WRITE).unwrap();
        
        let entries = fs.readdir("/test").unwrap();
        assert_eq!(entries.len(), 2);
    }
}
