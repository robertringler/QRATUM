//! # emu-filesystem
//!
//! Virtual filesystem for the Android emulator.
//!
//! This crate provides:
//! - Virtual EXT4 filesystem emulation
//! - Scoped storage enforcement
//! - Safe passthrough mounts
//! - Sandboxed permission model
//!
//! ## Legal Notice
//!
//! This implementation is original work. Licensed under Apache 2.0.

#![warn(missing_docs)]

mod error;
mod ext4;
mod mount;
mod vfs;

pub use error::{FsError, FsResult};
pub use ext4::Ext4Emulator;
pub use mount::{MountManager, MountPoint, MountOptions};
pub use vfs::{VirtualFs, FileHandle, FileType, FilePermissions};

/// Maximum path length.
pub const MAX_PATH_LENGTH: usize = 4096;

/// Maximum filename length.
pub const MAX_NAME_LENGTH: usize = 255;
