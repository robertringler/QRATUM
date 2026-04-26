//! # emu-pc-compat
//!
//! PC Game Compatibility Research Layer.
//!
//! This crate provides a technical research environment for:
//! - Studying x86 Windows app compatibility
//! - Syscall translation
//! - Graphics API abstraction
//!
//! ## Legal Notice
//!
//! This module does NOT include any games or proprietary software.
//! It is designed for research purposes only and accepts user-supplied,
//! legally owned software. Licensed under Apache 2.0.

#![warn(missing_docs)]

mod directx;
mod error;
mod syscall;
mod timing;

pub use directx::{DxToVulkanTranslator, D3DFeatureLevel};
pub use error::{PcCompatError, PcCompatResult};
pub use syscall::{Win32SyscallShim, SyscallTrace};
pub use timing::{TimingNormalizer, FrameLimiter};

/// Major Windows version being targeted.
pub const TARGET_WINDOWS_VERSION: u32 = 10;

/// Minimum Direct3D feature level.
pub const MIN_D3D_FEATURE_LEVEL: u32 = 0xb000; // 11_0
