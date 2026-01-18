//! # emu-android-runtime
//!
//! Android Runtime Layer emulation for the Android emulator.
//!
//! This crate provides:
//! - ART (Android Runtime) clean-room re-implementation
//! - Bionic libc compatibility layer
//! - Binder IPC emulation
//! - SurfaceFlinger-like compositor interface
//!
//! ## Legal Notice
//!
//! This implementation is original work, designed to be compatible with
//! Android AOSP interfaces without using proprietary code.
//! Licensed under Apache 2.0.

#![warn(missing_docs)]

mod binder;
mod bionic;
mod error;
mod runtime;
mod syscall;

pub use binder::{BinderMessage, BinderNode, BinderService};
pub use bionic::BionicEmulator;
pub use error::{RuntimeError, RuntimeResult};
pub use runtime::{AndroidRuntime, RuntimeConfig};
pub use syscall::{SyscallEmulator, SyscallResult};

/// Android API level being emulated.
pub const ANDROID_API_LEVEL: u32 = 33; // Android 13

/// ABI being emulated.
pub const TARGET_ABI: &str = "arm64-v8a";
