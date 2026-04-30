//! # emu-core
//!
//! Core emulator orchestration and lifecycle management.
//!
//! This crate ties together all emulator components and provides
//! the main entry point for running the emulator.
//!
//! ## Legal Notice
//!
//! This implementation is original work. Licensed under Apache 2.0.

#![warn(missing_docs)]

mod config;
mod emulator;
mod error;

pub use config::{EmulatorConfig, DeviceProfile, DisplayConfig};
pub use emulator::{Emulator, EmulatorState};
pub use error::{EmulatorError, EmulatorResult};

/// Emulator version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default emulator name.
pub const EMULATOR_NAME: &str = "QRATUM Android Emulator";
