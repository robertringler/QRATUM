//! # emu-input
//!
//! Input device emulation for the Android emulator.
//!
//! This crate provides:
//! - Touch screen emulation
//! - Keyboard mapping
//! - Mouse/trackpad support
//! - Gamepad HID support
//!
//! ## Legal Notice
//!
//! This implementation is original work. Licensed under Apache 2.0.

#![warn(missing_docs)]

mod error;
mod gamepad;
mod keyboard;
mod touch;

pub use error::{InputError, InputResult};
pub use gamepad::{GamepadState, GamepadButton, GamepadAxis};
pub use keyboard::{KeyboardState, KeyCode, KeyEvent};
pub use touch::{TouchState, TouchEvent, TouchPoint};

/// Maximum number of simultaneous touch points.
pub const MAX_TOUCH_POINTS: usize = 10;
