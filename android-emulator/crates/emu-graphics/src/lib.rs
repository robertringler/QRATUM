//! # emu-graphics
//!
//! GPU and graphics emulation for the Android emulator.
//!
//! This crate provides:
//! - Vulkan translation layer
//! - OpenGL ES → Vulkan backend
//! - Shader transpilation (SPIR-V)
//! - Frame pacing and vsync control
//!
//! ## Legal Notice
//!
//! This implementation is original work. Licensed under Apache 2.0.

#![warn(missing_docs)]

mod compositor;
mod error;
mod framebuffer;
mod gles;

pub use compositor::{Compositor, CompositorConfig, Layer};
pub use error::{GraphicsError, GraphicsResult};
pub use framebuffer::{Framebuffer, FramebufferConfig, PixelFormat};
pub use gles::{GlesContext, GlesState};

/// Default display width.
pub const DEFAULT_WIDTH: u32 = 1080;

/// Default display height.
pub const DEFAULT_HEIGHT: u32 = 1920;

/// Default refresh rate in Hz.
pub const DEFAULT_REFRESH_RATE: u32 = 60;
