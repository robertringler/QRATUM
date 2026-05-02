//! # emu-audio
//!
//! Audio emulation for the Android emulator.
//!
//! This crate provides:
//! - OpenSL ES emulation
//! - ALSA/AAudio backend abstraction
//! - Audio mixing and routing
//!
//! ## Legal Notice
//!
//! This implementation is original work. Licensed under Apache 2.0.

#![warn(missing_docs)]

mod error;
mod mixer;
mod opensl;
mod stream;

pub use error::{AudioError, AudioResult};
pub use mixer::{AudioMixer, MixerConfig};
pub use opensl::{OpenSlEngine, OpenSlPlayer};
pub use stream::{AudioStream, AudioFormat, StreamConfig};

/// Default sample rate in Hz.
pub const DEFAULT_SAMPLE_RATE: u32 = 48000;

/// Default channel count (stereo).
pub const DEFAULT_CHANNELS: u32 = 2;

/// Default buffer size in frames.
pub const DEFAULT_BUFFER_FRAMES: u32 = 256;
