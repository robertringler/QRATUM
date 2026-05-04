//! Audio stream abstraction.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use crate::error::{AudioError, AudioResult};

/// Audio sample format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioFormat {
    /// 16-bit signed integer PCM.
    S16,
    /// 32-bit float PCM.
    F32,
    /// 8-bit unsigned integer PCM.
    U8,
}

impl AudioFormat {
    /// Get bytes per sample.
    pub fn bytes_per_sample(self) -> usize {
        match self {
            Self::S16 => 2,
            Self::F32 => 4,
            Self::U8 => 1,
        }
    }
}

impl Default for AudioFormat {
    fn default() -> Self {
        Self::S16
    }
}

/// Audio stream configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u32,
    /// Sample format.
    pub format: AudioFormat,
    /// Buffer size in frames.
    pub buffer_frames: u32,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            sample_rate: crate::DEFAULT_SAMPLE_RATE,
            channels: crate::DEFAULT_CHANNELS,
            format: AudioFormat::S16,
            buffer_frames: crate::DEFAULT_BUFFER_FRAMES,
        }
    }
}

/// Audio stream state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamState {
    /// Stream is stopped.
    Stopped,
    /// Stream is playing.
    Playing,
    /// Stream is paused.
    Paused,
}

/// Audio stream for playback or capture.
pub struct AudioStream {
    /// Configuration.
    config: StreamConfig,
    /// Current state.
    state: RwLock<StreamState>,
    /// Audio buffer.
    buffer: RwLock<VecDeque<u8>>,
    /// Frames written.
    frames_written: RwLock<u64>,
    /// Frames read.
    frames_read: RwLock<u64>,
}

impl AudioStream {
    /// Create a new audio stream.
    pub fn new(config: StreamConfig) -> Self {
        let buffer_size = config.buffer_frames as usize
            * config.channels as usize
            * config.format.bytes_per_sample()
            * 4; // 4x buffer for latency headroom

        Self {
            config,
            state: RwLock::new(StreamState::Stopped),
            buffer: RwLock::new(VecDeque::with_capacity(buffer_size)),
            frames_written: RwLock::new(0),
            frames_read: RwLock::new(0),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(StreamConfig::default())
    }

    /// Get the configuration.
    pub fn config(&self) -> &StreamConfig {
        &self.config
    }

    /// Get the current state.
    pub fn state(&self) -> StreamState {
        *self.state.read()
    }

    /// Start the stream.
    pub fn start(&self) -> AudioResult<()> {
        *self.state.write() = StreamState::Playing;
        Ok(())
    }

    /// Stop the stream.
    pub fn stop(&self) -> AudioResult<()> {
        *self.state.write() = StreamState::Stopped;
        self.buffer.write().clear();
        Ok(())
    }

    /// Pause the stream.
    pub fn pause(&self) -> AudioResult<()> {
        *self.state.write() = StreamState::Paused;
        Ok(())
    }

    /// Write audio data to the stream.
    pub fn write(&self, data: &[u8]) -> AudioResult<usize> {
        if *self.state.read() == StreamState::Stopped {
            return Err(AudioError::StreamNotRunning);
        }

        let mut buffer = self.buffer.write();
        let capacity = buffer.capacity();
        let available = capacity.saturating_sub(buffer.len());
        let to_write = data.len().min(available);

        buffer.extend(&data[..to_write]);

        let frame_size = self.config.channels as usize * self.config.format.bytes_per_sample();
        *self.frames_written.write() += (to_write / frame_size) as u64;

        Ok(to_write)
    }

    /// Read audio data from the stream buffer.
    pub fn read(&self, data: &mut [u8]) -> AudioResult<usize> {
        if *self.state.read() != StreamState::Playing {
            // Return silence if not playing
            data.fill(0);
            return Ok(data.len());
        }

        let mut buffer = self.buffer.write();
        let to_read = data.len().min(buffer.len());

        for (i, byte) in buffer.drain(..to_read).enumerate() {
            data[i] = byte;
        }

        // Fill remaining with silence
        data[to_read..].fill(0);

        let frame_size = self.config.channels as usize * self.config.format.bytes_per_sample();
        *self.frames_read.write() += (to_read / frame_size) as u64;

        if to_read < data.len() && *self.state.read() == StreamState::Playing {
            log::trace!("Audio buffer underrun");
        }

        Ok(data.len())
    }

    /// Get the number of frames available in the buffer.
    pub fn available_frames(&self) -> usize {
        let buffer = self.buffer.read();
        let frame_size = self.config.channels as usize * self.config.format.bytes_per_sample();
        buffer.len() / frame_size
    }

    /// Get the latency in frames.
    pub fn latency_frames(&self) -> u64 {
        let written = *self.frames_written.read();
        let read = *self.frames_read.read();
        written.saturating_sub(read)
    }

    /// Get the latency in milliseconds.
    pub fn latency_ms(&self) -> f64 {
        let frames = self.latency_frames() as f64;
        (frames * 1000.0) / self.config.sample_rate as f64
    }

    /// Flush the buffer.
    pub fn flush(&self) {
        self.buffer.write().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_creation() {
        let stream = AudioStream::with_defaults();
        assert_eq!(stream.state(), StreamState::Stopped);
    }

    #[test]
    fn test_stream_write_read() {
        let stream = AudioStream::with_defaults();
        stream.start().unwrap();

        let data = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let written = stream.write(&data).unwrap();
        assert_eq!(written, 8);

        let mut out = [0u8; 8];
        stream.read(&mut out).unwrap();
        assert_eq!(out, data);
    }
}
