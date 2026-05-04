//! Framebuffer management.
//!
//! Handles the display framebuffer for rendering output.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::error::{GraphicsError, GraphicsResult};

/// Supported pixel formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PixelFormat {
    /// 32-bit RGBA (8 bits per channel).
    Rgba8888,
    /// 32-bit RGBX (8 bits per channel, alpha ignored).
    Rgbx8888,
    /// 32-bit BGRA (8 bits per channel).
    Bgra8888,
    /// 16-bit RGB (5-6-5).
    Rgb565,
    /// 24-bit RGB (8 bits per channel).
    Rgb888,
}

impl PixelFormat {
    /// Get the bytes per pixel for this format.
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgba8888 | Self::Rgbx8888 | Self::Bgra8888 => 4,
            Self::Rgb888 => 3,
            Self::Rgb565 => 2,
        }
    }
}

impl Default for PixelFormat {
    fn default() -> Self {
        Self::Rgba8888
    }
}

/// Framebuffer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FramebufferConfig {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel format.
    pub format: PixelFormat,
    /// Enable double buffering.
    pub double_buffered: bool,
    /// Refresh rate in Hz.
    pub refresh_rate: u32,
}

impl Default for FramebufferConfig {
    fn default() -> Self {
        Self {
            width: crate::DEFAULT_WIDTH,
            height: crate::DEFAULT_HEIGHT,
            format: PixelFormat::Rgba8888,
            double_buffered: true,
            refresh_rate: crate::DEFAULT_REFRESH_RATE,
        }
    }
}

/// Framebuffer for display output.
pub struct Framebuffer {
    /// Configuration.
    config: FramebufferConfig,
    /// Front buffer (displayed).
    front_buffer: RwLock<Vec<u8>>,
    /// Back buffer (being rendered).
    back_buffer: RwLock<Vec<u8>>,
    /// Frame counter.
    frame_count: RwLock<u64>,
    /// Whether a frame is pending swap.
    pending_swap: RwLock<bool>,
}

impl Framebuffer {
    /// Create a new framebuffer.
    pub fn new(config: FramebufferConfig) -> GraphicsResult<Self> {
        let buffer_size = config.width as usize
            * config.height as usize
            * config.format.bytes_per_pixel();

        if buffer_size == 0 {
            return Err(GraphicsError::InvalidFramebufferConfig {
                reason: "buffer size is zero".to_string(),
            });
        }

        Ok(Self {
            config: config.clone(),
            front_buffer: RwLock::new(vec![0; buffer_size]),
            back_buffer: RwLock::new(vec![0; buffer_size]),
            frame_count: RwLock::new(0),
            pending_swap: RwLock::new(false),
        })
    }

    /// Create a framebuffer with default configuration.
    pub fn with_defaults() -> GraphicsResult<Self> {
        Self::new(FramebufferConfig::default())
    }

    /// Get the configuration.
    pub fn config(&self) -> &FramebufferConfig {
        &self.config
    }

    /// Get the width.
    pub fn width(&self) -> u32 {
        self.config.width
    }

    /// Get the height.
    pub fn height(&self) -> u32 {
        self.config.height
    }

    /// Get the pixel format.
    pub fn format(&self) -> PixelFormat {
        self.config.format
    }

    /// Get the stride (bytes per row).
    pub fn stride(&self) -> usize {
        self.config.width as usize * self.config.format.bytes_per_pixel()
    }

    /// Get the total buffer size in bytes.
    pub fn buffer_size(&self) -> usize {
        self.front_buffer.read().len()
    }

    /// Get a reference to the front buffer (read-only).
    pub fn front_buffer(&self) -> impl std::ops::Deref<Target = [u8]> + '_ {
        parking_lot::RwLockReadGuard::map(self.front_buffer.read(), |v| v.as_slice())
    }

    /// Get a mutable reference to the back buffer for rendering.
    pub fn back_buffer_mut(&self) -> impl std::ops::DerefMut<Target = [u8]> + '_ {
        parking_lot::RwLockWriteGuard::map(self.back_buffer.write(), |v| v.as_mut_slice())
    }

    /// Write a pixel to the back buffer.
    pub fn write_pixel(&self, x: u32, y: u32, color: &[u8]) -> GraphicsResult<()> {
        if x >= self.config.width || y >= self.config.height {
            return Ok(()); // Clip silently
        }

        let bpp = self.config.format.bytes_per_pixel();
        let offset = (y as usize * self.stride()) + (x as usize * bpp);
        let mut buffer = self.back_buffer.write();

        if offset + bpp <= buffer.len() && color.len() >= bpp {
            buffer[offset..offset + bpp].copy_from_slice(&color[..bpp]);
        }

        Ok(())
    }

    /// Fill the back buffer with a solid color.
    pub fn clear(&self, color: &[u8]) {
        let bpp = self.config.format.bytes_per_pixel();
        let mut buffer = self.back_buffer.write();
        
        for chunk in buffer.chunks_mut(bpp) {
            let copy_len = chunk.len().min(color.len());
            chunk[..copy_len].copy_from_slice(&color[..copy_len]);
        }
    }

    /// Swap front and back buffers.
    pub fn swap_buffers(&self) {
        if self.config.double_buffered {
            std::mem::swap(
                &mut *self.front_buffer.write(),
                &mut *self.back_buffer.write(),
            );
        }
        
        *self.frame_count.write() += 1;
        *self.pending_swap.write() = false;
    }

    /// Mark that a frame is ready to be swapped.
    pub fn present(&self) {
        *self.pending_swap.write() = true;
    }

    /// Check if a swap is pending.
    pub fn is_swap_pending(&self) -> bool {
        *self.pending_swap.read()
    }

    /// Get the frame count.
    pub fn frame_count(&self) -> u64 {
        *self.frame_count.read()
    }

    /// Copy the front buffer to a destination.
    pub fn copy_to(&self, dest: &mut [u8]) {
        let buffer = self.front_buffer.read();
        let copy_len = dest.len().min(buffer.len());
        dest[..copy_len].copy_from_slice(&buffer[..copy_len]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_framebuffer_creation() {
        let fb = Framebuffer::with_defaults().unwrap();
        assert_eq!(fb.width(), crate::DEFAULT_WIDTH);
        assert_eq!(fb.height(), crate::DEFAULT_HEIGHT);
    }

    #[test]
    fn test_pixel_write() {
        let config = FramebufferConfig {
            width: 100,
            height: 100,
            ..Default::default()
        };
        let fb = Framebuffer::new(config).unwrap();
        
        fb.write_pixel(50, 50, &[255, 0, 0, 255]).unwrap();
        
        let buffer = fb.back_buffer.read();
        let offset = 50 * fb.stride() + 50 * 4;
        assert_eq!(buffer[offset], 255);
        assert_eq!(buffer[offset + 1], 0);
    }

    #[test]
    fn test_swap_buffers() {
        let fb = Framebuffer::with_defaults().unwrap();
        
        fb.clear(&[255, 0, 0, 255]);
        fb.swap_buffers();
        
        assert_eq!(fb.frame_count(), 1);
    }
}
