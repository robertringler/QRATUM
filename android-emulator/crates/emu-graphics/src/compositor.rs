//! Display compositor (SurfaceFlinger-like).
//!
//! Composes multiple layers into the final display output.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::error::{GraphicsError, GraphicsResult};
use crate::framebuffer::{Framebuffer, PixelFormat};

/// Layer blend mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlendMode {
    /// No blending (opaque).
    None,
    /// Standard alpha blending.
    Alpha,
    /// Premultiplied alpha.
    PremultipliedAlpha,
    /// Additive blending.
    Additive,
}

impl Default for BlendMode {
    fn default() -> Self {
        Self::Alpha
    }
}

/// A compositing layer.
#[derive(Debug, Clone)]
pub struct Layer {
    /// Layer ID.
    pub id: u32,
    /// Z-order (higher = on top).
    pub z_order: i32,
    /// X position.
    pub x: i32,
    /// Y position.
    pub y: i32,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Alpha (0.0 - 1.0).
    pub alpha: f32,
    /// Blend mode.
    pub blend_mode: BlendMode,
    /// Whether the layer is visible.
    pub visible: bool,
    /// Buffer data (RGBA8888).
    pub buffer: Vec<u8>,
}

impl Layer {
    /// Create a new layer.
    pub fn new(id: u32, width: u32, height: u32) -> Self {
        let buffer_size = width as usize * height as usize * 4;
        Self {
            id,
            z_order: 0,
            x: 0,
            y: 0,
            width,
            height,
            alpha: 1.0,
            blend_mode: BlendMode::Alpha,
            visible: true,
            buffer: vec![0; buffer_size],
        }
    }

    /// Set the layer position.
    pub fn set_position(&mut self, x: i32, y: i32) {
        self.x = x;
        self.y = y;
    }

    /// Set the layer alpha.
    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha.clamp(0.0, 1.0);
    }

    /// Set the buffer contents.
    pub fn set_buffer(&mut self, data: &[u8]) {
        let expected_size = self.width as usize * self.height as usize * 4;
        if data.len() == expected_size {
            self.buffer.copy_from_slice(data);
        }
    }

    /// Fill the layer with a solid color.
    pub fn fill(&mut self, r: u8, g: u8, b: u8, a: u8) {
        for chunk in self.buffer.chunks_mut(4) {
            chunk[0] = r;
            chunk[1] = g;
            chunk[2] = b;
            chunk[3] = a;
        }
    }

    /// Get a pixel from the layer.
    fn get_pixel(&self, x: u32, y: u32) -> [u8; 4] {
        if x >= self.width || y >= self.height {
            return [0, 0, 0, 0];
        }
        let offset = (y as usize * self.width as usize + x as usize) * 4;
        if offset + 4 <= self.buffer.len() {
            [
                self.buffer[offset],
                self.buffer[offset + 1],
                self.buffer[offset + 2],
                self.buffer[offset + 3],
            ]
        } else {
            [0, 0, 0, 0]
        }
    }
}

/// Compositor configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositorConfig {
    /// Display width.
    pub width: u32,
    /// Display height.
    pub height: u32,
    /// Background color (RGBA).
    pub background_color: [u8; 4],
}

impl Default for CompositorConfig {
    fn default() -> Self {
        Self {
            width: crate::DEFAULT_WIDTH,
            height: crate::DEFAULT_HEIGHT,
            background_color: [0, 0, 0, 255], // Black
        }
    }
}

/// Display compositor.
pub struct Compositor {
    /// Configuration.
    config: CompositorConfig,
    /// Layers by ID, sorted by z-order.
    layers: RwLock<BTreeMap<u32, Layer>>,
    /// Next layer ID.
    next_layer_id: RwLock<u32>,
    /// Output framebuffer.
    output: RwLock<Vec<u8>>,
}

impl Compositor {
    /// Create a new compositor.
    pub fn new(config: CompositorConfig) -> Self {
        let buffer_size = config.width as usize * config.height as usize * 4;
        Self {
            config,
            layers: RwLock::new(BTreeMap::new()),
            next_layer_id: RwLock::new(0),
            output: RwLock::new(vec![0; buffer_size]),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(CompositorConfig::default())
    }

    /// Create a new layer.
    pub fn create_layer(&self, width: u32, height: u32) -> u32 {
        let mut next_id = self.next_layer_id.write();
        let id = *next_id;
        *next_id += 1;

        let layer = Layer::new(id, width, height);
        self.layers.write().insert(id, layer);

        id
    }

    /// Get a layer by ID.
    pub fn get_layer(&self, id: u32) -> Option<Layer> {
        self.layers.read().get(&id).cloned()
    }

    /// Update a layer.
    pub fn update_layer<F>(&self, id: u32, f: F) -> GraphicsResult<()>
    where
        F: FnOnce(&mut Layer),
    {
        if let Some(layer) = self.layers.write().get_mut(&id) {
            f(layer);
            Ok(())
        } else {
            Err(GraphicsError::InvalidOperation {
                reason: format!("layer {} not found", id),
            })
        }
    }

    /// Remove a layer.
    pub fn remove_layer(&self, id: u32) -> Option<Layer> {
        self.layers.write().remove(&id)
    }

    /// Compose all layers into the output buffer.
    pub fn compose(&self) {
        let layers = self.layers.read();
        let mut output = self.output.write();
        let width = self.config.width;
        let height = self.config.height;

        // Clear with background color
        for chunk in output.chunks_mut(4) {
            chunk.copy_from_slice(&self.config.background_color);
        }

        // Sort layers by z-order
        let mut sorted_layers: Vec<&Layer> = layers.values().filter(|l| l.visible).collect();
        sorted_layers.sort_by_key(|l| l.z_order);

        // Compose each layer
        for layer in sorted_layers {
            self.blend_layer(&mut output, layer, width, height);
        }
    }

    fn blend_layer(&self, output: &mut [u8], layer: &Layer, width: u32, height: u32) {
        let layer_alpha = (layer.alpha * 255.0) as u8;

        for ly in 0..layer.height {
            let dy = layer.y + ly as i32;
            if dy < 0 || dy >= height as i32 {
                continue;
            }

            for lx in 0..layer.width {
                let dx = layer.x + lx as i32;
                if dx < 0 || dx >= width as i32 {
                    continue;
                }

                let src = layer.get_pixel(lx, ly);
                let dst_offset = (dy as usize * width as usize + dx as usize) * 4;

                if dst_offset + 4 > output.len() {
                    continue;
                }

                let src_alpha = ((src[3] as u32 * layer_alpha as u32) / 255) as u8;

                match layer.blend_mode {
                    BlendMode::None => {
                        output[dst_offset] = src[0];
                        output[dst_offset + 1] = src[1];
                        output[dst_offset + 2] = src[2];
                        output[dst_offset + 3] = 255;
                    }
                    BlendMode::Alpha | BlendMode::PremultipliedAlpha => {
                        let dst_r = output[dst_offset];
                        let dst_g = output[dst_offset + 1];
                        let dst_b = output[dst_offset + 2];
                        let dst_a = output[dst_offset + 3];

                        let inv_alpha = 255 - src_alpha;

                        output[dst_offset] = ((src[0] as u32 * src_alpha as u32
                            + dst_r as u32 * inv_alpha as u32) / 255) as u8;
                        output[dst_offset + 1] = ((src[1] as u32 * src_alpha as u32
                            + dst_g as u32 * inv_alpha as u32) / 255) as u8;
                        output[dst_offset + 2] = ((src[2] as u32 * src_alpha as u32
                            + dst_b as u32 * inv_alpha as u32) / 255) as u8;
                        output[dst_offset + 3] = src_alpha.max(dst_a);
                    }
                    BlendMode::Additive => {
                        output[dst_offset] = output[dst_offset].saturating_add(
                            ((src[0] as u32 * src_alpha as u32) / 255) as u8
                        );
                        output[dst_offset + 1] = output[dst_offset + 1].saturating_add(
                            ((src[1] as u32 * src_alpha as u32) / 255) as u8
                        );
                        output[dst_offset + 2] = output[dst_offset + 2].saturating_add(
                            ((src[2] as u32 * src_alpha as u32) / 255) as u8
                        );
                    }
                }
            }
        }
    }

    /// Get the composed output.
    pub fn output(&self) -> impl std::ops::Deref<Target = [u8]> + '_ {
        parking_lot::RwLockReadGuard::map(self.output.read(), |v| v.as_slice())
    }

    /// Copy the output to a framebuffer.
    pub fn copy_to_framebuffer(&self, fb: &Framebuffer) {
        let output = self.output.read();
        let mut fb_buffer = fb.back_buffer_mut();
        let copy_len = fb_buffer.len().min(output.len());
        fb_buffer[..copy_len].copy_from_slice(&output[..copy_len]);
    }

    /// Get the layer count.
    pub fn layer_count(&self) -> usize {
        self.layers.read().len()
    }

    /// Get the configuration.
    pub fn config(&self) -> &CompositorConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compositor_creation() {
        let compositor = Compositor::with_defaults();
        assert_eq!(compositor.layer_count(), 0);
    }

    #[test]
    fn test_layer_creation() {
        let compositor = Compositor::with_defaults();
        let id = compositor.create_layer(100, 100);
        assert_eq!(compositor.layer_count(), 1);
        
        let layer = compositor.get_layer(id).unwrap();
        assert_eq!(layer.width, 100);
        assert_eq!(layer.height, 100);
    }

    #[test]
    fn test_composition() {
        let config = CompositorConfig {
            width: 100,
            height: 100,
            background_color: [0, 0, 0, 255],
        };
        let compositor = Compositor::new(config);
        
        // Create a red layer
        let id = compositor.create_layer(50, 50);
        compositor.update_layer(id, |layer| {
            layer.fill(255, 0, 0, 255);
            layer.set_position(25, 25);
        }).unwrap();
        
        compositor.compose();
        
        let output = compositor.output();
        // Check that the center pixel is red
        let center_offset = (50 * 100 + 50) * 4;
        assert_eq!(output[center_offset], 255); // R
        assert_eq!(output[center_offset + 1], 0); // G
        assert_eq!(output[center_offset + 2], 0); // B
    }
}
