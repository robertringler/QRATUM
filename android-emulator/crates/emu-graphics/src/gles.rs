//! OpenGL ES emulation state.
//!
//! Provides basic GLES state tracking (context would translate to Vulkan).

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{GraphicsError, GraphicsResult};

/// OpenGL ES version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GlesVersion {
    /// OpenGL ES 2.0
    Es20,
    /// OpenGL ES 3.0
    Es30,
    /// OpenGL ES 3.1
    Es31,
    /// OpenGL ES 3.2
    Es32,
}

impl Default for GlesVersion {
    fn default() -> Self {
        Self::Es32
    }
}

/// GL object handle.
pub type GlHandle = u32;

/// GL state for a context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlesState {
    /// Currently bound array buffer.
    pub array_buffer: GlHandle,
    /// Currently bound element array buffer.
    pub element_array_buffer: GlHandle,
    /// Currently bound texture (per unit).
    pub bound_textures: HashMap<u32, GlHandle>,
    /// Currently bound framebuffer.
    pub framebuffer: GlHandle,
    /// Currently bound renderbuffer.
    pub renderbuffer: GlHandle,
    /// Currently bound shader program.
    pub program: GlHandle,
    /// Viewport.
    pub viewport: [i32; 4],
    /// Scissor rect.
    pub scissor: [i32; 4],
    /// Clear color.
    pub clear_color: [f32; 4],
    /// Clear depth.
    pub clear_depth: f32,
    /// Clear stencil.
    pub clear_stencil: i32,
    /// Depth test enabled.
    pub depth_test: bool,
    /// Stencil test enabled.
    pub stencil_test: bool,
    /// Blend enabled.
    pub blend: bool,
    /// Cull face enabled.
    pub cull_face: bool,
    /// Scissor test enabled.
    pub scissor_test: bool,
    /// Active texture unit.
    pub active_texture: u32,
    /// Depth mask.
    pub depth_mask: bool,
    /// Color mask.
    pub color_mask: [bool; 4],
    /// Front face winding.
    pub front_face: u32,
    /// Cull face mode.
    pub cull_face_mode: u32,
    /// Depth function.
    pub depth_func: u32,
    /// Blend source factor.
    pub blend_src: u32,
    /// Blend destination factor.
    pub blend_dst: u32,
}

impl Default for GlesState {
    fn default() -> Self {
        Self {
            array_buffer: 0,
            element_array_buffer: 0,
            bound_textures: HashMap::new(),
            framebuffer: 0,
            renderbuffer: 0,
            program: 0,
            viewport: [0, 0, 0, 0],
            scissor: [0, 0, 0, 0],
            clear_color: [0.0, 0.0, 0.0, 0.0],
            clear_depth: 1.0,
            clear_stencil: 0,
            depth_test: false,
            stencil_test: false,
            blend: false,
            cull_face: false,
            scissor_test: false,
            active_texture: 0,
            depth_mask: true,
            color_mask: [true, true, true, true],
            front_face: 0x0901, // GL_CCW
            cull_face_mode: 0x0405, // GL_BACK
            depth_func: 0x0201, // GL_LESS
            blend_src: 0x0001, // GL_ONE
            blend_dst: 0x0000, // GL_ZERO
        }
    }
}

/// OpenGL ES context.
pub struct GlesContext {
    /// GLES version.
    version: GlesVersion,
    /// GL state.
    state: RwLock<GlesState>,
    /// Next handle to allocate.
    next_handle: RwLock<GlHandle>,
    /// Texture storage (handle -> data).
    textures: RwLock<HashMap<GlHandle, TextureObject>>,
    /// Buffer storage.
    buffers: RwLock<HashMap<GlHandle, BufferObject>>,
    /// Shader storage.
    shaders: RwLock<HashMap<GlHandle, ShaderObject>>,
    /// Program storage.
    programs: RwLock<HashMap<GlHandle, ProgramObject>>,
}

#[derive(Debug, Clone)]
struct TextureObject {
    width: u32,
    height: u32,
    format: u32,
    data: Vec<u8>,
}

#[derive(Debug, Clone)]
struct BufferObject {
    target: u32,
    usage: u32,
    data: Vec<u8>,
}

#[derive(Debug, Clone)]
struct ShaderObject {
    shader_type: u32,
    source: String,
    compiled: bool,
}

#[derive(Debug, Clone)]
struct ProgramObject {
    vertex_shader: Option<GlHandle>,
    fragment_shader: Option<GlHandle>,
    linked: bool,
}

impl GlesContext {
    /// Create a new GLES context.
    pub fn new(version: GlesVersion) -> Self {
        Self {
            version,
            state: RwLock::new(GlesState::default()),
            next_handle: RwLock::new(1),
            textures: RwLock::new(HashMap::new()),
            buffers: RwLock::new(HashMap::new()),
            shaders: RwLock::new(HashMap::new()),
            programs: RwLock::new(HashMap::new()),
        }
    }

    /// Create with default version.
    pub fn with_defaults() -> Self {
        Self::new(GlesVersion::default())
    }

    /// Get the GLES version.
    pub fn version(&self) -> GlesVersion {
        self.version
    }

    /// Allocate a new handle.
    fn alloc_handle(&self) -> GlHandle {
        let mut next = self.next_handle.write();
        let handle = *next;
        *next += 1;
        handle
    }

    /// Get the current state (read-only).
    pub fn state(&self) -> GlesState {
        self.state.read().clone()
    }

    // GL functions

    /// glGenTextures
    pub fn gen_textures(&self, n: i32) -> Vec<GlHandle> {
        (0..n).map(|_| self.alloc_handle()).collect()
    }

    /// glBindTexture
    pub fn bind_texture(&self, _target: u32, texture: GlHandle) {
        let mut state = self.state.write();
        let active = state.active_texture;
        state.bound_textures.insert(active, texture);
    }

    /// glTexImage2D
    pub fn tex_image_2d(
        &self,
        target: u32,
        level: i32,
        internal_format: i32,
        width: i32,
        height: i32,
        format: u32,
        type_: u32,
        data: Option<&[u8]>,
    ) -> GraphicsResult<()> {
        let state = self.state.read();
        let texture = state.bound_textures.get(&state.active_texture).copied().unwrap_or(0);
        
        if texture == 0 {
            return Err(GraphicsError::InvalidOperation {
                reason: "no texture bound".to_string(),
            });
        }

        let mut textures = self.textures.write();
        let tex_obj = textures.entry(texture).or_insert_with(|| TextureObject {
            width: 0,
            height: 0,
            format: 0,
            data: Vec::new(),
        });

        tex_obj.width = width as u32;
        tex_obj.height = height as u32;
        tex_obj.format = format;
        
        if let Some(data) = data {
            tex_obj.data = data.to_vec();
        } else {
            let size = width as usize * height as usize * 4;
            tex_obj.data = vec![0; size];
        }

        Ok(())
    }

    /// glGenBuffers
    pub fn gen_buffers(&self, n: i32) -> Vec<GlHandle> {
        (0..n).map(|_| self.alloc_handle()).collect()
    }

    /// glBindBuffer
    pub fn bind_buffer(&self, target: u32, buffer: GlHandle) {
        let mut state = self.state.write();
        match target {
            0x8892 => state.array_buffer = buffer, // GL_ARRAY_BUFFER
            0x8893 => state.element_array_buffer = buffer, // GL_ELEMENT_ARRAY_BUFFER
            _ => {}
        }
    }

    /// glBufferData
    pub fn buffer_data(&self, target: u32, size: isize, data: Option<&[u8]>, usage: u32) -> GraphicsResult<()> {
        let state = self.state.read();
        let buffer = match target {
            0x8892 => state.array_buffer,
            0x8893 => state.element_array_buffer,
            _ => return Err(GraphicsError::InvalidOperation {
                reason: "invalid buffer target".to_string(),
            }),
        };

        if buffer == 0 {
            return Err(GraphicsError::InvalidOperation {
                reason: "no buffer bound".to_string(),
            });
        }

        let mut buffers = self.buffers.write();
        let buf_obj = buffers.entry(buffer).or_insert_with(|| BufferObject {
            target,
            usage,
            data: Vec::new(),
        });

        buf_obj.usage = usage;
        if let Some(data) = data {
            buf_obj.data = data.to_vec();
        } else {
            buf_obj.data = vec![0; size as usize];
        }

        Ok(())
    }

    /// glCreateShader
    pub fn create_shader(&self, shader_type: u32) -> GlHandle {
        let handle = self.alloc_handle();
        self.shaders.write().insert(handle, ShaderObject {
            shader_type,
            source: String::new(),
            compiled: false,
        });
        handle
    }

    /// glShaderSource
    pub fn shader_source(&self, shader: GlHandle, source: &str) -> GraphicsResult<()> {
        if let Some(s) = self.shaders.write().get_mut(&shader) {
            s.source = source.to_string();
            Ok(())
        } else {
            Err(GraphicsError::InvalidOperation {
                reason: "invalid shader".to_string(),
            })
        }
    }

    /// glCompileShader
    pub fn compile_shader(&self, shader: GlHandle) -> GraphicsResult<()> {
        if let Some(s) = self.shaders.write().get_mut(&shader) {
            // In a real implementation, this would compile to SPIR-V
            s.compiled = true;
            Ok(())
        } else {
            Err(GraphicsError::InvalidOperation {
                reason: "invalid shader".to_string(),
            })
        }
    }

    /// glCreateProgram
    pub fn create_program(&self) -> GlHandle {
        let handle = self.alloc_handle();
        self.programs.write().insert(handle, ProgramObject {
            vertex_shader: None,
            fragment_shader: None,
            linked: false,
        });
        handle
    }

    /// glUseProgram
    pub fn use_program(&self, program: GlHandle) {
        self.state.write().program = program;
    }

    /// glViewport
    pub fn viewport(&self, x: i32, y: i32, width: i32, height: i32) {
        self.state.write().viewport = [x, y, width, height];
    }

    /// glClearColor
    pub fn clear_color(&self, r: f32, g: f32, b: f32, a: f32) {
        self.state.write().clear_color = [r, g, b, a];
    }

    /// glClear
    pub fn clear(&self, mask: u32) {
        // In a real implementation, this would issue draw commands
        log::trace!("glClear(mask={:#x})", mask);
    }

    /// glEnable
    pub fn enable(&self, cap: u32) {
        let mut state = self.state.write();
        match cap {
            0x0B71 => state.depth_test = true,    // GL_DEPTH_TEST
            0x0B90 => state.stencil_test = true,  // GL_STENCIL_TEST
            0x0BE2 => state.blend = true,         // GL_BLEND
            0x0B44 => state.cull_face = true,     // GL_CULL_FACE
            0x0C11 => state.scissor_test = true,  // GL_SCISSOR_TEST
            _ => {}
        }
    }

    /// glDisable
    pub fn disable(&self, cap: u32) {
        let mut state = self.state.write();
        match cap {
            0x0B71 => state.depth_test = false,
            0x0B90 => state.stencil_test = false,
            0x0BE2 => state.blend = false,
            0x0B44 => state.cull_face = false,
            0x0C11 => state.scissor_test = false,
            _ => {}
        }
    }

    /// glDrawArrays
    pub fn draw_arrays(&self, mode: u32, first: i32, count: i32) {
        log::trace!("glDrawArrays(mode={}, first={}, count={})", mode, first, count);
        // In a real implementation, this would issue Vulkan draw commands
    }

    /// glDrawElements
    pub fn draw_elements(&self, mode: u32, count: i32, type_: u32, indices: u64) {
        log::trace!("glDrawElements(mode={}, count={}, type={}, indices={})", mode, count, type_, indices);
        // In a real implementation, this would issue Vulkan draw commands
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let ctx = GlesContext::with_defaults();
        assert_eq!(ctx.version(), GlesVersion::Es32);
    }

    #[test]
    fn test_buffer_operations() {
        let ctx = GlesContext::with_defaults();
        
        let buffers = ctx.gen_buffers(1);
        assert_eq!(buffers.len(), 1);
        
        ctx.bind_buffer(0x8892, buffers[0]); // GL_ARRAY_BUFFER
        
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let bytes: &[u8] = bytemuck_pod_slice(&data);
        ctx.buffer_data(0x8892, bytes.len() as isize, Some(bytes), 0x88E4).unwrap();
    }

    #[test]
    fn test_state_changes() {
        let ctx = GlesContext::with_defaults();
        
        ctx.enable(0x0B71); // GL_DEPTH_TEST
        assert!(ctx.state().depth_test);
        
        ctx.disable(0x0B71);
        assert!(!ctx.state().depth_test);
    }

    fn bytemuck_pod_slice(data: &[f32]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>(),
            )
        }
    }
}
