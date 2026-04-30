//! OpenSL ES emulation.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{AudioError, AudioResult};
use crate::stream::{AudioFormat, AudioStream, StreamConfig};

/// OpenSL ES object handle.
pub type SlObjectHandle = u32;

/// OpenSL ES engine.
pub struct OpenSlEngine {
    /// Created objects.
    objects: RwLock<HashMap<SlObjectHandle, SlObject>>,
    /// Next handle.
    next_handle: RwLock<SlObjectHandle>,
}

enum SlObject {
    Player(OpenSlPlayer),
    OutputMix,
    Recorder,
}

impl OpenSlEngine {
    /// Create a new OpenSL ES engine.
    pub fn new() -> AudioResult<Self> {
        Ok(Self {
            objects: RwLock::new(HashMap::new()),
            next_handle: RwLock::new(1),
        })
    }

    fn alloc_handle(&self) -> SlObjectHandle {
        let mut next = self.next_handle.write();
        let handle = *next;
        *next += 1;
        handle
    }

    /// Create an output mix.
    pub fn create_output_mix(&self) -> AudioResult<SlObjectHandle> {
        let handle = self.alloc_handle();
        self.objects.write().insert(handle, SlObject::OutputMix);
        Ok(handle)
    }

    /// Create an audio player.
    pub fn create_audio_player(&self, output_mix: SlObjectHandle, config: StreamConfig) -> AudioResult<SlObjectHandle> {
        if !self.objects.read().contains_key(&output_mix) {
            return Err(AudioError::InitializationFailed {
                reason: "invalid output mix".to_string(),
            });
        }

        let player = OpenSlPlayer::new(config)?;
        let handle = self.alloc_handle();
        self.objects.write().insert(handle, SlObject::Player(player));
        Ok(handle)
    }

    /// Get a player by handle.
    pub fn get_player(&self, handle: SlObjectHandle) -> Option<OpenSlPlayer> {
        let objects = self.objects.read();
        if let Some(SlObject::Player(player)) = objects.get(&handle) {
            Some(player.clone())
        } else {
            None
        }
    }

    /// Destroy an object.
    pub fn destroy(&self, handle: SlObjectHandle) {
        self.objects.write().remove(&handle);
    }
}

impl Default for OpenSlEngine {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

/// OpenSL ES audio player.
#[derive(Clone)]
pub struct OpenSlPlayer {
    /// Underlying audio stream.
    stream: Arc<AudioStream>,
    /// Volume (0.0 - 1.0).
    volume: Arc<RwLock<f32>>,
    /// Muted state.
    muted: Arc<RwLock<bool>>,
}

impl OpenSlPlayer {
    /// Create a new player.
    pub fn new(config: StreamConfig) -> AudioResult<Self> {
        Ok(Self {
            stream: Arc::new(AudioStream::new(config)),
            volume: Arc::new(RwLock::new(1.0)),
            muted: Arc::new(RwLock::new(false)),
        })
    }

    /// Get the underlying stream.
    pub fn stream(&self) -> &Arc<AudioStream> {
        &self.stream
    }

    /// Set the play state to playing.
    pub fn set_play_state_playing(&self) -> AudioResult<()> {
        self.stream.start()
    }

    /// Set the play state to stopped.
    pub fn set_play_state_stopped(&self) -> AudioResult<()> {
        self.stream.stop()
    }

    /// Set the play state to paused.
    pub fn set_play_state_paused(&self) -> AudioResult<()> {
        self.stream.pause()
    }

    /// Set the volume.
    pub fn set_volume(&self, volume: f32) {
        *self.volume.write() = volume.clamp(0.0, 1.0);
    }

    /// Get the volume.
    pub fn get_volume(&self) -> f32 {
        *self.volume.read()
    }

    /// Set mute state.
    pub fn set_mute(&self, muted: bool) {
        *self.muted.write() = muted;
    }

    /// Check if muted.
    pub fn is_muted(&self) -> bool {
        *self.muted.read()
    }

    /// Enqueue a buffer for playback.
    pub fn enqueue(&self, buffer: &[u8]) -> AudioResult<()> {
        self.stream.write(buffer)?;
        Ok(())
    }

    /// Get the buffer queue count.
    pub fn get_buffer_queue_count(&self) -> usize {
        self.stream.available_frames()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = OpenSlEngine::new().unwrap();
        let mix = engine.create_output_mix().unwrap();
        assert!(mix > 0);
    }

    #[test]
    fn test_player_creation() {
        let engine = OpenSlEngine::new().unwrap();
        let mix = engine.create_output_mix().unwrap();
        let player = engine.create_audio_player(mix, StreamConfig::default()).unwrap();
        assert!(player > 0);
    }

    #[test]
    fn test_player_operations() {
        let engine = OpenSlEngine::new().unwrap();
        let mix = engine.create_output_mix().unwrap();
        let handle = engine.create_audio_player(mix, StreamConfig::default()).unwrap();
        
        let player = engine.get_player(handle).unwrap();
        player.set_play_state_playing().unwrap();
        player.set_volume(0.5);
        assert_eq!(player.get_volume(), 0.5);
    }
}
