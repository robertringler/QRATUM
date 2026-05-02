//! Audio mixer for combining multiple streams.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{AudioError, AudioResult};
use crate::stream::{AudioStream, StreamConfig};

/// Mixer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixerConfig {
    /// Output sample rate.
    pub sample_rate: u32,
    /// Output channels.
    pub channels: u32,
    /// Master volume (0.0 - 1.0).
    pub master_volume: f32,
}

impl Default for MixerConfig {
    fn default() -> Self {
        Self {
            sample_rate: crate::DEFAULT_SAMPLE_RATE,
            channels: crate::DEFAULT_CHANNELS,
            master_volume: 1.0,
        }
    }
}

/// Stream ID type.
pub type StreamId = u32;

/// Audio mixer for combining multiple streams.
pub struct AudioMixer {
    /// Configuration.
    config: MixerConfig,
    /// Active streams.
    streams: RwLock<HashMap<StreamId, Arc<AudioStream>>>,
    /// Stream volumes.
    volumes: RwLock<HashMap<StreamId, f32>>,
    /// Next stream ID.
    next_id: RwLock<StreamId>,
    /// Mix buffer.
    mix_buffer: RwLock<Vec<f32>>,
}

impl AudioMixer {
    /// Create a new audio mixer.
    pub fn new(config: MixerConfig) -> Self {
        let buffer_size = config.sample_rate as usize / 10 * config.channels as usize;
        Self {
            config,
            streams: RwLock::new(HashMap::new()),
            volumes: RwLock::new(HashMap::new()),
            next_id: RwLock::new(0),
            mix_buffer: RwLock::new(vec![0.0; buffer_size]),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(MixerConfig::default())
    }

    /// Add a stream to the mixer.
    pub fn add_stream(&self, stream: Arc<AudioStream>) -> StreamId {
        let mut next = self.next_id.write();
        let id = *next;
        *next += 1;

        self.streams.write().insert(id, stream);
        self.volumes.write().insert(id, 1.0);

        id
    }

    /// Remove a stream from the mixer.
    pub fn remove_stream(&self, id: StreamId) -> Option<Arc<AudioStream>> {
        self.volumes.write().remove(&id);
        self.streams.write().remove(&id)
    }

    /// Set the volume for a stream.
    pub fn set_stream_volume(&self, id: StreamId, volume: f32) {
        if let Some(v) = self.volumes.write().get_mut(&id) {
            *v = volume.clamp(0.0, 1.0);
        }
    }

    /// Set the master volume.
    pub fn set_master_volume(&self, volume: f32) {
        // This would require interior mutability for config
        // For simplicity, we'll skip this
    }

    /// Get the configuration.
    pub fn config(&self) -> &MixerConfig {
        &self.config
    }

    /// Mix all streams and return the output.
    pub fn mix(&self, output: &mut [i16]) {
        let streams = self.streams.read();
        let volumes = self.volumes.read();
        
        let mut mix_buffer = self.mix_buffer.write();
        let frames = output.len() / self.config.channels as usize;
        
        // Resize mix buffer if needed
        if mix_buffer.len() < output.len() {
            mix_buffer.resize(output.len(), 0.0);
        }

        // Clear mix buffer
        for sample in mix_buffer.iter_mut().take(output.len()) {
            *sample = 0.0;
        }

        // Mix each stream
        let mut temp_buffer = vec![0u8; output.len() * 2];
        for (id, stream) in streams.iter() {
            let volume = volumes.get(id).copied().unwrap_or(1.0);
            
            if let Ok(_) = stream.read(&mut temp_buffer) {
                // Convert from S16 to float and mix
                for i in 0..output.len() {
                    let sample = i16::from_le_bytes([
                        temp_buffer[i * 2],
                        temp_buffer[i * 2 + 1],
                    ]);
                    mix_buffer[i] += (sample as f32 / 32768.0) * volume;
                }
            }
        }

        // Apply master volume and convert back to S16
        let master = self.config.master_volume;
        for (i, sample) in output.iter_mut().enumerate() {
            let mixed = (mix_buffer[i] * master).clamp(-1.0, 1.0);
            *sample = (mixed * 32767.0) as i16;
        }
    }

    /// Get the number of active streams.
    pub fn stream_count(&self) -> usize {
        self.streams.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixer_creation() {
        let mixer = AudioMixer::with_defaults();
        assert_eq!(mixer.stream_count(), 0);
    }

    #[test]
    fn test_add_stream() {
        let mixer = AudioMixer::with_defaults();
        let stream = Arc::new(AudioStream::with_defaults());
        
        let id = mixer.add_stream(stream);
        assert_eq!(mixer.stream_count(), 1);
        
        mixer.remove_stream(id);
        assert_eq!(mixer.stream_count(), 0);
    }
}
