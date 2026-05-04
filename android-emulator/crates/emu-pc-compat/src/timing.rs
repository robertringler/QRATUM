//! Timing normalization and frame limiting.
//!
//! Provides consistent timing for research and compatibility.

use parking_lot::RwLock;
use std::time::{Duration, Instant};

use crate::error::PcCompatResult;

/// Timing normalizer for consistent execution.
pub struct TimingNormalizer {
    /// Start time of the session.
    start_time: Instant,
    /// Virtual time offset.
    virtual_offset: RwLock<Duration>,
    /// Time scale (1.0 = real-time).
    time_scale: RwLock<f64>,
    /// High-resolution timer frequency.
    frequency: u64,
}

impl TimingNormalizer {
    /// Create a new timing normalizer.
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            virtual_offset: RwLock::new(Duration::ZERO),
            time_scale: RwLock::new(1.0),
            frequency: 10_000_000, // 100ns resolution (Windows default)
        }
    }

    /// Get the virtual elapsed time.
    pub fn elapsed(&self) -> Duration {
        let real_elapsed = self.start_time.elapsed();
        let scale = *self.time_scale.read();
        let offset = *self.virtual_offset.read();
        
        Duration::from_secs_f64(real_elapsed.as_secs_f64() * scale) + offset
    }

    /// Get the performance counter value (Windows-style).
    pub fn query_performance_counter(&self) -> u64 {
        let elapsed = self.elapsed();
        (elapsed.as_nanos() as u64 * self.frequency) / 1_000_000_000
    }

    /// Get the performance counter frequency.
    pub fn query_performance_frequency(&self) -> u64 {
        self.frequency
    }

    /// Get the tick count (milliseconds).
    pub fn get_tick_count(&self) -> u32 {
        self.elapsed().as_millis() as u32
    }

    /// Get the tick count 64-bit (milliseconds).
    pub fn get_tick_count64(&self) -> u64 {
        self.elapsed().as_millis() as u64
    }

    /// Set the time scale.
    pub fn set_time_scale(&self, scale: f64) {
        *self.time_scale.write() = scale.clamp(0.1, 10.0);
    }

    /// Get the time scale.
    pub fn time_scale(&self) -> f64 {
        *self.time_scale.read()
    }

    /// Add a virtual time offset.
    pub fn add_offset(&self, offset: Duration) {
        *self.virtual_offset.write() += offset;
    }

    /// Reset the timer.
    pub fn reset(&mut self) {
        self.start_time = Instant::now();
        *self.virtual_offset.write() = Duration::ZERO;
    }
}

impl Default for TimingNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Frame limiter for consistent frame pacing.
pub struct FrameLimiter {
    /// Target frame rate.
    target_fps: f64,
    /// Target frame time.
    target_frame_time: Duration,
    /// Last frame time.
    last_frame_time: RwLock<Instant>,
    /// Frame count.
    frame_count: RwLock<u64>,
    /// Enable frame limiting.
    enabled: RwLock<bool>,
    /// Statistics.
    stats: RwLock<FrameStats>,
}

/// Frame statistics.
#[derive(Debug, Clone, Default)]
pub struct FrameStats {
    /// Total frames rendered.
    pub total_frames: u64,
    /// Average frame time (microseconds).
    pub avg_frame_time_us: u64,
    /// Minimum frame time (microseconds).
    pub min_frame_time_us: u64,
    /// Maximum frame time (microseconds).
    pub max_frame_time_us: u64,
    /// Frame times for the last 60 frames.
    frame_times: Vec<u64>,
}

impl FrameLimiter {
    /// Create a new frame limiter.
    pub fn new(target_fps: f64) -> Self {
        Self {
            target_fps,
            target_frame_time: Duration::from_secs_f64(1.0 / target_fps),
            last_frame_time: RwLock::new(Instant::now()),
            frame_count: RwLock::new(0),
            enabled: RwLock::new(true),
            stats: RwLock::new(FrameStats::default()),
        }
    }

    /// Create with 60 FPS target.
    pub fn with_60fps() -> Self {
        Self::new(60.0)
    }

    /// Set the target FPS.
    pub fn set_target_fps(&mut self, fps: f64) {
        self.target_fps = fps.clamp(1.0, 1000.0);
        self.target_frame_time = Duration::from_secs_f64(1.0 / self.target_fps);
    }

    /// Get the target FPS.
    pub fn target_fps(&self) -> f64 {
        self.target_fps
    }

    /// Enable or disable frame limiting.
    pub fn set_enabled(&self, enabled: bool) {
        *self.enabled.write() = enabled;
    }

    /// Check if frame limiting is enabled.
    pub fn is_enabled(&self) -> bool {
        *self.enabled.read()
    }

    /// Wait for the next frame (call at end of frame).
    pub fn wait_for_next_frame(&self) {
        if !*self.enabled.read() {
            self.record_frame();
            return;
        }

        let last = *self.last_frame_time.read();
        let elapsed = last.elapsed();
        
        if elapsed < self.target_frame_time {
            let sleep_time = self.target_frame_time - elapsed;
            std::thread::sleep(sleep_time);
        }
        
        self.record_frame();
    }

    /// Begin a frame (call at start of frame).
    pub fn begin_frame(&self) {
        *self.last_frame_time.write() = Instant::now();
    }

    /// End a frame and get the frame time.
    pub fn end_frame(&self) -> Duration {
        let last = *self.last_frame_time.read();
        let frame_time = last.elapsed();
        
        self.record_frame();
        
        if *self.enabled.read() {
            if frame_time < self.target_frame_time {
                let sleep_time = self.target_frame_time - frame_time;
                std::thread::sleep(sleep_time);
            }
        }
        
        frame_time
    }

    fn record_frame(&self) {
        let last = *self.last_frame_time.read();
        let frame_time_us = last.elapsed().as_micros() as u64;
        
        *self.frame_count.write() += 1;
        *self.last_frame_time.write() = Instant::now();
        
        let mut stats = self.stats.write();
        stats.total_frames += 1;
        stats.frame_times.push(frame_time_us);
        
        // Keep only last 60 frames
        if stats.frame_times.len() > 60 {
            stats.frame_times.remove(0);
        }
        
        // Update statistics
        if !stats.frame_times.is_empty() {
            let sum: u64 = stats.frame_times.iter().sum();
            stats.avg_frame_time_us = sum / stats.frame_times.len() as u64;
            stats.min_frame_time_us = *stats.frame_times.iter().min().unwrap_or(&0);
            stats.max_frame_time_us = *stats.frame_times.iter().max().unwrap_or(&0);
        }
    }

    /// Get the frame count.
    pub fn frame_count(&self) -> u64 {
        *self.frame_count.read()
    }

    /// Get frame statistics.
    pub fn get_statistics(&self) -> FrameStats {
        self.stats.read().clone()
    }

    /// Get the current FPS (estimated).
    pub fn current_fps(&self) -> f64 {
        let stats = self.stats.read();
        if stats.avg_frame_time_us > 0 {
            1_000_000.0 / stats.avg_frame_time_us as f64
        } else {
            0.0
        }
    }
}

impl Default for FrameLimiter {
    fn default() -> Self {
        Self::with_60fps()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_normalizer() {
        let normalizer = TimingNormalizer::new();
        
        let qpc = normalizer.query_performance_counter();
        assert!(qpc >= 0);
        
        let freq = normalizer.query_performance_frequency();
        assert_eq!(freq, 10_000_000);
    }

    #[test]
    fn test_time_scale() {
        let normalizer = TimingNormalizer::new();
        
        normalizer.set_time_scale(2.0);
        assert_eq!(normalizer.time_scale(), 2.0);
        
        // Scale should be clamped
        normalizer.set_time_scale(100.0);
        assert_eq!(normalizer.time_scale(), 10.0);
    }

    #[test]
    fn test_frame_limiter() {
        let limiter = FrameLimiter::with_60fps();
        
        assert_eq!(limiter.target_fps(), 60.0);
        assert!(limiter.is_enabled());
        
        limiter.begin_frame();
        // Simulate some work
        std::thread::sleep(Duration::from_millis(1));
        let frame_time = limiter.end_frame();
        
        assert!(frame_time > Duration::ZERO);
    }
}
