//! Configuration types for the deterministic runtime

use serde::{Deserialize, Serialize};

/// Runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Scheduler configuration
    pub scheduler: SchedulerConfig,
    /// Memory configuration
    pub memory: MemoryConfig,
    /// Timing configuration
    pub timing: TimingConfig,
    /// Deterministic seed
    pub seed: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            scheduler: SchedulerConfig::default(),
            memory: MemoryConfig::default(),
            timing: TimingConfig::default(),
            seed: 20251206,
        }
    }
}

/// Scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Scheduler type
    pub scheduler_type: SchedulerType,
    /// Maximum concurrent tasks
    pub max_concurrent_tasks: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            scheduler_type: SchedulerType::Edf,
            max_concurrent_tasks: 128,
        }
    }
}

/// Scheduler type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchedulerType {
    /// Earliest Deadline First
    Edf,
    /// Rate Monotonic
    RateMonotonic,
    /// Fixed Priority
    FixedPriority,
}

/// Memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum heap size in bytes
    pub max_heap_size: usize,
    /// Enable memory pooling
    pub enable_pooling: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_heap_size: 1024 * 1024 * 1024, // 1 GB
            enable_pooling: true,
        }
    }
}

/// Timing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingConfig {
    /// Time resolution in nanoseconds
    pub resolution_ns: u64,
    /// Enable WCET checking
    pub enable_wcet_check: bool,
}

impl Default for TimingConfig {
    fn default() -> Self {
        Self {
            resolution_ns: 1000, // 1 microsecond
            enable_wcet_check: true,
        }
    }
}
