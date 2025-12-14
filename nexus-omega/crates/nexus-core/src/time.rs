//! Time primitives for deterministic execution

use crate::error::{CoreError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use parking_lot::Mutex;

/// Timestamp in nanoseconds since epoch
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Timestamp(pub u64);

impl Timestamp {
    /// Create a new timestamp
    pub fn new(nanos: u64) -> Self {
        Self(nanos)
    }

    /// Get the timestamp in nanoseconds
    pub fn as_nanos(&self) -> u64 {
        self.0
    }

    /// Add duration in nanoseconds
    pub fn add_nanos(&self, nanos: u64) -> Self {
        Self(self.0 + nanos)
    }
}

/// Monotonic clock trait
pub trait MonotonicClock: Send + Sync {
    /// Get the current timestamp
    fn now(&self) -> Timestamp;

    /// Advance the clock (for simulated clocks)
    fn advance(&self, nanos: u64) -> Result<()>;
}

/// System clock using real wall time
#[derive(Debug)]
pub struct SystemClock {
    start_time: DateTime<Utc>,
}

impl SystemClock {
    /// Create a new system clock
    pub fn new() -> Self {
        Self {
            start_time: Utc::now(),
        }
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl MonotonicClock for SystemClock {
    fn now(&self) -> Timestamp {
        let elapsed = Utc::now()
            .signed_duration_since(self.start_time)
            .num_nanoseconds()
            .unwrap_or(0);
        Timestamp::new(elapsed as u64)
    }

    fn advance(&self, _nanos: u64) -> Result<()> {
        Err(CoreError::Time(
            "Cannot advance system clock".to_string(),
        ))
    }
}

/// Simulated clock for deterministic testing
#[derive(Debug, Clone)]
pub struct SimulatedClock {
    current: Arc<Mutex<Timestamp>>,
}

impl SimulatedClock {
    /// Create a new simulated clock starting at zero
    pub fn new() -> Self {
        Self {
            current: Arc::new(Mutex::new(Timestamp::new(0))),
        }
    }

    /// Create a new simulated clock starting at the given timestamp
    pub fn with_start(start: Timestamp) -> Self {
        Self {
            current: Arc::new(Mutex::new(start)),
        }
    }
}

impl Default for SimulatedClock {
    fn default() -> Self {
        Self::new()
    }
}

impl MonotonicClock for SimulatedClock {
    fn now(&self) -> Timestamp {
        *self.current.lock()
    }

    fn advance(&self, nanos: u64) -> Result<()> {
        let mut current = self.current.lock();
        *current = current.add_nanos(nanos);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp_ordering() {
        let t1 = Timestamp::new(100);
        let t2 = Timestamp::new(200);
        assert!(t1 < t2);
    }

    #[test]
    fn test_simulated_clock() {
        let clock = SimulatedClock::new();
        assert_eq!(clock.now().as_nanos(), 0);

        clock.advance(1000).unwrap();
        assert_eq!(clock.now().as_nanos(), 1000);

        clock.advance(500).unwrap();
        assert_eq!(clock.now().as_nanos(), 1500);
    }
}
