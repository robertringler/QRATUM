//! NEXUS-Ω Deterministic Runtime Core
//!
//! Provides deterministic runtime primitives for quantum-inspired simulation.

pub mod config;
pub mod determinism;
pub mod error;
pub mod executor;
pub mod scheduler;
pub mod time;

pub use config::RuntimeConfig;
pub use determinism::{DeterministicHash, DeterministicRng};
pub use error::{CoreError, Result};
pub use executor::Executor;
pub use scheduler::{Priority, Scheduler};
pub use time::{MonotonicClock, SimulatedClock, SystemClock, Timestamp};
