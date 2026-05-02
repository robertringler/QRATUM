pub mod health;
pub mod kernel;
pub mod wasm_runtime;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub id: u64,
    pub timestamp: String,
    pub message: String,
    pub level: String,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub cpu_percent: f32,
    pub memory_used_mb: f32,
    pub memory_total_mb: f32,
    pub uptime_seconds: u64,
    pub backend_version: String,
    pub timestamp: String,
}
