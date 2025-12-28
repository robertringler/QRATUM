use crate::backend::health::{self, HealthStatus};
use crate::backend::kernel;
use anyhow::Result;

/// Get system health status
#[tauri::command]
pub async fn get_health_status() -> Result<HealthStatus, String> {
    health::get_backend_status()
        .map_err(|e| e.to_string())
}

/// Get CPU usage
#[tauri::command]
pub async fn get_cpu_usage() -> Result<f32, String> {
    health::get_cpu_usage()
        .map_err(|e| e.to_string())
}

/// Get memory usage
#[tauri::command]
pub async fn get_memory_usage() -> Result<(u64, u64), String> {
    health::get_memory_usage()
        .map_err(|e| e.to_string())
}

/// Get disk usage
#[tauri::command]
pub async fn get_disk_usage() -> Result<(u64, u64), String> {
    health::get_disk_usage()
        .map_err(|e| e.to_string())
}

/// Get kernel status
#[tauri::command]
pub async fn get_kernel_status() -> Result<kernel::KernelStatus, String> {
    kernel::get_status()
        .map_err(|e| e.to_string())
}

/// Execute kernel computation
#[tauri::command]
pub async fn execute_computation(input: serde_json::Value) -> Result<serde_json::Value, String> {
    kernel::execute_computation(input)
        .await
        .map_err(|e| e.to_string())
}

/// Get application info
#[tauri::command]
pub async fn get_app_info() -> Result<serde_json::Value, String> {
    Ok(serde_json::json!({
        "name": "QRATUM Desktop Edition",
        "version": env!("CARGO_PKG_VERSION"),
        "description": env!("CARGO_PKG_DESCRIPTION"),
    }))
}
