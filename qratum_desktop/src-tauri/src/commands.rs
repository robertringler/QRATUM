use crate::backend::{health, kernel, LogEntry};
use crate::AppState;
use tauri::State;

#[tauri::command]
pub async fn get_health() -> Result<health::HealthResponse, String> {
    Ok(health::get_health())
}

#[tauri::command]
pub async fn execute_kernel(request: kernel::KernelRequest) -> Result<kernel::KernelResponse, String> {
    kernel::execute_kernel(request).await
}

#[tauri::command]
pub fn get_logs(state: State<AppState>, limit: Option<usize>) -> Vec<LogEntry> {
    let logs = state.logs.lock().unwrap();
    let limit = limit.unwrap_or(100).min(logs.len());
    logs.iter().rev().take(limit).cloned().collect()
}
