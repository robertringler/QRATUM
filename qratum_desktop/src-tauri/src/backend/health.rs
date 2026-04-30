use super::HealthResponse;
use std::time::{SystemTime, UNIX_EPOCH};

// Minimal health check without heavy dependencies
pub fn get_health() -> HealthResponse {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // Basic memory info (platform-specific, fallback to estimates)
    let (memory_used_mb, memory_total_mb) = get_memory_info();

    HealthResponse {
        status: "healthy".to_string(),
        cpu_percent: 0.0, // Placeholder (no sysinfo crate)
        memory_used_mb,
        memory_total_mb,
        uptime_seconds: timestamp,
        backend_version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: format_timestamp(timestamp),
    }
}

fn format_timestamp(secs: u64) -> String {
    // Simple ISO 8601 format without external dependencies
    let days = secs / 86400;
    let hours = (secs % 86400) / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;

    // Approximate date since UNIX epoch (1970-01-01)
    let year = 1970 + (days / 365);
    let day_of_year = days % 365;

    format!(
        "{}-01-01T{:02}:{:02}:{:02}Z (approx day {})",
        year, hours, minutes, seconds, day_of_year
    )
}

#[cfg(target_os = "windows")]
fn get_memory_info() -> (f32, f32) {
    // Use Windows API (minimal overhead)
    use std::mem;
    use winapi::um::sysinfoapi::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    unsafe {
        let mut mem_status: MEMORYSTATUSEX = mem::zeroed();
        mem_status.dwLength = mem::size_of::<MEMORYSTATUSEX>() as u32;
        GlobalMemoryStatusEx(&mut mem_status);

        let used = (mem_status.ullTotalPhys - mem_status.ullAvailPhys) as f32 / 1_048_576.0;
        let total = mem_status.ullTotalPhys as f32 / 1_048_576.0;
        (used, total)
    }
}

#[cfg(not(target_os = "windows"))]
fn get_memory_info() -> (f32, f32) {
    // Fallback for Linux/macOS
    (0.0, 16384.0) // Placeholder
}
