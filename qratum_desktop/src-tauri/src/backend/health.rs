use serde::{Deserialize, Serialize};
use sysinfo::{System, Disks};
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub cpu_usage: f32,
    pub memory_used: u64,
    pub memory_total: u64,
    pub disk_used: u64,
    pub disk_total: u64,
}

/// Get current CPU usage percentage
pub fn get_cpu_usage() -> Result<f32> {
    let mut sys = System::new_all();
    sys.refresh_cpu_all();
    
    // Wait a bit to get accurate CPU measurements
    std::thread::sleep(std::time::Duration::from_millis(200));
    sys.refresh_cpu_all();
    
    let cpu_usage = sys.global_cpu_usage();
    Ok(cpu_usage)
}

/// Get memory usage (used, total) in bytes
pub fn get_memory_usage() -> Result<(u64, u64)> {
    let mut sys = System::new_all();
    sys.refresh_memory();
    
    let used = sys.used_memory();
    let total = sys.total_memory();
    
    Ok((used, total))
}

/// Get disk usage (used, total) in bytes
pub fn get_disk_usage() -> Result<(u64, u64)> {
    let disks = Disks::new_with_refreshed_list();
    
    let mut total_space = 0u64;
    let mut available_space = 0u64;
    
    // Sum up all disks
    for disk in disks.list() {
        total_space += disk.total_space();
        available_space += disk.available_space();
    }
    
    let used_space = total_space.saturating_sub(available_space);
    
    Ok((used_space, total_space))
}

/// Get overall backend health status
pub fn get_backend_status() -> Result<HealthStatus> {
    let cpu_usage = get_cpu_usage()?;
    let (memory_used, memory_total) = get_memory_usage()?;
    let (disk_used, disk_total) = get_disk_usage()?;
    
    // Determine overall status based on resource usage
    let status = if cpu_usage > 90.0 || 
                   (memory_used as f64 / memory_total as f64) > 0.95 ||
                   (disk_used as f64 / disk_total as f64) > 0.95 {
        "critical".to_string()
    } else if cpu_usage > 70.0 || 
              (memory_used as f64 / memory_total as f64) > 0.80 ||
              (disk_used as f64 / disk_total as f64) > 0.85 {
        "warning".to_string()
    } else {
        "healthy".to_string()
    };
    
    Ok(HealthStatus {
        status,
        cpu_usage,
        memory_used,
        memory_total,
        disk_used,
        disk_total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_cpu_usage() {
        let result = get_cpu_usage();
        assert!(result.is_ok());
        let cpu = result.unwrap();
        assert!(cpu >= 0.0 && cpu <= 100.0);
    }

    #[test]
    fn test_get_memory_usage() {
        let result = get_memory_usage();
        assert!(result.is_ok());
        let (used, total) = result.unwrap();
        assert!(used <= total);
        assert!(total > 0);
    }

    #[test]
    fn test_get_disk_usage() {
        let result = get_disk_usage();
        assert!(result.is_ok());
        let (used, total) = result.unwrap();
        assert!(used <= total);
    }

    #[test]
    fn test_get_backend_status() {
        let result = get_backend_status();
        assert!(result.is_ok());
        let status = result.unwrap();
        assert!(["healthy", "warning", "critical"].contains(&status.status.as_str()));
    }
}
