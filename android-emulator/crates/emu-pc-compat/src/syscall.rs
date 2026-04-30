//! Win32 syscall translation shim.
//!
//! Provides research-oriented translation of Windows syscalls.
//! This is for compatibility research only.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use crate::error::{PcCompatError, PcCompatResult};

/// Syscall trace entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyscallTrace {
    /// Syscall number.
    pub number: u32,
    /// Syscall name (if known).
    pub name: Option<String>,
    /// Arguments.
    pub args: Vec<u64>,
    /// Return value.
    pub result: i64,
    /// Timestamp (microseconds).
    pub timestamp_us: u64,
}

/// Known Windows syscall numbers (Windows 10+).
pub mod syscall_numbers {
    /// NtCreateFile
    pub const NT_CREATE_FILE: u32 = 0x0055;
    /// NtReadFile
    pub const NT_READ_FILE: u32 = 0x0006;
    /// NtWriteFile
    pub const NT_WRITE_FILE: u32 = 0x0008;
    /// NtClose
    pub const NT_CLOSE: u32 = 0x000F;
    /// NtQueryInformationFile
    pub const NT_QUERY_INFORMATION_FILE: u32 = 0x0011;
    /// NtAllocateVirtualMemory
    pub const NT_ALLOCATE_VIRTUAL_MEMORY: u32 = 0x0018;
    /// NtFreeVirtualMemory
    pub const NT_FREE_VIRTUAL_MEMORY: u32 = 0x001E;
    /// NtQueryVirtualMemory
    pub const NT_QUERY_VIRTUAL_MEMORY: u32 = 0x0023;
    /// NtOpenProcess
    pub const NT_OPEN_PROCESS: u32 = 0x0026;
    /// NtCreateThread
    pub const NT_CREATE_THREAD: u32 = 0x004E;
    /// NtTerminateProcess
    pub const NT_TERMINATE_PROCESS: u32 = 0x002C;
}

/// Win32 syscall shim for compatibility research.
pub struct Win32SyscallShim {
    /// Trace buffer.
    traces: RwLock<VecDeque<SyscallTrace>>,
    /// Maximum trace buffer size.
    max_traces: usize,
    /// Enable tracing.
    tracing_enabled: RwLock<bool>,
    /// Timestamp counter.
    timestamp: RwLock<u64>,
}

impl Win32SyscallShim {
    /// Create a new syscall shim.
    pub fn new() -> Self {
        Self {
            traces: RwLock::new(VecDeque::new()),
            max_traces: 10000,
            tracing_enabled: RwLock::new(true),
            timestamp: RwLock::new(0),
        }
    }

    /// Enable or disable tracing.
    pub fn set_tracing(&self, enabled: bool) {
        *self.tracing_enabled.write() = enabled;
    }

    /// Clear the trace buffer.
    pub fn clear_traces(&self) {
        self.traces.write().clear();
    }

    /// Get all traces.
    pub fn get_traces(&self) -> Vec<SyscallTrace> {
        self.traces.read().iter().cloned().collect()
    }

    /// Handle a syscall.
    pub fn handle_syscall(&self, number: u32, args: &[u64]) -> PcCompatResult<i64> {
        let timestamp = {
            let mut ts = self.timestamp.write();
            *ts += 1;
            *ts
        };

        let result = self.dispatch_syscall(number, args);
        
        if *self.tracing_enabled.read() {
            let result_val = match &result {
                Ok(v) => *v,
                Err(_) => -1,
            };
            let trace = SyscallTrace {
                number,
                name: self.syscall_name(number),
                args: args.to_vec(),
                result: result_val,
                timestamp_us: timestamp,
            };
            
            let mut traces = self.traces.write();
            if traces.len() >= self.max_traces {
                traces.pop_front();
            }
            traces.push_back(trace);
        }

        result
    }

    fn dispatch_syscall(&self, number: u32, args: &[u64]) -> PcCompatResult<i64> {
        match number {
            syscall_numbers::NT_CLOSE => {
                // Stub: always succeed
                Ok(0)
            }
            syscall_numbers::NT_CREATE_FILE => {
                // Stub: return a fake handle
                Ok(0x100)
            }
            syscall_numbers::NT_READ_FILE => {
                // Stub: return 0 bytes read
                Ok(0)
            }
            syscall_numbers::NT_WRITE_FILE => {
                // Stub: return bytes "written"
                let bytes = args.get(6).copied().unwrap_or(0);
                Ok(bytes as i64)
            }
            syscall_numbers::NT_ALLOCATE_VIRTUAL_MEMORY => {
                // Stub: return success
                Ok(0)
            }
            syscall_numbers::NT_FREE_VIRTUAL_MEMORY => {
                Ok(0)
            }
            syscall_numbers::NT_TERMINATE_PROCESS => {
                Ok(0)
            }
            _ => {
                // Unknown syscall - log for research
                log::warn!("Unsupported Win32 syscall: {:#x}", number);
                Err(PcCompatError::UnsupportedSyscall { number })
            }
        }
    }

    fn syscall_name(&self, number: u32) -> Option<String> {
        match number {
            syscall_numbers::NT_CREATE_FILE => Some("NtCreateFile".to_string()),
            syscall_numbers::NT_READ_FILE => Some("NtReadFile".to_string()),
            syscall_numbers::NT_WRITE_FILE => Some("NtWriteFile".to_string()),
            syscall_numbers::NT_CLOSE => Some("NtClose".to_string()),
            syscall_numbers::NT_QUERY_INFORMATION_FILE => Some("NtQueryInformationFile".to_string()),
            syscall_numbers::NT_ALLOCATE_VIRTUAL_MEMORY => Some("NtAllocateVirtualMemory".to_string()),
            syscall_numbers::NT_FREE_VIRTUAL_MEMORY => Some("NtFreeVirtualMemory".to_string()),
            syscall_numbers::NT_QUERY_VIRTUAL_MEMORY => Some("NtQueryVirtualMemory".to_string()),
            syscall_numbers::NT_OPEN_PROCESS => Some("NtOpenProcess".to_string()),
            syscall_numbers::NT_CREATE_THREAD => Some("NtCreateThread".to_string()),
            syscall_numbers::NT_TERMINATE_PROCESS => Some("NtTerminateProcess".to_string()),
            _ => None,
        }
    }

    /// Get statistics about syscall usage.
    pub fn get_statistics(&self) -> SyscallStatistics {
        let traces = self.traces.read();
        let mut stats = SyscallStatistics::default();
        
        for trace in traces.iter() {
            stats.total_calls += 1;
            *stats.calls_by_number.entry(trace.number).or_insert(0) += 1;
        }
        
        stats
    }
}

impl Default for Win32SyscallShim {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about syscall usage.
#[derive(Debug, Default, Clone)]
pub struct SyscallStatistics {
    /// Total number of syscalls.
    pub total_calls: u64,
    /// Calls by syscall number.
    pub calls_by_number: std::collections::HashMap<u32, u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syscall_handling() {
        let shim = Win32SyscallShim::new();
        
        let result = shim.handle_syscall(syscall_numbers::NT_CLOSE, &[0x100]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tracing() {
        let shim = Win32SyscallShim::new();
        
        shim.handle_syscall(syscall_numbers::NT_CREATE_FILE, &[0, 0, 0]).ok();
        shim.handle_syscall(syscall_numbers::NT_CLOSE, &[0x100]).ok();
        
        let traces = shim.get_traces();
        assert_eq!(traces.len(), 2);
    }

    #[test]
    fn test_statistics() {
        let shim = Win32SyscallShim::new();
        
        for _ in 0..5 {
            shim.handle_syscall(syscall_numbers::NT_CLOSE, &[0]).ok();
        }
        
        let stats = shim.get_statistics();
        assert_eq!(stats.total_calls, 5);
    }
}
