//! Android Runtime emulation.
//!
//! Coordinates the various runtime components.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use emu_cpu::Arm64Cpu;
use emu_memory::Mmu;

use crate::binder::{BinderDriver, ServiceManager};
use crate::bionic::BionicEmulator;
use crate::error::{RuntimeError, RuntimeResult};
use crate::syscall::SyscallEmulator;

/// Runtime configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Heap size in bytes.
    pub heap_size: usize,
    /// Stack size in bytes.
    pub stack_size: usize,
    /// Enable syscall tracing.
    pub syscall_trace: bool,
    /// Enable Binder debugging.
    pub binder_debug: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            heap_size: 256 * 1024 * 1024, // 256 MB
            stack_size: 8 * 1024 * 1024,   // 8 MB
            syscall_trace: false,
            binder_debug: false,
        }
    }
}

/// Android Runtime state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeState {
    /// Runtime is uninitialized.
    Uninitialized,
    /// Runtime is initializing.
    Initializing,
    /// Runtime is ready.
    Ready,
    /// Runtime is running.
    Running,
    /// Runtime has stopped.
    Stopped,
}

/// Android Runtime implementation.
pub struct AndroidRuntime {
    /// Configuration.
    config: RuntimeConfig,
    /// Current state.
    state: RwLock<RuntimeState>,
    /// Binder driver.
    binder: Arc<BinderDriver>,
    /// Bionic emulator.
    bionic: BionicEmulator,
    /// Syscall emulator.
    syscall: RwLock<SyscallEmulator>,
}

impl AndroidRuntime {
    /// Create a new Android Runtime.
    pub fn new(config: RuntimeConfig) -> Self {
        let binder = Arc::new(BinderDriver::new());
        
        // Register service manager
        let service_manager = Arc::new(ServiceManager::new(binder.clone()));
        binder.register_service("android.os.IServiceManager", service_manager).ok();
        
        Self {
            bionic: BionicEmulator::new(0x2000_0000, config.heap_size as u64),
            config,
            state: RwLock::new(RuntimeState::Uninitialized),
            binder,
            syscall: RwLock::new(SyscallEmulator::new()),
        }
    }

    /// Initialize the runtime.
    pub fn initialize(&self, mmu: &Mmu) -> RuntimeResult<()> {
        *self.state.write() = RuntimeState::Initializing;
        
        log::info!("Initializing Android Runtime...");
        
        // Initialize system services
        self.init_system_services()?;
        
        *self.state.write() = RuntimeState::Ready;
        log::info!("Android Runtime initialized");
        
        Ok(())
    }

    /// Get the current state.
    pub fn state(&self) -> RuntimeState {
        *self.state.read()
    }

    /// Get the Binder driver.
    pub fn binder(&self) -> &Arc<BinderDriver> {
        &self.binder
    }

    /// Get the Bionic emulator.
    pub fn bionic(&self) -> &BionicEmulator {
        &self.bionic
    }

    /// Handle a syscall from the CPU.
    pub fn handle_syscall(&self, cpu: &mut Arm64Cpu, mmu: &Mmu) -> RuntimeResult<()> {
        let result = self.syscall.write().handle_syscall(&mut cpu.regs, mmu);
        
        match result {
            Ok(_) => Ok(()),
            Err(errno) if errno < -1000 => {
                // Exit requested
                let exit_code = -errno - 1000;
                log::info!("Process exited with code {}", exit_code);
                *self.state.write() = RuntimeState::Stopped;
                Ok(())
            }
            Err(_) => Ok(()), // Syscall returned error, but that's normal
        }
    }

    fn init_system_services(&self) -> RuntimeResult<()> {
        // Register core system services
        // In a full implementation, this would set up:
        // - ActivityManagerService
        // - PackageManagerService
        // - WindowManagerService
        // - etc.
        
        log::debug!("System services initialized");
        Ok(())
    }

    /// Allocate memory using the heap.
    pub fn malloc(&self, size: u64) -> RuntimeResult<u64> {
        self.bionic.malloc(size)
    }

    /// Free memory.
    pub fn free(&self, ptr: u64, size: u64) {
        self.bionic.free(ptr, size);
    }

    /// Get the configuration.
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }
}

impl Default for AndroidRuntime {
    fn default() -> Self {
        Self::new(RuntimeConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_creation() {
        let runtime = AndroidRuntime::default();
        assert_eq!(runtime.state(), RuntimeState::Uninitialized);
    }

    #[test]
    fn test_runtime_services() {
        let runtime = AndroidRuntime::default();
        
        // Service manager should be registered
        let handle = runtime.binder().get_service("android.os.IServiceManager");
        assert!(handle.is_some());
    }
}
