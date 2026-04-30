//! Q-Substrate Configuration Module
//!
//! Runtime and memory configuration for:
//! - Desktop, micro, and embedded modes
//! - Memory limits per pod
//! - Deterministic execution settings
//! - Hardware abstraction

extern crate alloc;

use alloc::string::String;
use serde::{Deserialize, Serialize};

/// Runtime modes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeMode {
    /// Desktop mode (full features)
    Desktop,
    /// Micro mode (ESP32, RP2040)
    Micro,
    /// Embedded mode (minimal footprint)
    Embedded,
    /// WASM browser mode
    WasmBrowser,
}

impl Default for RuntimeMode {
    fn default() -> Self {
        RuntimeMode::Desktop
    }
}

/// Memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Total memory limit in MB
    pub total_limit_mb: usize,
    /// AI pod memory limit in KB
    pub ai_pod_limit_kb: usize,
    /// Quantum pod memory limit in KB
    pub quantum_pod_limit_kb: usize,
    /// DCGE pod memory limit in KB
    pub dcge_pod_limit_kb: usize,
    /// Stack size limit in bytes
    pub stack_limit: usize,
    /// Enable streaming inference to reduce memory
    pub streaming_inference: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        MemoryConfig {
            total_limit_mb: 32,
            ai_pod_limit_kb: 8192,     // 8 MB for AI
            quantum_pod_limit_kb: 64,   // 64 KB for quantum (32KB state + overhead)
            dcge_pod_limit_kb: 4,       // 4 KB for DCGE
            stack_limit: 1024,
            streaming_inference: true,
        }
    }
}

impl MemoryConfig {
    /// Create configuration for micro devices
    pub fn micro() -> Self {
        MemoryConfig {
            total_limit_mb: 4,
            ai_pod_limit_kb: 2048,    // 2 MB for streaming AI
            quantum_pod_limit_kb: 32,  // 32 KB (8 qubits)
            dcge_pod_limit_kb: 2,      // 2 KB for DCGE
            stack_limit: 512,
            streaming_inference: true,
        }
    }

    /// Create configuration for embedded devices
    pub fn embedded() -> Self {
        MemoryConfig {
            total_limit_mb: 1,
            ai_pod_limit_kb: 512,      // 512 KB
            quantum_pod_limit_kb: 16,   // 16 KB (6 qubits)
            dcge_pod_limit_kb: 1,       // 1 KB
            stack_limit: 256,
            streaming_inference: true,
        }
    }
}

/// Hardware configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Target CPU architecture
    pub cpu_arch: CpuArch,
    /// Available RAM in KB
    pub available_ram_kb: usize,
    /// Has floating point unit
    pub has_fpu: bool,
    /// Use fixed-point arithmetic
    pub use_fixed_point: bool,
    /// Number of cores
    pub num_cores: usize,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        HardwareConfig {
            cpu_arch: CpuArch::X86_64,
            available_ram_kb: 32768,  // 32 MB
            has_fpu: true,
            use_fixed_point: false,
            num_cores: 1,  // Single-threaded by default
        }
    }
}

/// CPU architectures
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CpuArch {
    X86_64,
    Arm64,
    Arm32,
    RiscV,
    Xtensa,  // ESP32
    Wasm32,
}

impl Default for CpuArch {
    fn default() -> Self {
        CpuArch::X86_64
    }
}

/// Build configuration for supremacy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildConfig {
    /// Optimization level
    pub opt_level: OptLevel,
    /// Enable LTO
    pub lto: bool,
    /// Panic mode
    pub panic_mode: PanicMode,
    /// Strip symbols
    pub strip: bool,
    /// Target compressed size in KB
    pub target_size_kb: usize,
}

impl Default for BuildConfig {
    fn default() -> Self {
        BuildConfig {
            opt_level: OptLevel::Z,
            lto: true,
            panic_mode: PanicMode::Abort,
            strip: true,
            target_size_kb: 500,
        }
    }
}

/// Optimization levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptLevel {
    /// No optimization
    O0,
    /// Basic optimization
    O1,
    /// Full optimization
    O2,
    /// Maximum optimization
    O3,
    /// Size optimization
    Os,
    /// Aggressive size optimization
    Z,
}

impl Default for OptLevel {
    fn default() -> Self {
        OptLevel::Z
    }
}

/// Panic modes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PanicMode {
    /// Unwind stack
    Unwind,
    /// Abort immediately
    Abort,
}

impl Default for PanicMode {
    fn default() -> Self {
        PanicMode::Abort
    }
}

/// Main Q-Substrate configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QSubstrateConfig {
    /// Runtime mode
    pub runtime_mode: RuntimeMode,
    /// Memory configuration
    pub memory: MemoryConfig,
    /// Hardware configuration
    pub hardware: HardwareConfig,
    /// Build configuration
    pub build: BuildConfig,
    /// Deterministic seed
    pub deterministic_seed: u32,
    /// Enable audit logging
    pub audit_logging: bool,
    /// Enable provenance tracking
    pub provenance_tracking: bool,
    /// Max qubits (6-12)
    pub max_qubits: usize,
    /// Enable rollback
    pub enable_rollback: bool,
}

impl Default for QSubstrateConfig {
    fn default() -> Self {
        QSubstrateConfig {
            runtime_mode: RuntimeMode::Desktop,
            memory: MemoryConfig::default(),
            hardware: HardwareConfig::default(),
            build: BuildConfig::default(),
            deterministic_seed: 42,
            audit_logging: true,
            provenance_tracking: true,
            max_qubits: 12,
            enable_rollback: true,
        }
    }
}

impl QSubstrateConfig {
    /// Create configuration for desktop mode
    pub fn desktop() -> Self {
        Self::default()
    }

    /// Create configuration for micro mode (ESP32, RP2040)
    pub fn micro() -> Self {
        QSubstrateConfig {
            runtime_mode: RuntimeMode::Micro,
            memory: MemoryConfig::micro(),
            hardware: HardwareConfig {
                cpu_arch: CpuArch::Xtensa,
                available_ram_kb: 4096,
                has_fpu: true,
                use_fixed_point: false,
                num_cores: 1,
            },
            max_qubits: 8,
            ..Self::default()
        }
    }

    /// Create configuration for embedded mode
    pub fn embedded() -> Self {
        QSubstrateConfig {
            runtime_mode: RuntimeMode::Embedded,
            memory: MemoryConfig::embedded(),
            hardware: HardwareConfig {
                cpu_arch: CpuArch::Arm32,
                available_ram_kb: 1024,
                has_fpu: false,
                use_fixed_point: true,
                num_cores: 1,
            },
            max_qubits: 6,
            ..Self::default()
        }
    }

    /// Create configuration for WASM browser mode
    pub fn wasm_browser() -> Self {
        QSubstrateConfig {
            runtime_mode: RuntimeMode::WasmBrowser,
            memory: MemoryConfig::default(),
            hardware: HardwareConfig {
                cpu_arch: CpuArch::Wasm32,
                available_ram_kb: 32768,
                has_fpu: true,
                use_fixed_point: false,
                num_cores: 1,
            },
            ..Self::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // Check qubit limits
        if self.max_qubits > 16 {
            return Err("Max qubits cannot exceed 16".into());
        }
        
        // Check memory consistency
        let total_pod_memory = self.memory.ai_pod_limit_kb 
            + self.memory.quantum_pod_limit_kb 
            + self.memory.dcge_pod_limit_kb;
        
        if total_pod_memory > self.memory.total_limit_mb * 1024 {
            return Err("Pod memory exceeds total limit".into());
        }
        
        // Check hardware/runtime consistency
        match (&self.runtime_mode, &self.hardware.cpu_arch) {
            (RuntimeMode::Embedded, CpuArch::X86_64) => {
                return Err("Embedded mode should not target x86_64".into());
            }
            _ => {}
        }
        
        Ok(())
    }

    /// Get quantum state size in bytes for current config
    pub fn quantum_state_size(&self) -> usize {
        (1 << self.max_qubits) * 8  // Complex = 2 * f32 = 8 bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = QSubstrateConfig::default();
        assert_eq!(config.runtime_mode, RuntimeMode::Desktop);
        assert_eq!(config.max_qubits, 12);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_micro_config() {
        let config = QSubstrateConfig::micro();
        assert_eq!(config.runtime_mode, RuntimeMode::Micro);
        assert_eq!(config.max_qubits, 8);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_embedded_config() {
        let config = QSubstrateConfig::embedded();
        assert_eq!(config.runtime_mode, RuntimeMode::Embedded);
        assert!(config.hardware.use_fixed_point);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_quantum_state_size() {
        let config = QSubstrateConfig::default();
        let size = config.quantum_state_size();
        assert_eq!(size, 32768);  // 4096 * 8 bytes
    }

    #[test]
    fn test_validation_qubit_limit() {
        let mut config = QSubstrateConfig::default();
        config.max_qubits = 20;
        assert!(config.validate().is_err());
    }
}
