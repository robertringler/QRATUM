//! AArch64 CPU core implementation.
//!
//! Combines registers, decoder, and executor into a complete CPU.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use emu_memory::Mmu;

use crate::error::{CpuError, CpuResult};
use crate::executor::{ExecutionMode, Executor};
use crate::registers::Arm64Registers;

/// CPU configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuConfig {
    /// CPU frequency in Hz.
    pub frequency_hz: u64,
    /// Enable deterministic mode (for record/replay).
    pub deterministic: bool,
    /// Maximum instructions before forced stop (0 = unlimited).
    pub instruction_limit: u64,
    /// Enable instruction tracing.
    pub trace_enabled: bool,
}

impl Default for CpuConfig {
    fn default() -> Self {
        Self {
            frequency_hz: 1_000_000_000, // 1 GHz
            deterministic: false,
            instruction_limit: 0,
            trace_enabled: false,
        }
    }
}

/// CPU execution state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CpuState {
    /// CPU is stopped.
    Stopped,
    /// CPU is running.
    Running,
    /// CPU is halted (WFI/WFE).
    Halted,
    /// CPU hit a breakpoint.
    Breakpoint,
    /// CPU raised an exception.
    Exception,
}

/// AArch64 CPU implementation.
pub struct Arm64Cpu {
    /// CPU configuration.
    config: CpuConfig,
    /// Register file.
    pub regs: Arm64Registers,
    /// Execution engine.
    executor: Executor,
    /// Current state.
    state: RwLock<CpuState>,
    /// Cycle counter.
    cycle_count: RwLock<u64>,
}

impl Arm64Cpu {
    /// Create a new CPU with the given configuration.
    pub fn new(config: CpuConfig) -> Self {
        let executor = if config.instruction_limit > 0 {
            Executor::with_limit(config.instruction_limit)
        } else {
            Executor::new()
        };

        Self {
            config,
            regs: Arm64Registers::new(),
            executor,
            state: RwLock::new(CpuState::Stopped),
            cycle_count: RwLock::new(0),
        }
    }

    /// Create a CPU with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(CpuConfig::default())
    }

    /// Get the current CPU state.
    pub fn state(&self) -> CpuState {
        *self.state.read()
    }

    /// Get the cycle count.
    pub fn cycle_count(&self) -> u64 {
        *self.cycle_count.read()
    }

    /// Get the instruction count.
    pub fn instruction_count(&self) -> u64 {
        self.executor.instruction_count()
    }

    /// Reset the CPU to power-on state.
    pub fn reset(&mut self) {
        self.regs.reset();
        self.executor.reset_instruction_count();
        *self.state.write() = CpuState::Stopped;
        *self.cycle_count.write() = 0;
    }

    /// Set the program counter.
    pub fn set_pc(&mut self, pc: u64) {
        self.regs.pc = pc;
    }

    /// Get the program counter.
    pub fn pc(&self) -> u64 {
        self.regs.pc
    }

    /// Add a breakpoint.
    pub fn add_breakpoint(&self, addr: u64) {
        self.executor.add_breakpoint(addr);
    }

    /// Remove a breakpoint.
    pub fn remove_breakpoint(&self, addr: u64) {
        self.executor.remove_breakpoint(addr);
    }

    /// Clear all breakpoints.
    pub fn clear_breakpoints(&self) {
        self.executor.clear_breakpoints();
    }

    /// Execute a single instruction.
    pub fn step(&mut self, mmu: &Mmu) -> CpuResult<()> {
        *self.state.write() = CpuState::Running;
        
        match self.executor.step(&mut self.regs, mmu) {
            Ok(()) => {
                *self.cycle_count.write() += 1;
                Ok(())
            }
            Err(CpuError::Breakpoint { pc }) => {
                *self.state.write() = CpuState::Breakpoint;
                Err(CpuError::Breakpoint { pc })
            }
            Err(CpuError::SupervisorCall { imm }) => {
                // SVC is expected - advance PC and return the call
                self.regs.advance_pc();
                Err(CpuError::SupervisorCall { imm })
            }
            Err(e) => {
                *self.state.write() = CpuState::Exception;
                Err(e)
            }
        }
    }

    /// Run until a stop condition.
    pub fn run(&mut self, mmu: &Mmu, max_instructions: u64) -> CpuResult<()> {
        *self.state.write() = CpuState::Running;
        
        for _ in 0..max_instructions {
            match self.step(mmu) {
                Ok(()) => continue,
                Err(CpuError::InstructionLimit { count }) => {
                    *self.state.write() = CpuState::Stopped;
                    return Ok(());
                }
                Err(e) => return Err(e),
            }
        }
        
        *self.state.write() = CpuState::Stopped;
        Ok(())
    }

    /// Set single-step mode.
    pub fn set_single_step(&self, enabled: bool) {
        self.executor.set_mode(if enabled {
            ExecutionMode::SingleStep
        } else {
            ExecutionMode::Normal
        });
    }

    /// Get the configuration.
    pub fn config(&self) -> &CpuConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use emu_memory::{MemoryRegion, RegionType, RegionPermissions};

    #[test]
    fn test_cpu_creation() {
        let cpu = Arm64Cpu::with_defaults();
        assert_eq!(cpu.state(), CpuState::Stopped);
        assert_eq!(cpu.pc(), 0);
    }

    #[test]
    fn test_cpu_reset() {
        let mut cpu = Arm64Cpu::with_defaults();
        cpu.set_pc(0x1000);
        cpu.reset();
        assert_eq!(cpu.pc(), 0);
    }

    #[test]
    fn test_cpu_execution() {
        let mut cpu = Arm64Cpu::with_defaults();
        let mmu = Mmu::new(16 * 1024 * 1024);
        
        // Map code region
        let code_region = MemoryRegion::new(
            "code",
            0x0,
            0x10000,
            RegionType::Ram,
            RegionPermissions::RWX,
        );
        mmu.map_region(code_region).unwrap();
        
        // Write a simple program: MOV X0, #42; RET
        mmu.write_u32(0, 0xD2800540).unwrap(); // MOV X0, #42
        mmu.write_u32(4, 0xD65F03C0).unwrap(); // RET
        
        // Execute MOV
        cpu.step(&mmu).unwrap();
        assert_eq!(cpu.regs.get_x(0), 42);
        assert_eq!(cpu.pc(), 4);
    }
}
