//! CPU emulation error types.

use thiserror::Error;

/// Result type for CPU operations.
pub type CpuResult<T> = Result<T, CpuError>;

/// Errors that can occur during CPU emulation.
#[derive(Debug, Error)]
pub enum CpuError {
    /// Undefined instruction encountered.
    #[error("undefined instruction at {pc:#x}: {opcode:#010x}")]
    UndefinedInstruction {
        /// Program counter where the error occurred.
        pc: u64,
        /// The undefined opcode.
        opcode: u32,
    },

    /// Illegal instruction for current exception level.
    #[error("illegal instruction at {pc:#x} for EL{el}")]
    IllegalInstruction {
        /// Program counter.
        pc: u64,
        /// Current exception level.
        el: u8,
    },

    /// Memory access fault.
    #[error("memory fault at {pc:#x}: address {addr:#x}")]
    MemoryFault {
        /// Program counter where the fault occurred.
        pc: u64,
        /// The faulting address.
        addr: u64,
    },

    /// Alignment fault.
    #[error("alignment fault at {pc:#x}: address {addr:#x}")]
    AlignmentFault {
        /// Program counter.
        pc: u64,
        /// The misaligned address.
        addr: u64,
    },

    /// Synchronous exception.
    #[error("synchronous exception: {syndrome:#x}")]
    SynchronousException {
        /// Exception syndrome register value.
        syndrome: u64,
    },

    /// Software interrupt (SVC instruction).
    #[error("supervisor call: imm={imm:#x}")]
    SupervisorCall {
        /// Immediate value from SVC instruction.
        imm: u16,
    },

    /// Hypervisor call (HVC instruction).
    #[error("hypervisor call: imm={imm:#x}")]
    HypervisorCall {
        /// Immediate value from HVC instruction.
        imm: u16,
    },

    /// Secure monitor call (SMC instruction).
    #[error("secure monitor call: imm={imm:#x}")]
    SecureMonitorCall {
        /// Immediate value from SMC instruction.
        imm: u16,
    },

    /// Breakpoint hit.
    #[error("breakpoint at {pc:#x}")]
    Breakpoint {
        /// Address of the breakpoint.
        pc: u64,
    },

    /// Single-step complete.
    #[error("single step complete at {pc:#x}")]
    SingleStep {
        /// Current program counter.
        pc: u64,
    },

    /// CPU halted.
    #[error("CPU halted")]
    Halted,

    /// Instruction count limit reached.
    #[error("instruction limit reached: {count} instructions")]
    InstructionLimit {
        /// Number of instructions executed.
        count: u64,
    },

    /// Memory error from emu-memory.
    #[error("memory error: {0}")]
    Memory(#[from] emu_memory::MemoryError),

    /// Internal emulator error.
    #[error("internal error: {0}")]
    Internal(String),
}
