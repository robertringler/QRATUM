//! # emu-cpu
//!
//! ARMv8 ↔ x86-64 CPU emulation with dynamic binary translation.
//!
//! This crate provides:
//! - ARMv8-A (AArch64) instruction set emulation
//! - Dynamic binary translation for performance
//! - Register-accurate execution model
//! - Deterministic execution support for record/replay
//!
//! ## Architecture
//!
//! The CPU emulation uses an interpreter for initial execution and can
//! optionally use JIT compilation for hot paths (feature-gated).
//!
//! ## Legal Notice
//!
//! This implementation is original work based on publicly available
//! ARM Architecture Reference Manual. No proprietary code is used.
//! Licensed under Apache 2.0.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

mod arm64;
mod decoder;
mod error;
mod executor;
mod registers;

pub use arm64::{Arm64Cpu, CpuConfig, CpuState};
pub use decoder::{DecodedInstruction, InstructionDecoder};
pub use error::{CpuError, CpuResult};
pub use executor::{ExecutionMode, Executor};
pub use registers::{Arm64Registers, ExceptionLevel, PState};

/// Number of general-purpose registers in AArch64.
pub const NUM_GP_REGISTERS: usize = 31;

/// Number of SIMD/FP registers in AArch64.
pub const NUM_SIMD_REGISTERS: usize = 32;

/// Default instruction fetch size.
pub const INSTRUCTION_SIZE: usize = 4;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(NUM_GP_REGISTERS, 31);
        assert_eq!(NUM_SIMD_REGISTERS, 32);
        assert_eq!(INSTRUCTION_SIZE, 4);
    }
}
