//! AArch64 register file and processor state.
//!
//! Implements the ARMv8-A register model including:
//! - 31 general-purpose registers (X0-X30)
//! - Stack pointer (SP)
//! - Program counter (PC)
//! - Processor state (PSTATE)
//! - System registers

use bitflags::bitflags;
use serde::{Deserialize, Serialize};

/// Exception levels in ARMv8-A.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ExceptionLevel {
    /// EL0: User/Application level.
    EL0 = 0,
    /// EL1: Operating System kernel level.
    EL1 = 1,
    /// EL2: Hypervisor level.
    EL2 = 2,
    /// EL3: Secure Monitor level.
    EL3 = 3,
}

impl Default for ExceptionLevel {
    fn default() -> Self {
        Self::EL0
    }
}

impl TryFrom<u8> for ExceptionLevel {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::EL0),
            1 => Ok(Self::EL1),
            2 => Ok(Self::EL2),
            3 => Ok(Self::EL3),
            _ => Err(()),
        }
    }
}

bitflags! {
    /// Processor state flags (PSTATE).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PState: u64 {
        /// Negative condition flag.
        const N = 1 << 31;
        /// Zero condition flag.
        const Z = 1 << 30;
        /// Carry condition flag.
        const C = 1 << 29;
        /// Overflow condition flag.
        const V = 1 << 28;

        /// Debug mask.
        const D = 1 << 9;
        /// SError interrupt mask.
        const A = 1 << 8;
        /// IRQ interrupt mask.
        const I = 1 << 7;
        /// FIQ interrupt mask.
        const F = 1 << 6;

        /// Execution state (0 = AArch64).
        const M4 = 1 << 4;
        /// EL bits (combined).
        const EL_MASK = 0b11 << 2;
        /// Stack pointer select (0 = SP_EL0, 1 = SP_ELx).
        const SP = 1 << 0;

        /// NZCV flags combined.
        const NZCV = Self::N.bits() | Self::Z.bits() | Self::C.bits() | Self::V.bits();
        /// All interrupt masks.
        const DAIF = Self::D.bits() | Self::A.bits() | Self::I.bits() | Self::F.bits();
    }
}

// Manual Serialize/Deserialize for bitflags
impl Serialize for PState {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.bits().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PState {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bits = u64::deserialize(deserializer)?;
        Ok(Self::from_bits_truncate(bits))
    }
}

impl Default for PState {
    fn default() -> Self {
        // Default: All interrupts masked, EL0, AArch64 mode
        Self::DAIF
    }
}

impl PState {
    /// Get the current exception level.
    pub fn exception_level(&self) -> ExceptionLevel {
        let el = ((self.bits() >> 2) & 0b11) as u8;
        ExceptionLevel::try_from(el).unwrap_or(ExceptionLevel::EL0)
    }

    /// Set the exception level.
    pub fn set_exception_level(&mut self, el: ExceptionLevel) {
        *self = (*self - Self::EL_MASK) | Self::from_bits_truncate((el as u64) << 2);
    }

    /// Check if the N (negative) flag is set.
    pub fn is_negative(&self) -> bool {
        self.contains(Self::N)
    }

    /// Check if the Z (zero) flag is set.
    pub fn is_zero(&self) -> bool {
        self.contains(Self::Z)
    }

    /// Check if the C (carry) flag is set.
    pub fn is_carry(&self) -> bool {
        self.contains(Self::C)
    }

    /// Check if the V (overflow) flag is set.
    pub fn is_overflow(&self) -> bool {
        self.contains(Self::V)
    }

    /// Set or clear the N flag.
    pub fn set_negative(&mut self, value: bool) {
        self.set(Self::N, value);
    }

    /// Set or clear the Z flag.
    pub fn set_zero(&mut self, value: bool) {
        self.set(Self::Z, value);
    }

    /// Set or clear the C flag.
    pub fn set_carry(&mut self, value: bool) {
        self.set(Self::C, value);
    }

    /// Set or clear the V flag.
    pub fn set_overflow(&mut self, value: bool) {
        self.set(Self::V, value);
    }

    /// Update NZCV flags based on a result.
    pub fn update_nzcv_64(&mut self, result: u64, carry: bool, overflow: bool) {
        self.set_negative((result as i64) < 0);
        self.set_zero(result == 0);
        self.set_carry(carry);
        self.set_overflow(overflow);
    }

    /// Update NZCV flags for 32-bit operations.
    pub fn update_nzcv_32(&mut self, result: u32, carry: bool, overflow: bool) {
        self.set_negative((result as i32) < 0);
        self.set_zero(result == 0);
        self.set_carry(carry);
        self.set_overflow(overflow);
    }
}

/// SIMD/FP register (128-bit).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct SimdRegister {
    /// Low 64 bits.
    pub low: u64,
    /// High 64 bits.
    pub high: u64,
}

impl SimdRegister {
    /// Create a new zero register.
    pub fn zero() -> Self {
        Self { low: 0, high: 0 }
    }

    /// Get as 64-bit value (D register view).
    pub fn as_d(&self) -> u64 {
        self.low
    }

    /// Get as 32-bit value (S register view).
    pub fn as_s(&self) -> u32 {
        self.low as u32
    }

    /// Get as 16-bit value (H register view).
    pub fn as_h(&self) -> u16 {
        self.low as u16
    }

    /// Get as 8-bit value (B register view).
    pub fn as_b(&self) -> u8 {
        self.low as u8
    }

    /// Set from 64-bit value.
    pub fn set_d(&mut self, value: u64) {
        self.low = value;
        self.high = 0;
    }

    /// Set from 32-bit value.
    pub fn set_s(&mut self, value: u32) {
        self.low = value as u64;
        self.high = 0;
    }

    /// Get as f64.
    pub fn as_f64(&self) -> f64 {
        f64::from_bits(self.low)
    }

    /// Get as f32.
    pub fn as_f32(&self) -> f32 {
        f32::from_bits(self.low as u32)
    }

    /// Set from f64.
    pub fn set_f64(&mut self, value: f64) {
        self.low = value.to_bits();
        self.high = 0;
    }

    /// Set from f32.
    pub fn set_f32(&mut self, value: f32) {
        self.low = value.to_bits() as u64;
        self.high = 0;
    }
}

/// ARMv8-A AArch64 register file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Arm64Registers {
    /// General-purpose registers X0-X30.
    pub x: [u64; 31],
    /// Stack pointer for each exception level.
    pub sp: [u64; 4],
    /// Program counter.
    pub pc: u64,
    /// Processor state.
    pub pstate: PState,
    /// SIMD/FP registers V0-V31.
    pub v: [SimdRegister; 32],
    /// Floating-point control register.
    pub fpcr: u32,
    /// Floating-point status register.
    pub fpsr: u32,
    /// Exception Link Register (ELR) for each EL.
    pub elr: [u64; 4],
    /// Saved Program Status Register (SPSR) for each EL.
    pub spsr: [PState; 4],
    /// System Control Register (SCTLR) for each EL.
    pub sctlr: [u64; 4],
    /// Translation Table Base Register 0 (TTBR0) for each EL.
    pub ttbr0: [u64; 4],
    /// Translation Table Base Register 1 (TTBR1) for EL1.
    pub ttbr1_el1: u64,
    /// Translation Control Register (TCR) for each EL.
    pub tcr: [u64; 4],
    /// Memory Attribute Indirection Register (MAIR) for each EL.
    pub mair: [u64; 4],
    /// Vector Base Address Register (VBAR) for each EL.
    pub vbar: [u64; 4],
    /// Thread ID registers.
    pub tpidr_el0: u64,
    /// Thread ID register EL1.
    pub tpidr_el1: u64,
    /// Counter frequency.
    pub cntfrq_el0: u64,
    /// Virtual counter value.
    pub cntvct_el0: u64,
}

impl Default for Arm64Registers {
    fn default() -> Self {
        Self {
            x: [0; 31],
            sp: [0; 4],
            pc: 0,
            pstate: PState::default(),
            v: [SimdRegister::zero(); 32],
            fpcr: 0,
            fpsr: 0,
            elr: [0; 4],
            spsr: [PState::default(); 4],
            sctlr: [0; 4],
            ttbr0: [0; 4],
            ttbr1_el1: 0,
            tcr: [0; 4],
            mair: [0; 4],
            vbar: [0; 4],
            tpidr_el0: 0,
            tpidr_el1: 0,
            // 1 GHz default frequency
            cntfrq_el0: 1_000_000_000,
            cntvct_el0: 0,
        }
    }
}

impl Arm64Registers {
    /// Create a new register file with all registers zeroed.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a general-purpose register value (X0-X30).
    ///
    /// Reading X31 returns the zero register (XZR).
    pub fn get_x(&self, reg: usize) -> u64 {
        if reg >= 31 {
            0 // XZR
        } else {
            self.x[reg]
        }
    }

    /// Set a general-purpose register value (X0-X30).
    ///
    /// Writing to X31 (XZR) is discarded.
    pub fn set_x(&mut self, reg: usize, value: u64) {
        if reg < 31 {
            self.x[reg] = value;
        }
    }

    /// Get a 32-bit view of a general-purpose register (W0-W30).
    pub fn get_w(&self, reg: usize) -> u32 {
        self.get_x(reg) as u32
    }

    /// Set a 32-bit view of a general-purpose register.
    ///
    /// This zero-extends the value to 64 bits.
    pub fn set_w(&mut self, reg: usize, value: u32) {
        self.set_x(reg, value as u64);
    }

    /// Get the current stack pointer.
    pub fn sp(&self) -> u64 {
        if self.pstate.contains(PState::SP) {
            let el = self.pstate.exception_level() as usize;
            self.sp[el]
        } else {
            self.sp[0]
        }
    }

    /// Set the current stack pointer.
    pub fn set_sp(&mut self, value: u64) {
        if self.pstate.contains(PState::SP) {
            let el = self.pstate.exception_level() as usize;
            self.sp[el] = value;
        } else {
            self.sp[0] = value;
        }
    }

    /// Get the link register (X30/LR).
    pub fn lr(&self) -> u64 {
        self.x[30]
    }

    /// Set the link register.
    pub fn set_lr(&mut self, value: u64) {
        self.x[30] = value;
    }

    /// Get the frame pointer (X29/FP).
    pub fn fp(&self) -> u64 {
        self.x[29]
    }

    /// Set the frame pointer.
    pub fn set_fp(&mut self, value: u64) {
        self.x[29] = value;
    }

    /// Get the current exception level.
    pub fn exception_level(&self) -> ExceptionLevel {
        self.pstate.exception_level()
    }

    /// Advance the program counter by one instruction.
    pub fn advance_pc(&mut self) {
        self.pc = self.pc.wrapping_add(4);
    }

    /// Reset the CPU to power-on state.
    pub fn reset(&mut self) {
        *self = Self::default();
        // Start at EL1 after reset
        self.pstate.set_exception_level(ExceptionLevel::EL1);
        // All interrupts masked
        self.pstate |= PState::DAIF;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pstate_nzcv() {
        let mut pstate = PState::default();
        
        pstate.set_negative(true);
        assert!(pstate.is_negative());
        
        pstate.update_nzcv_64(0, false, false);
        assert!(!pstate.is_negative());
        assert!(pstate.is_zero());
    }

    #[test]
    fn test_exception_level() {
        let mut pstate = PState::default();
        
        pstate.set_exception_level(ExceptionLevel::EL1);
        assert_eq!(pstate.exception_level(), ExceptionLevel::EL1);
        
        pstate.set_exception_level(ExceptionLevel::EL2);
        assert_eq!(pstate.exception_level(), ExceptionLevel::EL2);
    }

    #[test]
    fn test_registers_xzr() {
        let mut regs = Arm64Registers::new();
        
        // XZR reads as zero
        assert_eq!(regs.get_x(31), 0);
        
        // Writing to XZR is discarded
        regs.set_x(31, 0xDEADBEEF);
        assert_eq!(regs.get_x(31), 0);
    }

    #[test]
    fn test_registers_w_view() {
        let mut regs = Arm64Registers::new();
        
        regs.set_x(0, 0xFFFF_FFFF_DEAD_BEEF);
        assert_eq!(regs.get_w(0), 0xDEAD_BEEF);
        
        // Setting W zero-extends
        regs.set_w(1, 0xCAFE_BABE);
        assert_eq!(regs.get_x(1), 0xCAFE_BABE);
    }
}
