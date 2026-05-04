//! CPU execution engine.
//!
//! Provides the interpreter and execution control for the CPU emulator.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use emu_memory::Mmu;

use crate::decoder::{Condition, DecodedInstruction, InstructionDecoder, MemSize, ShiftType, SystemReg};
use crate::error::{CpuError, CpuResult};
use crate::registers::{Arm64Registers, ExceptionLevel, PState};

/// Execution mode for the CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionMode {
    /// Normal execution.
    Normal,
    /// Single-step mode.
    SingleStep,
    /// Run until breakpoint.
    Continue,
}

/// CPU execution engine.
pub struct Executor {
    /// Current execution mode.
    mode: RwLock<ExecutionMode>,
    /// Breakpoint addresses.
    breakpoints: RwLock<Vec<u64>>,
    /// Instruction count for this run.
    instruction_count: RwLock<u64>,
    /// Maximum instructions before forced stop (0 = unlimited).
    instruction_limit: u64,
}

impl Executor {
    /// Create a new executor.
    pub fn new() -> Self {
        Self {
            mode: RwLock::new(ExecutionMode::Normal),
            breakpoints: RwLock::new(Vec::new()),
            instruction_count: RwLock::new(0),
            instruction_limit: 0,
        }
    }

    /// Create an executor with an instruction limit.
    pub fn with_limit(limit: u64) -> Self {
        Self {
            mode: RwLock::new(ExecutionMode::Normal),
            breakpoints: RwLock::new(Vec::new()),
            instruction_count: RwLock::new(0),
            instruction_limit: limit,
        }
    }

    /// Set the execution mode.
    pub fn set_mode(&self, mode: ExecutionMode) {
        *self.mode.write() = mode;
    }

    /// Get the current execution mode.
    pub fn mode(&self) -> ExecutionMode {
        *self.mode.read()
    }

    /// Add a breakpoint.
    pub fn add_breakpoint(&self, addr: u64) {
        let mut bps = self.breakpoints.write();
        if !bps.contains(&addr) {
            bps.push(addr);
        }
    }

    /// Remove a breakpoint.
    pub fn remove_breakpoint(&self, addr: u64) {
        self.breakpoints.write().retain(|&a| a != addr);
    }

    /// Clear all breakpoints.
    pub fn clear_breakpoints(&self) {
        self.breakpoints.write().clear();
    }

    /// Check if an address is a breakpoint.
    pub fn is_breakpoint(&self, addr: u64) -> bool {
        self.breakpoints.read().contains(&addr)
    }

    /// Get the instruction count.
    pub fn instruction_count(&self) -> u64 {
        *self.instruction_count.read()
    }

    /// Reset the instruction count.
    pub fn reset_instruction_count(&self) {
        *self.instruction_count.write() = 0;
    }

    /// Execute a single instruction.
    pub fn step(&self, regs: &mut Arm64Registers, mmu: &Mmu) -> CpuResult<()> {
        // Check instruction limit
        if self.instruction_limit > 0 {
            let count = *self.instruction_count.read();
            if count >= self.instruction_limit {
                return Err(CpuError::InstructionLimit { count });
            }
        }

        // Check for breakpoints
        if self.is_breakpoint(regs.pc) {
            return Err(CpuError::Breakpoint { pc: regs.pc });
        }

        // Fetch instruction
        let opcode = mmu.read_u32(regs.pc)?;
        
        // Decode
        let insn = InstructionDecoder::decode(opcode);
        
        // Execute
        self.execute_instruction(regs, mmu, &insn)?;

        // Increment instruction count
        *self.instruction_count.write() += 1;

        Ok(())
    }

    /// Run until completion, breakpoint, or error.
    pub fn run(&self, regs: &mut Arm64Registers, mmu: &Mmu) -> CpuResult<()> {
        loop {
            match self.step(regs, mmu) {
                Ok(()) => {
                    if *self.mode.read() == ExecutionMode::SingleStep {
                        return Err(CpuError::SingleStep { pc: regs.pc });
                    }
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Execute a decoded instruction.
    fn execute_instruction(
        &self,
        regs: &mut Arm64Registers,
        mmu: &Mmu,
        insn: &DecodedInstruction,
    ) -> CpuResult<()> {
        match *insn {
            // Branches
            DecodedInstruction::Branch { offset } => {
                regs.pc = regs.pc.wrapping_add_signed(offset as i64);
                return Ok(());
            }
            DecodedInstruction::BranchLink { offset } => {
                regs.set_lr(regs.pc.wrapping_add(4));
                regs.pc = regs.pc.wrapping_add_signed(offset as i64);
                return Ok(());
            }
            DecodedInstruction::BranchReg { rn } => {
                regs.pc = regs.get_x(rn as usize);
                return Ok(());
            }
            DecodedInstruction::BranchLinkReg { rn } => {
                let target = regs.get_x(rn as usize);
                regs.set_lr(regs.pc.wrapping_add(4));
                regs.pc = target;
                return Ok(());
            }
            DecodedInstruction::Ret { rn } => {
                regs.pc = regs.get_x(rn as usize);
                return Ok(());
            }
            DecodedInstruction::BranchCond { offset, cond } => {
                if self.evaluate_condition(&regs.pstate, cond) {
                    regs.pc = regs.pc.wrapping_add_signed(offset as i64);
                    return Ok(());
                }
            }
            DecodedInstruction::Cbz { rt, offset, sf } => {
                let val = if sf { regs.get_x(rt as usize) } else { regs.get_w(rt as usize) as u64 };
                if val == 0 {
                    regs.pc = regs.pc.wrapping_add_signed(offset as i64);
                    return Ok(());
                }
            }
            DecodedInstruction::Cbnz { rt, offset, sf } => {
                let val = if sf { regs.get_x(rt as usize) } else { regs.get_w(rt as usize) as u64 };
                if val != 0 {
                    regs.pc = regs.pc.wrapping_add_signed(offset as i64);
                    return Ok(());
                }
            }
            DecodedInstruction::Tbz { rt, bit, offset } => {
                let val = regs.get_x(rt as usize);
                if (val >> bit) & 1 == 0 {
                    regs.pc = regs.pc.wrapping_add_signed(offset as i64);
                    return Ok(());
                }
            }
            DecodedInstruction::Tbnz { rt, bit, offset } => {
                let val = regs.get_x(rt as usize);
                if (val >> bit) & 1 != 0 {
                    regs.pc = regs.pc.wrapping_add_signed(offset as i64);
                    return Ok(());
                }
            }

            // PC-relative addressing
            DecodedInstruction::Adr { rd, imm } => {
                let result = regs.pc.wrapping_add_signed(imm as i64);
                regs.set_x(rd as usize, result);
            }
            DecodedInstruction::Adrp { rd, imm } => {
                let base = regs.pc & !0xFFF;
                let result = base.wrapping_add_signed((imm as i64) << 12);
                regs.set_x(rd as usize, result);
            }

            // Arithmetic - Immediate
            DecodedInstruction::AddImm { rd, rn, imm, shift, sf } => {
                let op1 = if sf { regs.get_x(rn as usize) } else { regs.get_w(rn as usize) as u64 };
                let op2 = (imm as u64) << shift;
                let result = op1.wrapping_add(op2);
                if sf {
                    regs.set_x(rd as usize, result);
                } else {
                    regs.set_w(rd as usize, result as u32);
                }
            }
            DecodedInstruction::SubImm { rd, rn, imm, shift, sf } => {
                let op1 = if sf { regs.get_x(rn as usize) } else { regs.get_w(rn as usize) as u64 };
                let op2 = (imm as u64) << shift;
                let result = op1.wrapping_sub(op2);
                if sf {
                    regs.set_x(rd as usize, result);
                } else {
                    regs.set_w(rd as usize, result as u32);
                }
            }
            DecodedInstruction::AddsImm { rd, rn, imm, shift, sf } => {
                let op1 = if sf { regs.get_x(rn as usize) } else { regs.get_w(rn as usize) as u64 };
                let op2 = (imm as u64) << shift;
                let (result, carry) = op1.overflowing_add(op2);
                let overflow = ((op1 ^ result) & (op2 ^ result)) >> (if sf { 63 } else { 31 }) & 1 != 0;
                
                if sf {
                    regs.pstate.update_nzcv_64(result, carry, overflow);
                    regs.set_x(rd as usize, result);
                } else {
                    let result32 = result as u32;
                    regs.pstate.update_nzcv_32(result32, carry, overflow);
                    regs.set_w(rd as usize, result32);
                }
            }
            DecodedInstruction::SubsImm { rd, rn, imm, shift, sf } => {
                let op1 = if sf { regs.get_x(rn as usize) } else { regs.get_w(rn as usize) as u64 };
                let op2 = (imm as u64) << shift;
                let (result, borrow) = op1.overflowing_sub(op2);
                let overflow = ((op1 ^ op2) & (op1 ^ result)) >> (if sf { 63 } else { 31 }) & 1 != 0;
                
                if sf {
                    regs.pstate.update_nzcv_64(result, !borrow, overflow);
                    regs.set_x(rd as usize, result);
                } else {
                    let result32 = result as u32;
                    regs.pstate.update_nzcv_32(result32, !borrow, overflow);
                    regs.set_w(rd as usize, result32);
                }
            }

            // Arithmetic - Register
            DecodedInstruction::AddShiftedReg { rd, rn, rm, shift, amount, sf } => {
                let op1 = if sf { regs.get_x(rn as usize) } else { regs.get_w(rn as usize) as u64 };
                let op2_base = if sf { regs.get_x(rm as usize) } else { regs.get_w(rm as usize) as u64 };
                let op2 = self.apply_shift(op2_base, shift, amount, sf);
                let result = op1.wrapping_add(op2);
                if sf {
                    regs.set_x(rd as usize, result);
                } else {
                    regs.set_w(rd as usize, result as u32);
                }
            }
            DecodedInstruction::SubShiftedReg { rd, rn, rm, shift, amount, sf } => {
                let op1 = if sf { regs.get_x(rn as usize) } else { regs.get_w(rn as usize) as u64 };
                let op2_base = if sf { regs.get_x(rm as usize) } else { regs.get_w(rm as usize) as u64 };
                let op2 = self.apply_shift(op2_base, shift, amount, sf);
                let result = op1.wrapping_sub(op2);
                if sf {
                    regs.set_x(rd as usize, result);
                } else {
                    regs.set_w(rd as usize, result as u32);
                }
            }

            // Logical - Register
            DecodedInstruction::AndShiftedReg { rd, rn, rm, shift, amount, sf } => {
                let op1 = if sf { regs.get_x(rn as usize) } else { regs.get_w(rn as usize) as u64 };
                let op2_base = if sf { regs.get_x(rm as usize) } else { regs.get_w(rm as usize) as u64 };
                let op2 = self.apply_shift(op2_base, shift, amount, sf);
                let result = op1 & op2;
                if sf {
                    regs.set_x(rd as usize, result);
                } else {
                    regs.set_w(rd as usize, result as u32);
                }
            }
            DecodedInstruction::OrrShiftedReg { rd, rn, rm, shift, amount, sf } => {
                let op1 = if sf { regs.get_x(rn as usize) } else { regs.get_w(rn as usize) as u64 };
                let op2_base = if sf { regs.get_x(rm as usize) } else { regs.get_w(rm as usize) as u64 };
                let op2 = self.apply_shift(op2_base, shift, amount, sf);
                let result = op1 | op2;
                if sf {
                    regs.set_x(rd as usize, result);
                } else {
                    regs.set_w(rd as usize, result as u32);
                }
            }
            DecodedInstruction::EorShiftedReg { rd, rn, rm, shift, amount, sf } => {
                let op1 = if sf { regs.get_x(rn as usize) } else { regs.get_w(rn as usize) as u64 };
                let op2_base = if sf { regs.get_x(rm as usize) } else { regs.get_w(rm as usize) as u64 };
                let op2 = self.apply_shift(op2_base, shift, amount, sf);
                let result = op1 ^ op2;
                if sf {
                    regs.set_x(rd as usize, result);
                } else {
                    regs.set_w(rd as usize, result as u32);
                }
            }

            // Move immediate
            DecodedInstruction::Movz { rd, imm, shift, sf } => {
                let result = (imm as u64) << shift;
                if sf {
                    regs.set_x(rd as usize, result);
                } else {
                    regs.set_w(rd as usize, result as u32);
                }
            }
            DecodedInstruction::Movn { rd, imm, shift, sf } => {
                let result = !((imm as u64) << shift);
                if sf {
                    regs.set_x(rd as usize, result);
                } else {
                    regs.set_w(rd as usize, result as u32);
                }
            }
            DecodedInstruction::Movk { rd, imm, shift, sf } => {
                let current = regs.get_x(rd as usize);
                let mask = !((0xFFFF_u64) << shift);
                let result = (current & mask) | ((imm as u64) << shift);
                if sf {
                    regs.set_x(rd as usize, result);
                } else {
                    regs.set_w(rd as usize, result as u32);
                }
            }

            // Loads and Stores
            DecodedInstruction::LdrImm { rt, rn, imm, size, sign_extend } => {
                let base = if rn == 31 { regs.sp() } else { regs.get_x(rn as usize) };
                let addr = base.wrapping_add_signed(imm as i64);
                
                let value = match size {
                    MemSize::Byte => mmu.read_u8(addr)? as u64,
                    MemSize::Half => mmu.read_u16(addr)? as u64,
                    MemSize::Word => mmu.read_u32(addr)? as u64,
                    MemSize::Double => mmu.read_u64(addr)?,
                    MemSize::Quad => return Err(CpuError::Internal("128-bit load not implemented".into())),
                };

                let result = if sign_extend {
                    match size {
                        MemSize::Byte => (value as i8) as i64 as u64,
                        MemSize::Half => (value as i16) as i64 as u64,
                        MemSize::Word => (value as i32) as i64 as u64,
                        _ => value,
                    }
                } else {
                    value
                };

                regs.set_x(rt as usize, result);
            }
            DecodedInstruction::StrImm { rt, rn, imm, size } => {
                let base = if rn == 31 { regs.sp() } else { regs.get_x(rn as usize) };
                let addr = base.wrapping_add_signed(imm as i64);
                let value = regs.get_x(rt as usize);

                match size {
                    MemSize::Byte => mmu.write_u8(addr, value as u8)?,
                    MemSize::Half => mmu.write_u16(addr, value as u16)?,
                    MemSize::Word => mmu.write_u32(addr, value as u32)?,
                    MemSize::Double => mmu.write_u64(addr, value)?,
                    MemSize::Quad => return Err(CpuError::Internal("128-bit store not implemented".into())),
                }
            }
            DecodedInstruction::Ldp { rt, rt2, rn, imm, sf } => {
                let base = if rn == 31 { regs.sp() } else { regs.get_x(rn as usize) };
                let addr = base.wrapping_add_signed(imm as i64);
                let size = if sf { 8 } else { 4 };

                let val1 = if sf {
                    mmu.read_u64(addr)?
                } else {
                    mmu.read_u32(addr)? as u64
                };
                let val2 = if sf {
                    mmu.read_u64(addr + size)?
                } else {
                    mmu.read_u32(addr + size)? as u64
                };

                if sf {
                    regs.set_x(rt as usize, val1);
                    regs.set_x(rt2 as usize, val2);
                } else {
                    regs.set_w(rt as usize, val1 as u32);
                    regs.set_w(rt2 as usize, val2 as u32);
                }
            }
            DecodedInstruction::Stp { rt, rt2, rn, imm, sf } => {
                let base = if rn == 31 { regs.sp() } else { regs.get_x(rn as usize) };
                let addr = base.wrapping_add_signed(imm as i64);
                let size = if sf { 8 } else { 4 };

                let val1 = regs.get_x(rt as usize);
                let val2 = regs.get_x(rt2 as usize);

                if sf {
                    mmu.write_u64(addr, val1)?;
                    mmu.write_u64(addr + size, val2)?;
                } else {
                    mmu.write_u32(addr, val1 as u32)?;
                    mmu.write_u32(addr + size, val2 as u32)?;
                }
            }

            // Multiply
            DecodedInstruction::Madd { rd, rn, rm, ra, sf } => {
                let a = regs.get_x(ra as usize);
                let n = regs.get_x(rn as usize);
                let m = regs.get_x(rm as usize);
                let result = a.wrapping_add(n.wrapping_mul(m));
                if sf {
                    regs.set_x(rd as usize, result);
                } else {
                    regs.set_w(rd as usize, result as u32);
                }
            }
            DecodedInstruction::Msub { rd, rn, rm, ra, sf } => {
                let a = regs.get_x(ra as usize);
                let n = regs.get_x(rn as usize);
                let m = regs.get_x(rm as usize);
                let result = a.wrapping_sub(n.wrapping_mul(m));
                if sf {
                    regs.set_x(rd as usize, result);
                } else {
                    regs.set_w(rd as usize, result as u32);
                }
            }

            // Division
            DecodedInstruction::Sdiv { rd, rn, rm, sf } => {
                if sf {
                    let n = regs.get_x(rn as usize) as i64;
                    let m = regs.get_x(rm as usize) as i64;
                    let result = if m == 0 { 0 } else { n.wrapping_div(m) };
                    regs.set_x(rd as usize, result as u64);
                } else {
                    let n = regs.get_w(rn as usize) as i32;
                    let m = regs.get_w(rm as usize) as i32;
                    let result = if m == 0 { 0 } else { n.wrapping_div(m) };
                    regs.set_w(rd as usize, result as u32);
                }
            }
            DecodedInstruction::Udiv { rd, rn, rm, sf } => {
                if sf {
                    let n = regs.get_x(rn as usize);
                    let m = regs.get_x(rm as usize);
                    let result = if m == 0 { 0 } else { n / m };
                    regs.set_x(rd as usize, result);
                } else {
                    let n = regs.get_w(rn as usize);
                    let m = regs.get_w(rm as usize);
                    let result = if m == 0 { 0 } else { n / m };
                    regs.set_w(rd as usize, result);
                }
            }

            // Conditional select
            DecodedInstruction::Csel { rd, rn, rm, cond, sf } => {
                let val = if self.evaluate_condition(&regs.pstate, cond) {
                    regs.get_x(rn as usize)
                } else {
                    regs.get_x(rm as usize)
                };
                if sf {
                    regs.set_x(rd as usize, val);
                } else {
                    regs.set_w(rd as usize, val as u32);
                }
            }
            DecodedInstruction::Csinc { rd, rn, rm, cond, sf } => {
                let val = if self.evaluate_condition(&regs.pstate, cond) {
                    regs.get_x(rn as usize)
                } else {
                    regs.get_x(rm as usize).wrapping_add(1)
                };
                if sf {
                    regs.set_x(rd as usize, val);
                } else {
                    regs.set_w(rd as usize, val as u32);
                }
            }
            DecodedInstruction::Csinv { rd, rn, rm, cond, sf } => {
                let val = if self.evaluate_condition(&regs.pstate, cond) {
                    regs.get_x(rn as usize)
                } else {
                    !regs.get_x(rm as usize)
                };
                if sf {
                    regs.set_x(rd as usize, val);
                } else {
                    regs.set_w(rd as usize, val as u32);
                }
            }
            DecodedInstruction::Csneg { rd, rn, rm, cond, sf } => {
                let val = if self.evaluate_condition(&regs.pstate, cond) {
                    regs.get_x(rn as usize)
                } else {
                    (regs.get_x(rm as usize) as i64).wrapping_neg() as u64
                };
                if sf {
                    regs.set_x(rd as usize, val);
                } else {
                    regs.set_w(rd as usize, val as u32);
                }
            }

            // Bit manipulation
            DecodedInstruction::Clz { rd, rn, sf } => {
                let val = if sf { regs.get_x(rn as usize) } else { regs.get_w(rn as usize) as u64 };
                let result = if sf { val.leading_zeros() as u64 } else { (val as u32).leading_zeros() as u64 };
                if sf {
                    regs.set_x(rd as usize, result);
                } else {
                    regs.set_w(rd as usize, result as u32);
                }
            }
            DecodedInstruction::Rbit { rd, rn, sf } => {
                let val = if sf { regs.get_x(rn as usize) } else { regs.get_w(rn as usize) as u64 };
                let result = if sf { val.reverse_bits() } else { (val as u32).reverse_bits() as u64 };
                if sf {
                    regs.set_x(rd as usize, result);
                } else {
                    regs.set_w(rd as usize, result as u32);
                }
            }
            DecodedInstruction::Rev { rd, rn, sf } => {
                let val = if sf { regs.get_x(rn as usize) } else { regs.get_w(rn as usize) as u64 };
                let result = if sf { val.swap_bytes() } else { (val as u32).swap_bytes() as u64 };
                if sf {
                    regs.set_x(rd as usize, result);
                } else {
                    regs.set_w(rd as usize, result as u32);
                }
            }

            // Bitfield
            DecodedInstruction::Ubfm { rd, rn, immr, imms, sf } => {
                let src = if sf { regs.get_x(rn as usize) } else { regs.get_w(rn as usize) as u64 };
                let width = if sf { 64 } else { 32 };
                
                let result = if imms >= immr {
                    // UBFX case
                    let len = imms - immr + 1;
                    (src >> immr) & ((1u64 << len) - 1)
                } else {
                    // LSL case
                    let shift = width - 1 - imms;
                    src << shift
                };

                if sf {
                    regs.set_x(rd as usize, result);
                } else {
                    regs.set_w(rd as usize, result as u32);
                }
            }
            DecodedInstruction::Sbfm { rd, rn, immr, imms, sf } => {
                let src = if sf { regs.get_x(rn as usize) } else { regs.get_w(rn as usize) as u64 };
                let width = if sf { 64 } else { 32 };
                
                let result = if imms >= immr {
                    // SBFX case - sign extend
                    let len = imms - immr + 1;
                    let extracted = (src >> immr) & ((1u64 << len) - 1);
                    let sign_bit = (extracted >> (len - 1)) & 1;
                    if sign_bit == 1 {
                        extracted | (!0u64 << len)
                    } else {
                        extracted
                    }
                } else {
                    // ASR case
                    if sf {
                        ((src as i64) >> immr) as u64
                    } else {
                        ((src as i32) >> immr) as u64
                    }
                };

                if sf {
                    regs.set_x(rd as usize, result);
                } else {
                    regs.set_w(rd as usize, result as u32);
                }
            }

            // System instructions
            DecodedInstruction::Nop => {}
            DecodedInstruction::Svc { imm } => {
                return Err(CpuError::SupervisorCall { imm });
            }
            DecodedInstruction::Hvc { imm } => {
                return Err(CpuError::HypervisorCall { imm });
            }
            DecodedInstruction::Smc { imm } => {
                return Err(CpuError::SecureMonitorCall { imm });
            }
            DecodedInstruction::Brk { imm: _ } => {
                return Err(CpuError::Breakpoint { pc: regs.pc });
            }
            DecodedInstruction::Mrs { rt, reg } => {
                let value = self.read_system_reg(regs, reg);
                regs.set_x(rt as usize, value);
            }
            DecodedInstruction::Msr { reg, rt } => {
                let value = regs.get_x(rt as usize);
                self.write_system_reg(regs, reg, value);
            }

            // Comparison (aliases)
            DecodedInstruction::CmpImm { rn, imm, sf } => {
                // CMP is SUBS with Rd=XZR
                let op1 = if sf { regs.get_x(rn as usize) } else { regs.get_w(rn as usize) as u64 };
                let op2 = imm as u64;
                let (result, borrow) = op1.overflowing_sub(op2);
                let overflow = ((op1 ^ op2) & (op1 ^ result)) >> (if sf { 63 } else { 31 }) & 1 != 0;
                
                if sf {
                    regs.pstate.update_nzcv_64(result, !borrow, overflow);
                } else {
                    regs.pstate.update_nzcv_32(result as u32, !borrow, overflow);
                }
            }

            DecodedInstruction::Undefined { opcode } => {
                return Err(CpuError::UndefinedInstruction { pc: regs.pc, opcode });
            }

            // Default for unimplemented instructions
            _ => {
                log::warn!("Unimplemented instruction: {:?}", insn);
            }
        }

        // Advance PC for non-branch instructions
        regs.advance_pc();
        Ok(())
    }

    /// Apply a shift operation.
    fn apply_shift(&self, value: u64, shift: ShiftType, amount: u8, sf: bool) -> u64 {
        let width = if sf { 64 } else { 32 };
        let amount = (amount as u32) % width;

        match shift {
            ShiftType::Lsl => value << amount,
            ShiftType::Lsr => {
                if sf {
                    value >> amount
                } else {
                    ((value as u32) >> amount) as u64
                }
            }
            ShiftType::Asr => {
                if sf {
                    ((value as i64) >> amount) as u64
                } else {
                    ((value as i32) >> amount) as u64
                }
            }
            ShiftType::Ror => {
                if sf {
                    value.rotate_right(amount)
                } else {
                    (value as u32).rotate_right(amount) as u64
                }
            }
        }
    }

    /// Evaluate a condition code.
    fn evaluate_condition(&self, pstate: &PState, cond: Condition) -> bool {
        match cond {
            Condition::Eq => pstate.is_zero(),
            Condition::Ne => !pstate.is_zero(),
            Condition::Cs => pstate.is_carry(),
            Condition::Cc => !pstate.is_carry(),
            Condition::Mi => pstate.is_negative(),
            Condition::Pl => !pstate.is_negative(),
            Condition::Vs => pstate.is_overflow(),
            Condition::Vc => !pstate.is_overflow(),
            Condition::Hi => pstate.is_carry() && !pstate.is_zero(),
            Condition::Ls => !pstate.is_carry() || pstate.is_zero(),
            Condition::Ge => pstate.is_negative() == pstate.is_overflow(),
            Condition::Lt => pstate.is_negative() != pstate.is_overflow(),
            Condition::Gt => !pstate.is_zero() && pstate.is_negative() == pstate.is_overflow(),
            Condition::Le => pstate.is_zero() || pstate.is_negative() != pstate.is_overflow(),
            Condition::Al | Condition::Nv => true,
        }
    }

    /// Read a system register.
    fn read_system_reg(&self, regs: &Arm64Registers, reg: SystemReg) -> u64 {
        match reg {
            SystemReg::SpsrEl1 => {
                let el = ExceptionLevel::EL1 as usize;
                regs.spsr[el].bits()
            }
            SystemReg::ElrEl1 => {
                let el = ExceptionLevel::EL1 as usize;
                regs.elr[el]
            }
            SystemReg::SctlrEl1 => {
                let el = ExceptionLevel::EL1 as usize;
                regs.sctlr[el]
            }
            SystemReg::Ttbr0El1 => {
                let el = ExceptionLevel::EL1 as usize;
                regs.ttbr0[el]
            }
            SystemReg::Ttbr1El1 => regs.ttbr1_el1,
            SystemReg::TcrEl1 => {
                let el = ExceptionLevel::EL1 as usize;
                regs.tcr[el]
            }
            SystemReg::MairEl1 => {
                let el = ExceptionLevel::EL1 as usize;
                regs.mair[el]
            }
            SystemReg::VbarEl1 => {
                let el = ExceptionLevel::EL1 as usize;
                regs.vbar[el]
            }
            SystemReg::TpidrEl0 => regs.tpidr_el0,
            SystemReg::TpidrEl1 => regs.tpidr_el1,
            SystemReg::CntfrqEl0 => regs.cntfrq_el0,
            SystemReg::CntvctEl0 => regs.cntvct_el0,
            SystemReg::Nzcv => (regs.pstate & PState::NZCV).bits(),
            SystemReg::Fpcr => regs.fpcr as u64,
            SystemReg::Fpsr => regs.fpsr as u64,
            SystemReg::Unknown(_) => 0,
        }
    }

    /// Write a system register.
    fn write_system_reg(&self, regs: &mut Arm64Registers, reg: SystemReg, value: u64) {
        match reg {
            SystemReg::SpsrEl1 => {
                let el = ExceptionLevel::EL1 as usize;
                regs.spsr[el] = PState::from_bits_truncate(value);
            }
            SystemReg::ElrEl1 => {
                let el = ExceptionLevel::EL1 as usize;
                regs.elr[el] = value;
            }
            SystemReg::SctlrEl1 => {
                let el = ExceptionLevel::EL1 as usize;
                regs.sctlr[el] = value;
            }
            SystemReg::Ttbr0El1 => {
                let el = ExceptionLevel::EL1 as usize;
                regs.ttbr0[el] = value;
            }
            SystemReg::Ttbr1El1 => regs.ttbr1_el1 = value,
            SystemReg::TcrEl1 => {
                let el = ExceptionLevel::EL1 as usize;
                regs.tcr[el] = value;
            }
            SystemReg::MairEl1 => {
                let el = ExceptionLevel::EL1 as usize;
                regs.mair[el] = value;
            }
            SystemReg::VbarEl1 => {
                let el = ExceptionLevel::EL1 as usize;
                regs.vbar[el] = value;
            }
            SystemReg::TpidrEl0 => regs.tpidr_el0 = value,
            SystemReg::TpidrEl1 => regs.tpidr_el1 = value,
            SystemReg::CntfrqEl0 => regs.cntfrq_el0 = value,
            SystemReg::CntvctEl0 => regs.cntvct_el0 = value,
            SystemReg::Nzcv => {
                regs.pstate = (regs.pstate - PState::NZCV) | PState::from_bits_truncate(value & PState::NZCV.bits());
            }
            SystemReg::Fpcr => regs.fpcr = value as u32,
            SystemReg::Fpsr => regs.fpsr = value as u32,
            SystemReg::Unknown(_) => {}
        }
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use emu_memory::{MemoryRegion, RegionType, RegionPermissions};

    fn setup_test_env() -> (Arm64Registers, Mmu, Executor) {
        let regs = Arm64Registers::new();
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
        
        // Map stack region
        let stack_region = MemoryRegion::new(
            "stack",
            0x100000,
            0x10000,
            RegionType::Stack,
            RegionPermissions::RW,
        );
        mmu.map_region(stack_region).unwrap();
        
        let executor = Executor::new();
        (regs, mmu, executor)
    }

    #[test]
    fn test_add_immediate() {
        let (mut regs, mmu, executor) = setup_test_env();
        
        // MOV X0, #10
        mmu.write_u32(0, 0xD2800140).unwrap();
        // ADD X1, X0, #5
        mmu.write_u32(4, 0x91001401).unwrap();
        
        executor.step(&mut regs, &mmu).unwrap();
        assert_eq!(regs.get_x(0), 10);
        
        executor.step(&mut regs, &mmu).unwrap();
        assert_eq!(regs.get_x(1), 15);
    }

    #[test]
    fn test_branch() {
        let (mut regs, mmu, executor) = setup_test_env();
        
        // B +8 (skip next instruction)
        mmu.write_u32(0, 0x14000002).unwrap();
        // MOV X0, #1 (should be skipped)
        mmu.write_u32(4, 0xD2800020).unwrap();
        // MOV X0, #2
        mmu.write_u32(8, 0xD2800040).unwrap();
        
        executor.step(&mut regs, &mmu).unwrap();
        assert_eq!(regs.pc, 8);
        
        executor.step(&mut regs, &mmu).unwrap();
        assert_eq!(regs.get_x(0), 2);
    }

    #[test]
    fn test_conditional_branch() {
        let (mut regs, mmu, executor) = setup_test_env();
        
        // Set zero flag
        regs.pstate |= PState::Z;
        
        // B.EQ +8
        mmu.write_u32(0, 0x54000040).unwrap();
        
        executor.step(&mut regs, &mmu).unwrap();
        assert_eq!(regs.pc, 8);
    }

    #[test]
    fn test_svc() {
        let (mut regs, mmu, executor) = setup_test_env();
        
        // SVC #123
        mmu.write_u32(0, 0xD4000F61).unwrap();
        
        let result = executor.step(&mut regs, &mmu);
        assert!(matches!(result, Err(CpuError::SupervisorCall { imm: 123 })));
    }
}
