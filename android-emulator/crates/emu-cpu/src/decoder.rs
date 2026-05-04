//! AArch64 instruction decoder.
//!
//! Decodes ARMv8-A instructions into an internal representation
//! suitable for interpretation or JIT compilation.

use serde::{Deserialize, Serialize};

/// Decoded instruction representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecodedInstruction {
    // Data Processing - Immediate
    /// ADD (immediate)
    AddImm { rd: u8, rn: u8, imm: u32, shift: u8, sf: bool },
    /// SUB (immediate)
    SubImm { rd: u8, rn: u8, imm: u32, shift: u8, sf: bool },
    /// ADDS (immediate) - sets flags
    AddsImm { rd: u8, rn: u8, imm: u32, shift: u8, sf: bool },
    /// SUBS (immediate) - sets flags
    SubsImm { rd: u8, rn: u8, imm: u32, shift: u8, sf: bool },

    // Data Processing - Register
    /// ADD (shifted register)
    AddShiftedReg { rd: u8, rn: u8, rm: u8, shift: ShiftType, amount: u8, sf: bool },
    /// SUB (shifted register)
    SubShiftedReg { rd: u8, rn: u8, rm: u8, shift: ShiftType, amount: u8, sf: bool },
    /// AND (shifted register)
    AndShiftedReg { rd: u8, rn: u8, rm: u8, shift: ShiftType, amount: u8, sf: bool },
    /// ORR (shifted register)
    OrrShiftedReg { rd: u8, rn: u8, rm: u8, shift: ShiftType, amount: u8, sf: bool },
    /// EOR (shifted register)
    EorShiftedReg { rd: u8, rn: u8, rm: u8, shift: ShiftType, amount: u8, sf: bool },

    // Move
    /// MOVZ - move wide with zero
    Movz { rd: u8, imm: u16, shift: u8, sf: bool },
    /// MOVN - move wide with NOT
    Movn { rd: u8, imm: u16, shift: u8, sf: bool },
    /// MOVK - move wide with keep
    Movk { rd: u8, imm: u16, shift: u8, sf: bool },

    // Branches
    /// B - unconditional branch
    Branch { offset: i32 },
    /// BL - branch with link
    BranchLink { offset: i32 },
    /// BR - branch to register
    BranchReg { rn: u8 },
    /// BLR - branch with link to register
    BranchLinkReg { rn: u8 },
    /// RET - return from subroutine
    Ret { rn: u8 },
    /// B.cond - conditional branch
    BranchCond { offset: i32, cond: Condition },
    /// CBZ - compare and branch if zero
    Cbz { rt: u8, offset: i32, sf: bool },
    /// CBNZ - compare and branch if non-zero
    Cbnz { rt: u8, offset: i32, sf: bool },
    /// TBZ - test bit and branch if zero
    Tbz { rt: u8, bit: u8, offset: i32 },
    /// TBNZ - test bit and branch if non-zero
    Tbnz { rt: u8, bit: u8, offset: i32 },

    // Loads and Stores
    /// LDR (immediate) - load register
    LdrImm { rt: u8, rn: u8, imm: i16, size: MemSize, sign_extend: bool },
    /// STR (immediate) - store register
    StrImm { rt: u8, rn: u8, imm: i16, size: MemSize },
    /// LDR (register) - load register
    LdrReg { rt: u8, rn: u8, rm: u8, extend: ExtendType, amount: u8, size: MemSize },
    /// STR (register) - store register
    StrReg { rt: u8, rn: u8, rm: u8, extend: ExtendType, amount: u8, size: MemSize },
    /// LDP - load pair
    Ldp { rt: u8, rt2: u8, rn: u8, imm: i16, sf: bool },
    /// STP - store pair
    Stp { rt: u8, rt2: u8, rn: u8, imm: i16, sf: bool },

    // PC-relative addressing
    /// ADR - form PC-relative address
    Adr { rd: u8, imm: i32 },
    /// ADRP - form PC-relative address to 4KB page
    Adrp { rd: u8, imm: i32 },

    // System
    /// SVC - supervisor call
    Svc { imm: u16 },
    /// HVC - hypervisor call
    Hvc { imm: u16 },
    /// SMC - secure monitor call
    Smc { imm: u16 },
    /// BRK - breakpoint
    Brk { imm: u16 },
    /// NOP - no operation
    Nop,
    /// MSR - move to system register
    Msr { reg: SystemReg, rt: u8 },
    /// MRS - move from system register
    Mrs { rt: u8, reg: SystemReg },

    // Comparison
    /// CMP (immediate) - alias for SUBS with Rd=XZR
    CmpImm { rn: u8, imm: u32, sf: bool },
    /// CMP (register) - alias for SUBS with Rd=XZR
    CmpReg { rn: u8, rm: u8, sf: bool },
    /// CMN (immediate) - alias for ADDS with Rd=XZR
    CmnImm { rn: u8, imm: u32, sf: bool },
    /// TST (immediate) - alias for ANDS with Rd=XZR
    TstImm { rn: u8, imm: u64, sf: bool },
    /// TST (register) - alias for ANDS with Rd=XZR
    TstReg { rn: u8, rm: u8, sf: bool },

    // Multiply
    /// MADD - multiply-add
    Madd { rd: u8, rn: u8, rm: u8, ra: u8, sf: bool },
    /// MSUB - multiply-subtract
    Msub { rd: u8, rn: u8, rm: u8, ra: u8, sf: bool },

    // Division
    /// SDIV - signed divide
    Sdiv { rd: u8, rn: u8, rm: u8, sf: bool },
    /// UDIV - unsigned divide
    Udiv { rd: u8, rn: u8, rm: u8, sf: bool },

    // Bitfield
    /// SBFM - signed bitfield move
    Sbfm { rd: u8, rn: u8, immr: u8, imms: u8, sf: bool },
    /// UBFM - unsigned bitfield move
    Ubfm { rd: u8, rn: u8, immr: u8, imms: u8, sf: bool },
    /// BFM - bitfield move
    Bfm { rd: u8, rn: u8, immr: u8, imms: u8, sf: bool },

    // Extract
    /// EXTR - extract register
    Extr { rd: u8, rn: u8, rm: u8, lsb: u8, sf: bool },

    // Conditional select
    /// CSEL - conditional select
    Csel { rd: u8, rn: u8, rm: u8, cond: Condition, sf: bool },
    /// CSINC - conditional select increment
    Csinc { rd: u8, rn: u8, rm: u8, cond: Condition, sf: bool },
    /// CSINV - conditional select invert
    Csinv { rd: u8, rn: u8, rm: u8, cond: Condition, sf: bool },
    /// CSNEG - conditional select negate
    Csneg { rd: u8, rn: u8, rm: u8, cond: Condition, sf: bool },

    // Reverse
    /// RBIT - reverse bits
    Rbit { rd: u8, rn: u8, sf: bool },
    /// REV - reverse bytes
    Rev { rd: u8, rn: u8, sf: bool },
    /// CLZ - count leading zeros
    Clz { rd: u8, rn: u8, sf: bool },
    /// CLS - count leading sign bits
    Cls { rd: u8, rn: u8, sf: bool },

    // Undefined instruction
    Undefined { opcode: u32 },
}

/// Shift types for data processing instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShiftType {
    /// Logical shift left.
    Lsl,
    /// Logical shift right.
    Lsr,
    /// Arithmetic shift right.
    Asr,
    /// Rotate right.
    Ror,
}

impl TryFrom<u8> for ShiftType {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Lsl),
            1 => Ok(Self::Lsr),
            2 => Ok(Self::Asr),
            3 => Ok(Self::Ror),
            _ => Err(()),
        }
    }
}

/// Memory access sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemSize {
    /// 8-bit (byte).
    Byte,
    /// 16-bit (halfword).
    Half,
    /// 32-bit (word).
    Word,
    /// 64-bit (doubleword).
    Double,
    /// 128-bit (quadword, for SIMD).
    Quad,
}

impl MemSize {
    /// Get the size in bytes.
    pub fn bytes(self) -> usize {
        match self {
            Self::Byte => 1,
            Self::Half => 2,
            Self::Word => 4,
            Self::Double => 8,
            Self::Quad => 16,
        }
    }
}

/// Extend types for addressing modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExtendType {
    /// Unsigned extend byte.
    Uxtb,
    /// Unsigned extend halfword.
    Uxth,
    /// Unsigned extend word.
    Uxtw,
    /// Unsigned extend doubleword (or LSL).
    Uxtx,
    /// Signed extend byte.
    Sxtb,
    /// Signed extend halfword.
    Sxth,
    /// Signed extend word.
    Sxtw,
    /// Signed extend doubleword.
    Sxtx,
}

/// Condition codes for conditional instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Condition {
    /// Equal (Z == 1).
    Eq = 0b0000,
    /// Not equal (Z == 0).
    Ne = 0b0001,
    /// Carry set / unsigned higher or same (C == 1).
    Cs = 0b0010,
    /// Carry clear / unsigned lower (C == 0).
    Cc = 0b0011,
    /// Minus / negative (N == 1).
    Mi = 0b0100,
    /// Plus / positive or zero (N == 0).
    Pl = 0b0101,
    /// Overflow (V == 1).
    Vs = 0b0110,
    /// No overflow (V == 0).
    Vc = 0b0111,
    /// Unsigned higher (C == 1 && Z == 0).
    Hi = 0b1000,
    /// Unsigned lower or same (C == 0 || Z == 1).
    Ls = 0b1001,
    /// Signed greater than or equal (N == V).
    Ge = 0b1010,
    /// Signed less than (N != V).
    Lt = 0b1011,
    /// Signed greater than (Z == 0 && N == V).
    Gt = 0b1100,
    /// Signed less than or equal (Z == 1 || N != V).
    Le = 0b1101,
    /// Always (unconditional).
    Al = 0b1110,
    /// Always (unconditional, alternate encoding).
    Nv = 0b1111,
}

impl TryFrom<u8> for Condition {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value & 0xF {
            0b0000 => Ok(Self::Eq),
            0b0001 => Ok(Self::Ne),
            0b0010 => Ok(Self::Cs),
            0b0011 => Ok(Self::Cc),
            0b0100 => Ok(Self::Mi),
            0b0101 => Ok(Self::Pl),
            0b0110 => Ok(Self::Vs),
            0b0111 => Ok(Self::Vc),
            0b1000 => Ok(Self::Hi),
            0b1001 => Ok(Self::Ls),
            0b1010 => Ok(Self::Ge),
            0b1011 => Ok(Self::Lt),
            0b1100 => Ok(Self::Gt),
            0b1101 => Ok(Self::Le),
            0b1110 => Ok(Self::Al),
            0b1111 => Ok(Self::Nv),
            _ => Err(()),
        }
    }
}

/// System register identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SystemReg {
    /// Saved Program Status Register EL1.
    SpsrEl1,
    /// Exception Link Register EL1.
    ElrEl1,
    /// System Control Register EL1.
    SctlrEl1,
    /// Translation Table Base Register 0 EL1.
    Ttbr0El1,
    /// Translation Table Base Register 1 EL1.
    Ttbr1El1,
    /// Translation Control Register EL1.
    TcrEl1,
    /// Memory Attribute Indirection Register EL1.
    MairEl1,
    /// Vector Base Address Register EL1.
    VbarEl1,
    /// Thread ID Register EL0.
    TpidrEl0,
    /// Thread ID Register EL1.
    TpidrEl1,
    /// Counter Frequency.
    CntfrqEl0,
    /// Virtual Counter Value.
    CntvctEl0,
    /// NZCV condition flags.
    Nzcv,
    /// Floating-point Control Register.
    Fpcr,
    /// Floating-point Status Register.
    Fpsr,
    /// Unknown/unsupported system register.
    Unknown(u16),
}

/// Instruction decoder for AArch64.
pub struct InstructionDecoder;

impl InstructionDecoder {
    /// Decode a 32-bit instruction.
    pub fn decode(opcode: u32) -> DecodedInstruction {
        // Main opcode field is bits [28:25]
        let op0 = (opcode >> 25) & 0xF;

        match op0 {
            // Data processing - immediate
            0b1000 | 0b1001 => Self::decode_dp_imm(opcode),
            // Branches, exception, system
            0b1010 | 0b1011 => Self::decode_branch_sys(opcode),
            // Loads and stores
            0b0100 | 0b0110 | 0b1100 | 0b1110 => Self::decode_load_store(opcode),
            // Data processing - register
            0b0101 | 0b1101 => Self::decode_dp_reg(opcode),
            // SIMD and FP
            0b0111 | 0b1111 => Self::decode_simd_fp(opcode),
            _ => DecodedInstruction::Undefined { opcode },
        }
    }

    fn decode_dp_imm(opcode: u32) -> DecodedInstruction {
        let sf = (opcode >> 31) & 1 != 0;
        let opc = (opcode >> 29) & 0x3;
        let op2 = (opcode >> 23) & 0x7;

        match op2 {
            // PC-rel addressing
            0b000 | 0b001 => {
                let rd = (opcode & 0x1F) as u8;
                let immlo = (opcode >> 29) & 0x3;
                let immhi = (opcode >> 5) & 0x7FFFF;
                let imm = ((immhi << 2) | immlo) as i32;
                let imm = (imm << 11) >> 11; // Sign extend

                if (opcode >> 31) & 1 == 0 {
                    DecodedInstruction::Adr { rd, imm }
                } else {
                    DecodedInstruction::Adrp { rd, imm }
                }
            }
            // Add/subtract immediate
            0b010 => {
                let rd = (opcode & 0x1F) as u8;
                let rn = ((opcode >> 5) & 0x1F) as u8;
                let imm12 = ((opcode >> 10) & 0xFFF) as u32;
                let shift = ((opcode >> 22) & 1) as u8 * 12;

                match opc {
                    0b00 => DecodedInstruction::AddImm { rd, rn, imm: imm12, shift, sf },
                    0b01 => DecodedInstruction::AddsImm { rd, rn, imm: imm12, shift, sf },
                    0b10 => DecodedInstruction::SubImm { rd, rn, imm: imm12, shift, sf },
                    0b11 => DecodedInstruction::SubsImm { rd, rn, imm: imm12, shift, sf },
                    _ => DecodedInstruction::Undefined { opcode },
                }
            }
            // Move wide immediate
            0b101 => {
                let rd = (opcode & 0x1F) as u8;
                let imm16 = ((opcode >> 5) & 0xFFFF) as u16;
                let hw = ((opcode >> 21) & 0x3) as u8 * 16;

                match opc {
                    0b00 => DecodedInstruction::Movn { rd, imm: imm16, shift: hw, sf },
                    0b10 => DecodedInstruction::Movz { rd, imm: imm16, shift: hw, sf },
                    0b11 => DecodedInstruction::Movk { rd, imm: imm16, shift: hw, sf },
                    _ => DecodedInstruction::Undefined { opcode },
                }
            }
            // Bitfield
            0b110 => {
                let rd = (opcode & 0x1F) as u8;
                let rn = ((opcode >> 5) & 0x1F) as u8;
                let imms = ((opcode >> 10) & 0x3F) as u8;
                let immr = ((opcode >> 16) & 0x3F) as u8;

                match opc {
                    0b00 => DecodedInstruction::Sbfm { rd, rn, immr, imms, sf },
                    0b01 => DecodedInstruction::Bfm { rd, rn, immr, imms, sf },
                    0b10 => DecodedInstruction::Ubfm { rd, rn, immr, imms, sf },
                    _ => DecodedInstruction::Undefined { opcode },
                }
            }
            // Extract
            0b111 => {
                let rd = (opcode & 0x1F) as u8;
                let rn = ((opcode >> 5) & 0x1F) as u8;
                let rm = ((opcode >> 16) & 0x1F) as u8;
                let imms = ((opcode >> 10) & 0x3F) as u8;
                DecodedInstruction::Extr { rd, rn, rm, lsb: imms, sf }
            }
            _ => DecodedInstruction::Undefined { opcode },
        }
    }

    fn decode_branch_sys(opcode: u32) -> DecodedInstruction {
        let op0 = (opcode >> 29) & 0x7;
        let op1 = (opcode >> 22) & 0x7F;

        match (op0, op1) {
            // Unconditional branch immediate
            (0b000, _) if (opcode >> 26) & 0x1F == 0b00101 => {
                let imm26 = (opcode & 0x3FF_FFFF) as i32;
                let offset = (imm26 << 6) >> 4; // Sign extend and multiply by 4
                DecodedInstruction::Branch { offset }
            }
            // Branch with link
            (0b100, _) if (opcode >> 26) & 0x1F == 0b00101 => {
                let imm26 = (opcode & 0x3FF_FFFF) as i32;
                let offset = (imm26 << 6) >> 4;
                DecodedInstruction::BranchLink { offset }
            }
            // Conditional branch
            (0b010, _) if (opcode >> 25) & 0x7F == 0b0101010 => {
                let cond = (opcode & 0xF) as u8;
                let imm19 = ((opcode >> 5) & 0x7FFFF) as i32;
                let offset = (imm19 << 13) >> 11;
                DecodedInstruction::BranchCond {
                    offset,
                    cond: Condition::try_from(cond).unwrap_or(Condition::Al),
                }
            }
            // Compare and branch
            (0b001, _) | (0b101, _) if (opcode >> 25) & 0x7F == 0b0110100 => {
                let sf = (opcode >> 31) & 1 != 0;
                let rt = (opcode & 0x1F) as u8;
                let imm19 = ((opcode >> 5) & 0x7FFFF) as i32;
                let offset = (imm19 << 13) >> 11;
                let is_nz = (opcode >> 24) & 1 != 0;

                if is_nz {
                    DecodedInstruction::Cbnz { rt, offset, sf }
                } else {
                    DecodedInstruction::Cbz { rt, offset, sf }
                }
            }
            // Test and branch
            (0b001, _) | (0b101, _) if (opcode >> 25) & 0x7F == 0b0110110 => {
                let rt = (opcode & 0x1F) as u8;
                let imm14 = ((opcode >> 5) & 0x3FFF) as i32;
                let offset = (imm14 << 18) >> 16;
                let b5 = ((opcode >> 31) & 1) as u8;
                let b40 = ((opcode >> 19) & 0x1F) as u8;
                let bit = (b5 << 5) | b40;
                let is_nz = (opcode >> 24) & 1 != 0;

                if is_nz {
                    DecodedInstruction::Tbnz { rt, bit, offset }
                } else {
                    DecodedInstruction::Tbz { rt, bit, offset }
                }
            }
            // Unconditional branch register
            (0b110, _) if (opcode >> 25) & 0x7F == 0b1101011 => {
                let rn = ((opcode >> 5) & 0x1F) as u8;
                let opc = (opcode >> 21) & 0xF;

                match opc {
                    0b0000 => DecodedInstruction::BranchReg { rn },
                    0b0001 => DecodedInstruction::BranchLinkReg { rn },
                    0b0010 => DecodedInstruction::Ret { rn },
                    _ => DecodedInstruction::Undefined { opcode },
                }
            }
            // Exception generation
            (0b110, _) if (opcode >> 24) & 0xFF == 0b11010100 => {
                let imm16 = ((opcode >> 5) & 0xFFFF) as u16;
                let opc = (opcode >> 21) & 0x7;
                let ll = opcode & 0x3;

                match (opc, ll) {
                    (0b000, 0b01) => DecodedInstruction::Svc { imm: imm16 },
                    (0b000, 0b10) => DecodedInstruction::Hvc { imm: imm16 },
                    (0b000, 0b11) => DecodedInstruction::Smc { imm: imm16 },
                    (0b001, 0b00) => DecodedInstruction::Brk { imm: imm16 },
                    _ => DecodedInstruction::Undefined { opcode },
                }
            }
            // System instructions
            (0b110, _) if (opcode >> 22) & 0x3FF == 0b1101010100 => {
                // NOP and hints
                if opcode == 0xD503201F {
                    return DecodedInstruction::Nop;
                }

                let rt = (opcode & 0x1F) as u8;
                let l = (opcode >> 21) & 1;
                let op0 = (opcode >> 19) & 0x3;
                let op1 = (opcode >> 16) & 0x7;
                let crn = (opcode >> 12) & 0xF;
                let crm = (opcode >> 8) & 0xF;
                let op2 = (opcode >> 5) & 0x7;

                let reg = Self::decode_system_reg(op0, op1, crn, crm, op2);

                if l == 1 {
                    DecodedInstruction::Mrs { rt, reg }
                } else {
                    DecodedInstruction::Msr { reg, rt }
                }
            }
            _ => DecodedInstruction::Undefined { opcode },
        }
    }

    fn decode_load_store(opcode: u32) -> DecodedInstruction {
        let op0 = (opcode >> 28) & 0xF;
        let op1 = (opcode >> 26) & 0x1;
        let op2 = (opcode >> 23) & 0x3;
        let op3 = (opcode >> 16) & 0x3F;
        let op4 = (opcode >> 10) & 0x3;

        // Load/store pair
        if op0 & 0b0011 == 0b0010 && op1 == 0 {
            let sf = (opcode >> 31) & 1 != 0;
            let rt = (opcode & 0x1F) as u8;
            let rn = ((opcode >> 5) & 0x1F) as u8;
            let rt2 = ((opcode >> 10) & 0x1F) as u8;
            let imm7 = ((opcode >> 15) & 0x7F) as i16;
            let imm = ((imm7 << 9) >> 9) * if sf { 8 } else { 4 };
            let l = (opcode >> 22) & 1;

            return if l == 1 {
                DecodedInstruction::Ldp { rt, rt2, rn, imm, sf }
            } else {
                DecodedInstruction::Stp { rt, rt2, rn, imm, sf }
            };
        }

        // Load/store register (unsigned immediate)
        if op0 & 0b0011 == 0b0011 && op1 == 0 && op2 == 0b01 {
            let size = ((opcode >> 30) & 0x3) as u8;
            let v = (opcode >> 26) & 1;
            let opc = (opcode >> 22) & 0x3;
            let imm12 = ((opcode >> 10) & 0xFFF) as i16;
            let rn = ((opcode >> 5) & 0x1F) as u8;
            let rt = (opcode & 0x1F) as u8;

            if v == 0 {
                let mem_size = match size {
                    0 => MemSize::Byte,
                    1 => MemSize::Half,
                    2 => MemSize::Word,
                    3 => MemSize::Double,
                    _ => return DecodedInstruction::Undefined { opcode },
                };
                let scale = 1i16 << size;
                let imm = imm12 * scale;

                return match opc {
                    0b00 => DecodedInstruction::StrImm { rt, rn, imm, size: mem_size },
                    0b01 => DecodedInstruction::LdrImm { rt, rn, imm, size: mem_size, sign_extend: false },
                    0b10 => DecodedInstruction::LdrImm { rt, rn, imm, size: mem_size, sign_extend: true },
                    0b11 => DecodedInstruction::LdrImm { rt, rn, imm, size: mem_size, sign_extend: true },
                    _ => DecodedInstruction::Undefined { opcode },
                };
            }
        }

        DecodedInstruction::Undefined { opcode }
    }

    fn decode_dp_reg(opcode: u32) -> DecodedInstruction {
        let sf = (opcode >> 31) & 1 != 0;
        let op0 = (opcode >> 30) & 1;
        let op1 = (opcode >> 28) & 1;
        let op2 = (opcode >> 21) & 0xF;

        let rd = (opcode & 0x1F) as u8;
        let rn = ((opcode >> 5) & 0x1F) as u8;
        let rm = ((opcode >> 16) & 0x1F) as u8;

        // Data processing (2 source)
        if op1 == 1 && op2 == 0b0110 {
            let opc = (opcode >> 10) & 0x3F;
            return match opc {
                0b000010 => DecodedInstruction::Udiv { rd, rn, rm, sf },
                0b000011 => DecodedInstruction::Sdiv { rd, rn, rm, sf },
                _ => DecodedInstruction::Undefined { opcode },
            };
        }

        // Data processing (3 source)
        if op1 == 1 && (op2 >> 3) == 1 {
            let ra = ((opcode >> 10) & 0x1F) as u8;
            let o0 = (opcode >> 15) & 1;

            return if o0 == 0 {
                DecodedInstruction::Madd { rd, rn, rm, ra, sf }
            } else {
                DecodedInstruction::Msub { rd, rn, rm, ra, sf }
            };
        }

        // Logical (shifted register)
        if op1 == 0 && (op2 >> 3) == 0 {
            let shift_type = ShiftType::try_from(((opcode >> 22) & 0x3) as u8)
                .unwrap_or(ShiftType::Lsl);
            let amount = ((opcode >> 10) & 0x3F) as u8;
            let opc = (opcode >> 29) & 0x3;
            let n = (opcode >> 21) & 1;

            return match (opc, n) {
                (0b00, 0) => DecodedInstruction::AndShiftedReg { rd, rn, rm, shift: shift_type, amount, sf },
                (0b01, 0) => DecodedInstruction::OrrShiftedReg { rd, rn, rm, shift: shift_type, amount, sf },
                (0b10, 0) => DecodedInstruction::EorShiftedReg { rd, rn, rm, shift: shift_type, amount, sf },
                _ => DecodedInstruction::Undefined { opcode },
            };
        }

        // Add/subtract (shifted register)
        if op1 == 0 && (op2 >> 3) == 1 {
            let shift_type = ShiftType::try_from(((opcode >> 22) & 0x3) as u8)
                .unwrap_or(ShiftType::Lsl);
            let amount = ((opcode >> 10) & 0x3F) as u8;
            let s = (opcode >> 29) & 1;

            return match (op0, s) {
                (0, 0) => DecodedInstruction::AddShiftedReg { rd, rn, rm, shift: shift_type, amount, sf },
                (0, 1) => DecodedInstruction::AddShiftedReg { rd, rn, rm, shift: shift_type, amount, sf }, // ADDS
                (1, 0) => DecodedInstruction::SubShiftedReg { rd, rn, rm, shift: shift_type, amount, sf },
                (1, 1) => DecodedInstruction::SubShiftedReg { rd, rn, rm, shift: shift_type, amount, sf }, // SUBS
                _ => DecodedInstruction::Undefined { opcode },
            };
        }

        // Conditional select
        if op1 == 1 && (op2 >> 2) == 0b0010 {
            let cond = Condition::try_from(((opcode >> 12) & 0xF) as u8).unwrap_or(Condition::Al);
            let op2_low = (opcode >> 10) & 0x3;
            let o2 = opcode >> 10 & 1;

            return match (op0, o2) {
                (0, 0) => DecodedInstruction::Csel { rd, rn, rm, cond, sf },
                (0, 1) => DecodedInstruction::Csinc { rd, rn, rm, cond, sf },
                (1, 0) => DecodedInstruction::Csinv { rd, rn, rm, cond, sf },
                (1, 1) => DecodedInstruction::Csneg { rd, rn, rm, cond, sf },
                _ => DecodedInstruction::Undefined { opcode },
            };
        }

        // Data processing (1 source)
        if op1 == 1 && op2 == 0b0110 {
            let opcode2 = (opcode >> 10) & 0x3F;
            
            return match opcode2 {
                0b000000 => DecodedInstruction::Rbit { rd, rn, sf },
                0b000001 => DecodedInstruction::Rev { rd, rn, sf: false },
                0b000010 | 0b000011 => DecodedInstruction::Rev { rd, rn, sf },
                0b000100 => DecodedInstruction::Clz { rd, rn, sf },
                0b000101 => DecodedInstruction::Cls { rd, rn, sf },
                _ => DecodedInstruction::Undefined { opcode },
            };
        }

        DecodedInstruction::Undefined { opcode }
    }

    fn decode_simd_fp(_opcode: u32) -> DecodedInstruction {
        // SIMD/FP instructions are complex - return undefined for now
        // Full implementation would include FADD, FMUL, FSUB, etc.
        DecodedInstruction::Undefined { opcode: _opcode }
    }

    fn decode_system_reg(op0: u32, op1: u32, crn: u32, crm: u32, op2: u32) -> SystemReg {
        match (op0, op1, crn, crm, op2) {
            (3, 0, 4, 0, 0) => SystemReg::SpsrEl1,
            (3, 0, 4, 0, 1) => SystemReg::ElrEl1,
            (3, 0, 1, 0, 0) => SystemReg::SctlrEl1,
            (3, 0, 2, 0, 0) => SystemReg::Ttbr0El1,
            (3, 0, 2, 0, 1) => SystemReg::Ttbr1El1,
            (3, 0, 2, 0, 2) => SystemReg::TcrEl1,
            (3, 0, 10, 2, 0) => SystemReg::MairEl1,
            (3, 0, 12, 0, 0) => SystemReg::VbarEl1,
            (3, 3, 13, 0, 2) => SystemReg::TpidrEl0,
            (3, 0, 13, 0, 4) => SystemReg::TpidrEl1,
            (3, 3, 14, 0, 0) => SystemReg::CntfrqEl0,
            (3, 3, 14, 0, 2) => SystemReg::CntvctEl0,
            (3, 3, 4, 2, 0) => SystemReg::Nzcv,
            (3, 3, 4, 4, 0) => SystemReg::Fpcr,
            (3, 3, 4, 4, 1) => SystemReg::Fpsr,
            _ => {
                let encoding = ((op0 as u16) << 14) | ((op1 as u16) << 11)
                    | ((crn as u16) << 7) | ((crm as u16) << 3) | (op2 as u16);
                SystemReg::Unknown(encoding)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_nop() {
        let insn = InstructionDecoder::decode(0xD503201F);
        assert_eq!(insn, DecodedInstruction::Nop);
    }

    #[test]
    fn test_decode_ret() {
        // RET (X30)
        let insn = InstructionDecoder::decode(0xD65F03C0);
        assert_eq!(insn, DecodedInstruction::Ret { rn: 30 });
    }

    #[test]
    fn test_decode_add_imm() {
        // ADD X0, X1, #0x10
        let insn = InstructionDecoder::decode(0x91004020);
        assert!(matches!(insn, DecodedInstruction::AddImm { rd: 0, rn: 1, imm: 0x10, shift: 0, sf: true }));
    }

    #[test]
    fn test_decode_mov() {
        // MOVZ X0, #0x1234
        let insn = InstructionDecoder::decode(0xD2824680);
        assert!(matches!(insn, DecodedInstruction::Movz { rd: 0, imm: 0x1234, shift: 0, sf: true }));
    }

    #[test]
    fn test_decode_svc() {
        // SVC #0
        let insn = InstructionDecoder::decode(0xD4000001);
        assert_eq!(insn, DecodedInstruction::Svc { imm: 0 });
    }
}
