//! Linux syscall emulation for Android.
//!
//! Implements the Linux syscall interface as used by Android.

use emu_cpu::Arm64Registers;
use emu_memory::Mmu;

use crate::error::{RuntimeError, RuntimeResult};

/// Linux syscall numbers for AArch64.
pub mod syscall_numbers {
    pub const SYS_READ: u64 = 63;
    pub const SYS_WRITE: u64 = 64;
    pub const SYS_OPENAT: u64 = 56;
    pub const SYS_CLOSE: u64 = 57;
    pub const SYS_FSTAT: u64 = 80;
    pub const SYS_LSEEK: u64 = 62;
    pub const SYS_MMAP: u64 = 222;
    pub const SYS_MPROTECT: u64 = 226;
    pub const SYS_MUNMAP: u64 = 215;
    pub const SYS_BRK: u64 = 214;
    pub const SYS_IOCTL: u64 = 29;
    pub const SYS_WRITEV: u64 = 66;
    pub const SYS_CLONE: u64 = 220;
    pub const SYS_EXIT: u64 = 93;
    pub const SYS_EXIT_GROUP: u64 = 94;
    pub const SYS_FUTEX: u64 = 98;
    pub const SYS_SET_TID_ADDRESS: u64 = 96;
    pub const SYS_CLOCK_GETTIME: u64 = 113;
    pub const SYS_GETPID: u64 = 172;
    pub const SYS_GETTID: u64 = 178;
    pub const SYS_GETUID: u64 = 174;
    pub const SYS_GETEUID: u64 = 175;
    pub const SYS_GETGID: u64 = 176;
    pub const SYS_GETEGID: u64 = 177;
    pub const SYS_GETRANDOM: u64 = 278;
    pub const SYS_PRCTL: u64 = 167;
    pub const SYS_RT_SIGACTION: u64 = 134;
    pub const SYS_RT_SIGPROCMASK: u64 = 135;
}

/// Syscall result (return value or error).
pub type SyscallResult = Result<u64, i32>;

/// Syscall emulator.
pub struct SyscallEmulator {
    /// Current process ID.
    pid: u32,
    /// Current thread ID.
    tid: u32,
    /// User ID.
    uid: u32,
    /// Group ID.
    gid: u32,
    /// Program break (heap end).
    brk: u64,
    /// Minimum brk value.
    brk_min: u64,
    /// Maximum brk value.
    brk_max: u64,
}

impl SyscallEmulator {
    /// Create a new syscall emulator.
    pub fn new() -> Self {
        Self {
            pid: 1000,
            tid: 1000,
            uid: 10000,  // First app UID
            gid: 10000,
            brk: 0x2000_0000,
            brk_min: 0x2000_0000,
            brk_max: 0x4000_0000,
        }
    }

    /// Handle a syscall.
    ///
    /// Takes the syscall number from X8 and arguments from X0-X5.
    /// Returns the result in X0.
    pub fn handle_syscall(&mut self, regs: &mut Arm64Registers, mmu: &Mmu) -> SyscallResult {
        let syscall_num = regs.get_x(8);
        let arg0 = regs.get_x(0);
        let arg1 = regs.get_x(1);
        let arg2 = regs.get_x(2);
        let arg3 = regs.get_x(3);
        let arg4 = regs.get_x(4);
        let arg5 = regs.get_x(5);

        log::trace!(
            "syscall: {} ({:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x})",
            syscall_num, arg0, arg1, arg2, arg3, arg4, arg5
        );

        let result = match syscall_num {
            syscall_numbers::SYS_READ => self.sys_read(arg0 as i32, arg1, arg2 as usize, mmu),
            syscall_numbers::SYS_WRITE => self.sys_write(arg0 as i32, arg1, arg2 as usize, mmu),
            syscall_numbers::SYS_CLOSE => self.sys_close(arg0 as i32),
            syscall_numbers::SYS_BRK => self.sys_brk(arg0),
            syscall_numbers::SYS_EXIT => self.sys_exit(arg0 as i32),
            syscall_numbers::SYS_EXIT_GROUP => self.sys_exit_group(arg0 as i32),
            syscall_numbers::SYS_GETPID => self.sys_getpid(),
            syscall_numbers::SYS_GETTID => self.sys_gettid(),
            syscall_numbers::SYS_GETUID => self.sys_getuid(),
            syscall_numbers::SYS_GETEUID => self.sys_geteuid(),
            syscall_numbers::SYS_GETGID => self.sys_getgid(),
            syscall_numbers::SYS_GETEGID => self.sys_getegid(),
            syscall_numbers::SYS_SET_TID_ADDRESS => self.sys_set_tid_address(arg0),
            syscall_numbers::SYS_CLOCK_GETTIME => self.sys_clock_gettime(arg0 as i32, arg1, mmu),
            syscall_numbers::SYS_GETRANDOM => self.sys_getrandom(arg0, arg1 as usize, arg2 as u32, mmu),
            syscall_numbers::SYS_MMAP => self.sys_mmap(arg0, arg1 as usize, arg2 as i32, arg3 as i32, arg4 as i32, arg5),
            syscall_numbers::SYS_MUNMAP => self.sys_munmap(arg0, arg1 as usize),
            syscall_numbers::SYS_MPROTECT => self.sys_mprotect(arg0, arg1 as usize, arg2 as i32),
            syscall_numbers::SYS_RT_SIGACTION => Ok(0), // Stub
            syscall_numbers::SYS_RT_SIGPROCMASK => Ok(0), // Stub
            syscall_numbers::SYS_FUTEX => Ok(0), // Stub
            syscall_numbers::SYS_PRCTL => Ok(0), // Stub
            _ => {
                log::warn!("Unimplemented syscall: {}", syscall_num);
                Err(-38) // ENOSYS
            }
        };

        // Set return value in X0
        match result {
            Ok(val) => {
                regs.set_x(0, val);
            }
            Err(errno) => {
                regs.set_x(0, (-errno) as u64);
            }
        }

        result
    }

    fn sys_read(&self, fd: i32, buf: u64, count: usize, mmu: &Mmu) -> SyscallResult {
        log::trace!("sys_read(fd={}, buf={:#x}, count={})", fd, buf, count);
        
        // For now, return EOF for stdin, error for others
        match fd {
            0 => Ok(0), // EOF for stdin
            _ => Err(-9), // EBADF
        }
    }

    fn sys_write(&self, fd: i32, buf: u64, count: usize, mmu: &Mmu) -> SyscallResult {
        log::trace!("sys_write(fd={}, buf={:#x}, count={})", fd, buf, count);
        
        // Read data from guest memory
        let mut data = vec![0u8; count.min(4096)];
        if let Err(_) = mmu.read(buf, &mut data) {
            return Err(-14); // EFAULT
        }

        match fd {
            1 | 2 => {
                // stdout/stderr - print to log
                let s = String::from_utf8_lossy(&data);
                log::info!("Guest output (fd {}): {}", fd, s.trim_end());
                Ok(count as u64)
            }
            _ => Err(-9), // EBADF
        }
    }

    fn sys_close(&self, fd: i32) -> SyscallResult {
        log::trace!("sys_close(fd={})", fd);
        Ok(0)
    }

    fn sys_brk(&mut self, addr: u64) -> SyscallResult {
        log::trace!("sys_brk(addr={:#x})", addr);
        
        if addr == 0 {
            return Ok(self.brk);
        }
        
        if addr >= self.brk_min && addr <= self.brk_max {
            self.brk = addr;
            Ok(self.brk)
        } else {
            Ok(self.brk) // Return current brk on failure
        }
    }

    fn sys_exit(&self, status: i32) -> SyscallResult {
        log::info!("sys_exit(status={})", status);
        Err(-1000 - status) // Special code to indicate exit
    }

    fn sys_exit_group(&self, status: i32) -> SyscallResult {
        log::info!("sys_exit_group(status={})", status);
        Err(-1000 - status) // Special code to indicate exit
    }

    fn sys_getpid(&self) -> SyscallResult {
        Ok(self.pid as u64)
    }

    fn sys_gettid(&self) -> SyscallResult {
        Ok(self.tid as u64)
    }

    fn sys_getuid(&self) -> SyscallResult {
        Ok(self.uid as u64)
    }

    fn sys_geteuid(&self) -> SyscallResult {
        Ok(self.uid as u64)
    }

    fn sys_getgid(&self) -> SyscallResult {
        Ok(self.gid as u64)
    }

    fn sys_getegid(&self) -> SyscallResult {
        Ok(self.gid as u64)
    }

    fn sys_set_tid_address(&mut self, tidptr: u64) -> SyscallResult {
        log::trace!("sys_set_tid_address(tidptr={:#x})", tidptr);
        Ok(self.tid as u64)
    }

    fn sys_clock_gettime(&self, clockid: i32, tp: u64, mmu: &Mmu) -> SyscallResult {
        log::trace!("sys_clock_gettime(clockid={}, tp={:#x})", clockid, tp);
        
        // Return a fake time
        let secs: u64 = 1704067200; // Jan 1, 2024
        let nsecs: u64 = 0;
        
        if mmu.write_u64(tp, secs).is_err() || mmu.write_u64(tp + 8, nsecs).is_err() {
            return Err(-14); // EFAULT
        }
        
        Ok(0)
    }

    fn sys_getrandom(&self, buf: u64, buflen: usize, flags: u32, mmu: &Mmu) -> SyscallResult {
        log::trace!("sys_getrandom(buf={:#x}, buflen={}, flags={})", buf, buflen, flags);
        
        // Fill with pseudo-random data
        for i in 0..buflen.min(256) {
            let byte = ((i * 17 + 42) % 256) as u8;
            if mmu.write_u8(buf + i as u64, byte).is_err() {
                return Err(-14); // EFAULT
            }
        }
        
        Ok(buflen.min(256) as u64)
    }

    fn sys_mmap(&mut self, addr: u64, length: usize, prot: i32, flags: i32, fd: i32, offset: u64) -> SyscallResult {
        log::trace!(
            "sys_mmap(addr={:#x}, length={}, prot={}, flags={}, fd={}, offset={})",
            addr, length, prot, flags, fd, offset
        );
        
        // Simplified mmap - just bump the brk
        let aligned_length = (length + 0xFFF) & !0xFFF;
        let result = self.brk;
        self.brk += aligned_length as u64;
        
        if self.brk > self.brk_max {
            self.brk = result;
            return Err(-12); // ENOMEM
        }
        
        Ok(result)
    }

    fn sys_munmap(&self, addr: u64, length: usize) -> SyscallResult {
        log::trace!("sys_munmap(addr={:#x}, length={})", addr, length);
        Ok(0) // Stub - pretend it worked
    }

    fn sys_mprotect(&self, addr: u64, len: usize, prot: i32) -> SyscallResult {
        log::trace!("sys_mprotect(addr={:#x}, len={}, prot={})", addr, len, prot);
        Ok(0) // Stub - pretend it worked
    }
}

impl Default for SyscallEmulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_getpid() {
        let mut emu = SyscallEmulator::new();
        assert_eq!(emu.sys_getpid(), Ok(1000));
    }

    #[test]
    fn test_brk() {
        let mut emu = SyscallEmulator::new();
        let initial_brk = emu.sys_brk(0).unwrap();
        
        let new_brk = initial_brk + 0x10000;
        let result = emu.sys_brk(new_brk).unwrap();
        assert_eq!(result, new_brk);
    }
}
