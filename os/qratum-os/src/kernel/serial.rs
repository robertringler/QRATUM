//! 16550-compatible UART driver (Agent 6).
//!
//! Polled byte I/O on COM1 (0x3F8). QEMU's `-serial stdio` exposes this on
//! the host terminal. The driver is `unsafe` only at the port-I/O boundary;
//! the `Serial` value itself is a zero-sized type so multiple users compose
//! safely without locking (P0.0 is single-CPU).

use core::fmt;

const PORT_BASE: u16 = 0x3F8;

#[inline] unsafe fn outb(port: u16, val: u8) {
    core::arch::asm!("out dx, al", in("dx") port, in("al") val, options(nomem, nostack, preserves_flags));
}
#[inline] unsafe fn inb(port: u16) -> u8 {
    let v: u8;
    core::arch::asm!("in al, dx", out("al") v, in("dx") port, options(nomem, nostack, preserves_flags));
    v
}

/// Initialise COM1 at 38400 8N1 with FIFO enabled.
pub fn init() {
    unsafe {
        outb(PORT_BASE + 1, 0x00);    // disable interrupts
        outb(PORT_BASE + 3, 0x80);    // DLAB on
        outb(PORT_BASE + 0, 0x03);    // divisor low  (38400)
        outb(PORT_BASE + 1, 0x00);    // divisor high
        outb(PORT_BASE + 3, 0x03);    // 8N1, DLAB off
        outb(PORT_BASE + 2, 0xC7);    // FIFO on, clear, 14-byte threshold
        outb(PORT_BASE + 4, 0x0B);    // RTS/DSR set, OUT2 (IRQs would need this)
    }
}

#[inline]
fn tx_ready() -> bool { unsafe { inb(PORT_BASE + 5) & 0x20 != 0 } }
#[inline]
fn rx_ready() -> bool { unsafe { inb(PORT_BASE + 5) & 0x01 != 0 } }

pub fn write_byte(b: u8) {
    while !tx_ready() {}
    unsafe { outb(PORT_BASE, b); }
}

pub fn write_str(s: &str) {
    for &b in s.as_bytes() {
        if b == b'\n' { write_byte(b'\r'); }
        write_byte(b);
    }
}

/// Non-blocking read. Returns `None` if no byte is available.
pub fn try_read_byte() -> Option<u8> {
    if rx_ready() { Some(unsafe { inb(PORT_BASE) }) } else { None }
}

/// Blocking read.
pub fn read_byte() -> u8 {
    while !rx_ready() {}
    unsafe { inb(PORT_BASE) }
}

pub struct Serial;
impl fmt::Write for Serial {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        write_str(s);
        Ok(())
    }
}

#[macro_export]
macro_rules! kprint {
    ($($arg:tt)*) => {{
        use core::fmt::Write as _;
        let _ = write!($crate::kernel::serial::Serial, $($arg)*);
    }};
}

#[macro_export]
macro_rules! kprintln {
    () => { $crate::kprint!("\n") };
    ($($arg:tt)*) => {{
        use core::fmt::Write as _;
        let _ = writeln!($crate::kernel::serial::Serial, $($arg)*);
    }};
}
