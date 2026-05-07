//! Kernel panic handler (Agent 10).
//!
//! Writes a final audit frame, prints the panic message over serial,
//! disables interrupts, and halts every CPU. P0.0 is single-core, but
//! the IPI broadcast is the same primitive used in P0.3.

use core::panic::PanicInfo;

use super::audit::{self, FrameKind};
use super::serial;

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    // Disable interrupts immediately.
    unsafe { core::arch::asm!("cli", options(nomem, nostack, preserves_flags)); }

    // Try to journal the event; payload truncated to fit a frame.
    let mut payload = [0u8; 62];
    let _ = write_to_buf(&mut payload, info);
    audit::append(FrameKind::KernelPanic, &payload);

    // Best-effort serial diagnostic.
    serial::write_str("\r\n!! KERNEL PANIC !!\r\n");
    if let Some(loc) = info.location() {
        let mut buf = [0u8; 96];
        let _ = format_location(&mut buf, loc.file(), loc.line());
        let s = core::str::from_utf8(&buf).unwrap_or("<panic loc>");
        serial::write_str(s);
        serial::write_str("\r\n");
    }
    {
        let mut tmp = [0u8; 256];
        let n = render_msg_into(&mut tmp, info.message());
        let s = core::str::from_utf8(&tmp[..n]).unwrap_or("<panic msg>");
        serial::write_str(s);
        serial::write_str("\r\n");
    }
    serial::write_str("system halted.\r\n");

    loop { unsafe { core::arch::asm!("hlt", options(nomem, nostack, preserves_flags)); } }
}

fn write_to_buf(buf: &mut [u8], info: &PanicInfo) -> usize {
    let mut n = 0;
    let prefix = b"PANIC ";
    let take = prefix.len().min(buf.len());
    buf[..take].copy_from_slice(&prefix[..take]);
    n += take;
    if let Some(loc) = info.location() {
        let f = loc.file().as_bytes();
        let take = f.len().min(buf.len() - n);
        buf[n..n + take].copy_from_slice(&f[..take]);
        n += take;
    }
    n
}

fn format_location(buf: &mut [u8], file: &str, line: u32) -> usize {
    use core::fmt::Write;
    struct W<'a> { buf: &'a mut [u8], n: usize }
    impl<'a> Write for W<'a> {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            let take = s.len().min(self.buf.len() - self.n);
            self.buf[self.n..self.n + take].copy_from_slice(&s.as_bytes()[..take]);
            self.n += take;
            Ok(())
        }
    }
    let mut w = W { buf, n: 0 };
    let _ = write!(w, "at {}:{}", file, line);
    w.n
}

fn render_msg_into(buf: &mut [u8], msg: core::panic::PanicMessage<'_>) -> usize {
    use core::fmt::Write;
    struct W<'a> { buf: &'a mut [u8], n: usize }
    impl<'a> Write for W<'a> {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            let take = s.len().min(self.buf.len() - self.n);
            self.buf[self.n..self.n + take].copy_from_slice(&s.as_bytes()[..take]);
            self.n += take;
            Ok(())
        }
    }
    let mut w = W { buf, n: 0 };
    let _ = write!(w, "{}", msg);
    w.n
}
