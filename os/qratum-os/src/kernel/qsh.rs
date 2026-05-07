//! qsh — kernel serial REPL (Agent 11).
//!
//! P0.1 commands:
//!   help              list commands
//!   ping              "qratum alive"
//!   status            tick + heap + sched + heartbeat counters
//!   intent <name>     submit a console intent through the kernel arbiter
//!   denytest          submit an intentionally-malformed intent
//!   privtest          attempt a console intent while pretending to hold System
//!   syscall <name>    invoke `int 0x80` syscall gate
//!   tasks             dump scheduler task table
//!   yield             voluntary yield to scheduler
//!   audit [n]         dump last n audit frames (default 16)
//!   ticks             show timer tick + context switch counters
//!
//! The shell yields to the scheduler on every input wait so the heartbeat
//! task makes progress while qsh idles at the prompt.

use core::fmt::Write;

use super::audit::{self, FrameKind};
use super::interrupts;
use super::heartbeat;
use super::intent;
use super::sched;
use super::serial;
use super::syscall;
use super::tick;

const PROMPT: &str = "qsh> ";
const LINE_CAP: usize = 256;

pub fn run() -> ! {
    let mut line: [u8; LINE_CAP] = [0; LINE_CAP];
    let mut len: usize = 0;

    serial::write_str("\r\nQRATUM shell ready (P0.1). Type `help`.\r\n");
    serial::write_str(PROMPT);

    loop {
        // Cooperative: if the timer set the resched flag while we were
        // running, yield. Otherwise spin on the UART (busy wait).
        sched::yield_if_needed();

        let b = match serial::try_read_byte() {
            Some(b) => b,
            None    => { sched::yield_if_needed(); continue; }
        };

        match b {
            b'\r' | b'\n' => {
                serial::write_str("\r\n");
                if len > 0 {
                    let cmd = core::str::from_utf8(&line[..len]).unwrap_or("");
                    let mut payload = [0u8; 62];
                    let n = cmd.as_bytes().len().min(payload.len());
                    payload[..n].copy_from_slice(&cmd.as_bytes()[..n]);
                    audit::append(FrameKind::QshCommand, &payload);
                    handle(cmd);
                }
                len = 0;
                serial::write_str(PROMPT);
            }
            0x08 | 0x7F => {
                if len > 0 { len -= 1; serial::write_str("\x08 \x08"); }
            }
            0x03 => {
                serial::write_str("^C\r\n");
                len = 0;
                serial::write_str(PROMPT);
            }
            0x20..=0x7E => {
                if len < LINE_CAP - 1 {
                    line[len] = b;
                    len += 1;
                    serial::write_byte(b);
                }
            }
            _ => {}
        }
    }
}

fn handle(cmd: &str) {
    let cmd = cmd.trim();
    if cmd.is_empty() { return; }

    let (head, rest) = match cmd.find(' ') {
        Some(i) => (&cmd[..i], cmd[i + 1..].trim_start()),
        None    => (cmd, ""),
    };

    match head {
        "help" => {
            let _ = writeln!(serial::Serial,
                "P0.1 commands:\r\n\
                 \x20\x20ping              echo aliveness\r\n\
                 \x20\x20status            tick + heap + sched + heartbeat\r\n\
                 \x20\x20intent <name>     submit a console intent through the arbiter\r\n\
                 \x20\x20denytest          submit a malformed intent (expects DENIED)\r\n\
                 \x20\x20privtest          submit Console while a System holder is active\r\n\
                 \x20\x20syscall <name>    invoke int 0x80 syscall gate\r\n\
                 \x20\x20tasks             dump scheduler task table\r\n\
                 \x20\x20yield             voluntary yield to scheduler\r\n\
                 \x20\x20ticks             timer tick + context switch counts\r\n\
                 \x20\x20audit [n]         dump last n audit frames\r\n\
                 \x20\x20help              this message");
        }
        "ping" => serial::write_str("qratum alive\r\n"),
        "status" => {
            let _ = writeln!(serial::Serial,
                "tick={} tsc={} audit={} heap_used={}B sched_switches={} heartbeat={}",
                tick::loc_now(), tick::rdtsc(), audit::count(),
                super::heap::used_bytes(),
                sched::switches(),
                heartbeat::beats());
        }
        "ticks" => {
            let _ = writeln!(serial::Serial,
                "timer_ticks={} sched_switches={} loc_tick={}",
                interrupts::timer_ticks(), sched::switches(), tick::loc_now());
        }
        "intent" => {
            if rest.is_empty() {
                serial::write_str("usage: intent <name>\r\n");
                return;
            }
            let env = intent::make_console_intent(rest);
            let r = intent::submit(&env);
            print_result(&r);
        }
        "denytest" => {
            let env = intent::make_bad_envelope();
            let r = intent::submit(&env);
            print_result(&r);
        }
        "privtest" => {
            // Force HOLDER_AUTHORITY=System(4) for one cycle by accepting a
            // System intent first, then submit a Console intent (should
            // be denied DenyConsoleDomin).
            let mut sys = intent::make_console_intent("system.holder");
            sys.authority = 4;
            let _ = intent::submit(&sys);
            let con = intent::make_console_intent("console.attempt");
            let r = intent::submit(&con);
            print_result(&r);
            intent::release_one();
        }
        "syscall" => {
            if rest.is_empty() {
                serial::write_str("usage: syscall <name>\r\n");
                return;
            }
            let env = intent::make_console_intent(rest);
            let mut out = crate::shared::ResultEnvelope {
                id: [0; 16], verdict: 0, reason: 0, holder_authority: 0,
                _pad: 0, error_code: 0, completed_ts: 0,
            };
            syscall::invoke(&env, &mut out);
            print_result(&out);
        }
        "tasks" => {
            sched::dump(|i, t| {
                let nm = core::str::from_utf8(&t.name)
                    .unwrap_or("<?>")
                    .trim_end_matches('\0');
                let _ = writeln!(serial::Serial,
                    "  [{}] state={:?} prio={} auth={} class={} rsp=0x{:016X} verdict={:?} name={}",
                    i, t.state, t.prio, t.auth_class, t.sched_class, t.rsp, t.verdict, nm);
            });
        }
        "yield" => sched::yield_now(),
        "audit" => {
            let n: usize = rest.parse().unwrap_or(16);
            audit::for_each_recent(n, |f| {
                let _ = writeln!(serial::Serial,
                    "  seq={:>5} kind=0x{:02X} loc_tick={:>5} len={}",
                    f.seq, f.kind, f.loc_tick, f.payload_len);
            });
        }
        other => {
            let _ = writeln!(serial::Serial, "unknown command: {}", other);
        }
    }
}

fn print_result(r: &crate::shared::ResultEnvelope) {
    let v = if r.verdict == 0 { "ACCEPTED" } else { "DENIED" };
    let _ = writeln!(serial::Serial,
        "[ARB] verdict={} reason={} authority={} errno={} completed_ts={}",
        v, r.reason, r.holder_authority, r.error_code, r.completed_ts);
}
