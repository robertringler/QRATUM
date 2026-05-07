//! PIT (8254) — channel 0 in mode 3 (square wave) at ~100 Hz (Agent 4).
//!
//! Frequency formula:  divisor = 1_193_182 / hz
//! At hz=100, divisor = 11932 (≈ 0.01 s tick, suitable for a preemption beat).

use x86_64::instructions::port::Port;

const CHANNEL0_DATA: u16 = 0x40;
const COMMAND:       u16 = 0x43;

pub const HZ: u32 = 100;

pub fn init() {
    let divisor: u16 = (1_193_182u32 / HZ) as u16;

    let mut cmd = Port::<u8>::new(COMMAND);
    let mut data = Port::<u8>::new(CHANNEL0_DATA);

    unsafe {
        // Channel 0, lo+hi byte, mode 3 (square wave), binary
        cmd.write(0b00_11_011_0);
        data.write((divisor & 0xFF) as u8);
        data.write((divisor >> 8) as u8);
    }
}
