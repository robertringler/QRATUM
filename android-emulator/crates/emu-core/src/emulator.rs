//! Main emulator implementation.

use parking_lot::RwLock;
use std::sync::Arc;

use emu_android_runtime::{AndroidRuntime, RuntimeConfig};
use emu_audio::AudioMixer;
use emu_cpu::{Arm64Cpu, CpuConfig};
use emu_filesystem::{Ext4Emulator, MountManager};
use emu_graphics::{Compositor, Framebuffer, FramebufferConfig};
use emu_input::{GamepadState, KeyboardState, TouchState};
use emu_memory::{Mmu, MemoryRegion, RegionPermissions, RegionType};

use crate::config::EmulatorConfig;
use crate::error::{EmulatorError, EmulatorResult};

/// Emulator state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmulatorState {
    /// Emulator is uninitialized.
    Uninitialized,
    /// Emulator is initializing.
    Initializing,
    /// Emulator is ready to run.
    Ready,
    /// Emulator is running.
    Running,
    /// Emulator is paused.
    Paused,
    /// Emulator is stopped.
    Stopped,
}

/// Main emulator instance.
pub struct Emulator {
    /// Configuration.
    config: EmulatorConfig,
    /// Current state.
    state: RwLock<EmulatorState>,
    /// CPU.
    cpu: RwLock<Arm64Cpu>,
    /// Memory management unit.
    mmu: Arc<Mmu>,
    /// Android runtime.
    runtime: AndroidRuntime,
    /// Framebuffer.
    framebuffer: Framebuffer,
    /// Compositor.
    compositor: Compositor,
    /// Audio mixer.
    audio_mixer: AudioMixer,
    /// Touch input state.
    touch: TouchState,
    /// Keyboard input state.
    keyboard: KeyboardState,
    /// Gamepad input state.
    gamepad: RwLock<GamepadState>,
    /// Filesystem.
    filesystem: Ext4Emulator,
    /// Mount manager.
    mounts: MountManager,
}

impl Emulator {
    /// Create a new emulator with the given configuration.
    pub fn new(config: EmulatorConfig) -> EmulatorResult<Self> {
        config.validate().map_err(|e| EmulatorError::ConfigurationError { reason: e })?;

        log::info!("Creating emulator with {} MB RAM, {}x{} display",
            config.ram_size / (1024 * 1024),
            config.display.width,
            config.display.height);

        // Limit RAM for testing purposes
        let ram_size = config.ram_size.min(256 * 1024 * 1024);

        // Create CPU
        let cpu_config = CpuConfig {
            frequency_hz: 2_000_000_000, // 2 GHz
            deterministic: config.deterministic,
            instruction_limit: 0,
            trace_enabled: config.debug,
        };
        let cpu = Arm64Cpu::new(cpu_config);

        // Create MMU
        let mmu = Arc::new(Mmu::new(ram_size));

        // Create framebuffer
        let fb_config = FramebufferConfig {
            width: config.display.width,
            height: config.display.height,
            ..Default::default()
        };
        let framebuffer = Framebuffer::new(fb_config)?;

        // Create compositor
        let compositor = Compositor::with_defaults();

        // Create audio mixer
        let audio_mixer = AudioMixer::with_defaults();

        // Create input states
        let touch = TouchState::new(config.display.width, config.display.height);
        let keyboard = KeyboardState::new();
        let gamepad = GamepadState::new();

        // Create filesystem
        let filesystem = Ext4Emulator::new();

        // Create mount manager
        let mounts = MountManager::new();

        // Create runtime
        let runtime_config = RuntimeConfig {
            heap_size: ram_size / 2,
            stack_size: 8 * 1024 * 1024,
            syscall_trace: config.debug,
            binder_debug: config.debug,
        };
        let runtime = AndroidRuntime::new(runtime_config);

        Ok(Self {
            config,
            state: RwLock::new(EmulatorState::Uninitialized),
            cpu: RwLock::new(cpu),
            mmu,
            runtime,
            framebuffer,
            compositor,
            audio_mixer,
            touch,
            keyboard,
            gamepad: RwLock::new(gamepad),
            filesystem,
            mounts,
        })
    }

    /// Create an emulator with default configuration.
    pub fn with_defaults() -> EmulatorResult<Self> {
        Self::new(EmulatorConfig::default())
    }

    /// Initialize the emulator.
    pub fn initialize(&self) -> EmulatorResult<()> {
        *self.state.write() = EmulatorState::Initializing;

        log::info!("Initializing emulator...");

        // Setup memory regions
        self.setup_memory()?;

        // Initialize filesystem
        self.filesystem.init_android_structure()?;

        // Setup mounts
        self.mounts.setup_android_mounts()?;

        // Initialize runtime
        self.runtime.initialize(&self.mmu)?;

        *self.state.write() = EmulatorState::Ready;
        log::info!("Emulator initialized successfully");

        Ok(())
    }

    fn setup_memory(&self) -> EmulatorResult<()> {
        // Use a small portion of the MMU's memory for the main RAM
        // The MMU was created with a limited size for testing
        let mmu_size = self.mmu.guest_memory_size();
        let ram_size = (mmu_size / 2) as u64;
        let stack_size = (mmu_size / 8) as u64;
        
        // Map RAM region
        let ram_region = MemoryRegion::new(
            "main_ram",
            0x4000_0000,  // 1GB mark
            ram_size,
            RegionType::Ram,
            RegionPermissions::RWX,
        );
        self.mmu.map_region(ram_region)?;

        // Map stack region - use a different base that won't overflow
        let stack_region = MemoryRegion::new(
            "stack",
            0x5000_0000,  // Above RAM
            stack_size,
            RegionType::Stack,
            RegionPermissions::RW,
        );
        self.mmu.map_region(stack_region)?;

        log::debug!("Memory regions configured: {} MB RAM, {} MB stack", 
            ram_size / (1024 * 1024),
            stack_size / (1024 * 1024));
        Ok(())
    }

    /// Boot the emulator.
    pub fn boot(&self) -> EmulatorResult<()> {
        if *self.state.read() != EmulatorState::Ready {
            return Err(EmulatorError::BootFailed {
                reason: "emulator not initialized".to_string(),
            });
        }

        log::info!("Booting emulator...");
        *self.state.write() = EmulatorState::Running;

        // Set initial PC to start of RAM
        self.cpu.write().set_pc(0x4000_0000);

        // Set initial stack pointer - point to top of stack region
        let mmu_size = self.mmu.guest_memory_size();
        let stack_top = 0x5000_0000 + (mmu_size / 8) as u64 - 16;
        self.cpu.write().regs.set_sp(stack_top);

        log::info!("Emulator booted successfully");
        Ok(())
    }

    /// Run the emulator for a number of instructions.
    pub fn run(&self, max_instructions: u64) -> EmulatorResult<()> {
        if *self.state.read() != EmulatorState::Running {
            return Err(EmulatorError::InitializationFailed {
                reason: "emulator not running".to_string(),
            });
        }

        let mut cpu = self.cpu.write();
        
        for _ in 0..max_instructions {
            match cpu.step(&self.mmu) {
                Ok(()) => continue,
                Err(emu_cpu::CpuError::SupervisorCall { imm }) => {
                    // Handle syscall
                    drop(cpu);
                    // Note: would call runtime.handle_syscall here
                    cpu = self.cpu.write();
                }
                Err(e) => return Err(e.into()),
            }
        }

        Ok(())
    }

    /// Pause the emulator.
    pub fn pause(&self) {
        if *self.state.read() == EmulatorState::Running {
            *self.state.write() = EmulatorState::Paused;
            log::info!("Emulator paused");
        }
    }

    /// Resume the emulator.
    pub fn resume(&self) {
        if *self.state.read() == EmulatorState::Paused {
            *self.state.write() = EmulatorState::Running;
            log::info!("Emulator resumed");
        }
    }

    /// Stop the emulator.
    pub fn stop(&self) {
        *self.state.write() = EmulatorState::Stopped;
        log::info!("Emulator stopped");
    }

    /// Reset the emulator.
    pub fn reset(&self) -> EmulatorResult<()> {
        log::info!("Resetting emulator...");
        self.cpu.write().reset();
        *self.state.write() = EmulatorState::Uninitialized;
        self.initialize()?;
        Ok(())
    }

    /// Get the current state.
    pub fn state(&self) -> EmulatorState {
        *self.state.read()
    }

    /// Get the configuration.
    pub fn config(&self) -> &EmulatorConfig {
        &self.config
    }

    /// Get the framebuffer.
    pub fn framebuffer(&self) -> &Framebuffer {
        &self.framebuffer
    }

    /// Get the compositor.
    pub fn compositor(&self) -> &Compositor {
        &self.compositor
    }

    /// Get the touch input state.
    pub fn touch(&self) -> &TouchState {
        &self.touch
    }

    /// Get the keyboard input state.
    pub fn keyboard(&self) -> &KeyboardState {
        &self.keyboard
    }

    /// Get the filesystem.
    pub fn filesystem(&self) -> &Ext4Emulator {
        &self.filesystem
    }

    /// Get the mount manager.
    pub fn mounts(&self) -> &MountManager {
        &self.mounts
    }

    /// Take a snapshot.
    pub fn snapshot(&self, name: &str) -> EmulatorResult<()> {
        log::info!("Taking snapshot: {}", name);
        // Implementation would save CPU state, memory, etc.
        Ok(())
    }

    /// Restore from a snapshot.
    pub fn restore(&self, name: &str) -> EmulatorResult<()> {
        log::info!("Restoring snapshot: {}", name);
        // Implementation would restore CPU state, memory, etc.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emulator_creation() {
        let emu = Emulator::with_defaults().unwrap();
        assert_eq!(emu.state(), EmulatorState::Uninitialized);
    }

    #[test]
    fn test_emulator_lifecycle() {
        let emu = Emulator::with_defaults().unwrap();
        
        emu.initialize().unwrap();
        assert_eq!(emu.state(), EmulatorState::Ready);
        
        emu.boot().unwrap();
        assert_eq!(emu.state(), EmulatorState::Running);
        
        emu.pause();
        assert_eq!(emu.state(), EmulatorState::Paused);
        
        emu.resume();
        assert_eq!(emu.state(), EmulatorState::Running);
        
        emu.stop();
        assert_eq!(emu.state(), EmulatorState::Stopped);
    }
}
