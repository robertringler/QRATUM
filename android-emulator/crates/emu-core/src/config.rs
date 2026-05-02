//! Emulator configuration.

use serde::{Deserialize, Serialize};

/// Display configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayConfig {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel density (DPI).
    pub density: u32,
    /// Refresh rate in Hz.
    pub refresh_rate: u32,
}

impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            width: 1080,
            height: 1920,
            density: 420,
            refresh_rate: 60,
        }
    }
}

/// Device profile presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceProfile {
    /// Phone (1080x1920, 420 DPI).
    Phone,
    /// Phone HD (720x1280, 320 DPI).
    PhoneHd,
    /// Tablet (1920x1200, 240 DPI).
    Tablet,
    /// Tablet HD (2560x1600, 320 DPI).
    TabletHd,
    /// TV (1920x1080, 320 DPI).
    Tv,
    /// Custom profile.
    Custom,
}

impl Default for DeviceProfile {
    fn default() -> Self {
        Self::Phone
    }
}

impl DeviceProfile {
    /// Get the display configuration for this profile.
    pub fn display_config(self) -> DisplayConfig {
        match self {
            Self::Phone => DisplayConfig {
                width: 1080,
                height: 1920,
                density: 420,
                refresh_rate: 60,
            },
            Self::PhoneHd => DisplayConfig {
                width: 720,
                height: 1280,
                density: 320,
                refresh_rate: 60,
            },
            Self::Tablet => DisplayConfig {
                width: 1920,
                height: 1200,
                density: 240,
                refresh_rate: 60,
            },
            Self::TabletHd => DisplayConfig {
                width: 2560,
                height: 1600,
                density: 320,
                refresh_rate: 60,
            },
            Self::Tv => DisplayConfig {
                width: 1920,
                height: 1080,
                density: 320,
                refresh_rate: 60,
            },
            Self::Custom => DisplayConfig::default(),
        }
    }

    /// Get the RAM size for this profile.
    pub fn ram_size(self) -> usize {
        match self {
            Self::Phone | Self::PhoneHd => 4 * 1024 * 1024 * 1024, // 4 GB
            Self::Tablet | Self::TabletHd => 8 * 1024 * 1024 * 1024, // 8 GB
            Self::Tv => 2 * 1024 * 1024 * 1024, // 2 GB
            Self::Custom => 4 * 1024 * 1024 * 1024,
        }
    }
}

/// Main emulator configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmulatorConfig {
    /// Device profile.
    pub profile: DeviceProfile,
    /// Display configuration.
    pub display: DisplayConfig,
    /// RAM size in bytes.
    pub ram_size: usize,
    /// Number of CPU cores.
    pub cpu_cores: u32,
    /// Enable hardware acceleration.
    pub hw_accel: bool,
    /// Enable audio.
    pub audio_enabled: bool,
    /// Enable network.
    pub network_enabled: bool,
    /// Snapshot path (for save/restore).
    pub snapshot_path: Option<String>,
    /// System image path.
    pub system_image: Option<String>,
    /// User data path.
    pub data_path: Option<String>,
    /// Enable deterministic mode.
    pub deterministic: bool,
    /// Enable debug mode.
    pub debug: bool,
    /// Headless mode (no display).
    pub headless: bool,
}

impl Default for EmulatorConfig {
    fn default() -> Self {
        let profile = DeviceProfile::default();
        Self {
            profile,
            display: profile.display_config(),
            ram_size: profile.ram_size(),
            cpu_cores: 4,
            hw_accel: true,
            audio_enabled: true,
            network_enabled: true,
            snapshot_path: None,
            system_image: None,
            data_path: None,
            deterministic: false,
            debug: false,
            headless: false,
        }
    }
}

impl EmulatorConfig {
    /// Create a configuration from a device profile.
    pub fn from_profile(profile: DeviceProfile) -> Self {
        Self {
            profile,
            display: profile.display_config(),
            ram_size: profile.ram_size(),
            ..Default::default()
        }
    }

    /// Set the display configuration.
    pub fn with_display(mut self, display: DisplayConfig) -> Self {
        self.display = display;
        self.profile = DeviceProfile::Custom;
        self
    }

    /// Set the RAM size.
    pub fn with_ram(mut self, size: usize) -> Self {
        self.ram_size = size;
        self
    }

    /// Set CPU core count.
    pub fn with_cpu_cores(mut self, cores: u32) -> Self {
        self.cpu_cores = cores;
        self
    }

    /// Enable/disable hardware acceleration.
    pub fn with_hw_accel(mut self, enabled: bool) -> Self {
        self.hw_accel = enabled;
        self
    }

    /// Enable headless mode.
    pub fn headless(mut self) -> Self {
        self.headless = true;
        self
    }

    /// Enable debug mode.
    pub fn debug(mut self) -> Self {
        self.debug = true;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.ram_size < 512 * 1024 * 1024 {
            return Err("RAM size must be at least 512 MB".to_string());
        }
        if self.cpu_cores == 0 {
            return Err("CPU cores must be at least 1".to_string());
        }
        if self.display.width == 0 || self.display.height == 0 {
            return Err("Display dimensions must be non-zero".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EmulatorConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_profile_config() {
        let config = EmulatorConfig::from_profile(DeviceProfile::Tablet);
        assert_eq!(config.display.width, 1920);
        assert_eq!(config.display.height, 1200);
    }

    #[test]
    fn test_validation() {
        let mut config = EmulatorConfig::default();
        config.cpu_cores = 0;
        assert!(config.validate().is_err());
    }
}
