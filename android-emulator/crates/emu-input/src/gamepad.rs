//! Gamepad input emulation.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Gamepad button codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum GamepadButton {
    /// A button (bottom).
    A = 0,
    /// B button (right).
    B = 1,
    /// X button (left).
    X = 2,
    /// Y button (top).
    Y = 3,
    /// Left bumper.
    L1 = 4,
    /// Right bumper.
    R1 = 5,
    /// Left trigger (digital).
    L2 = 6,
    /// Right trigger (digital).
    R2 = 7,
    /// Select/Back.
    Select = 8,
    /// Start/Menu.
    Start = 9,
    /// Left stick press.
    L3 = 10,
    /// Right stick press.
    R3 = 11,
    /// D-pad up.
    DpadUp = 12,
    /// D-pad down.
    DpadDown = 13,
    /// D-pad left.
    DpadLeft = 14,
    /// D-pad right.
    DpadRight = 15,
    /// Home/Guide button.
    Home = 16,
}

/// Gamepad axis codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum GamepadAxis {
    /// Left stick X.
    LeftX = 0,
    /// Left stick Y.
    LeftY = 1,
    /// Right stick X.
    RightX = 2,
    /// Right stick Y.
    RightY = 3,
    /// Left trigger (analog).
    LeftTrigger = 4,
    /// Right trigger (analog).
    RightTrigger = 5,
}

/// Gamepad event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GamepadEvent {
    /// Button pressed.
    ButtonDown(GamepadButton),
    /// Button released.
    ButtonUp(GamepadButton),
    /// Axis moved.
    AxisMotion {
        /// Axis that moved.
        axis: GamepadAxis,
        /// New value (-1.0 to 1.0 for sticks, 0.0 to 1.0 for triggers).
        value: f32,
    },
    /// Gamepad connected.
    Connected,
    /// Gamepad disconnected.
    Disconnected,
}

/// Gamepad state.
#[derive(Debug, Clone)]
pub struct GamepadState {
    /// Pressed buttons (bitfield).
    buttons: u32,
    /// Axis values.
    axes: [f32; 6],
    /// Connected state.
    connected: bool,
    /// Event queue.
    events: VecDeque<GamepadEvent>,
}

impl GamepadState {
    /// Create a new gamepad state.
    pub fn new() -> Self {
        Self {
            buttons: 0,
            axes: [0.0; 6],
            connected: false,
            events: VecDeque::new(),
        }
    }

    /// Mark the gamepad as connected.
    pub fn connect(&mut self) {
        if !self.connected {
            self.connected = true;
            self.events.push_back(GamepadEvent::Connected);
        }
    }

    /// Mark the gamepad as disconnected.
    pub fn disconnect(&mut self) {
        if self.connected {
            self.connected = false;
            self.buttons = 0;
            self.axes = [0.0; 6];
            self.events.push_back(GamepadEvent::Disconnected);
        }
    }

    /// Check if connected.
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Press a button.
    pub fn button_down(&mut self, button: GamepadButton) {
        let bit = 1 << (button as u32);
        if self.buttons & bit == 0 {
            self.buttons |= bit;
            self.events.push_back(GamepadEvent::ButtonDown(button));
        }
    }

    /// Release a button.
    pub fn button_up(&mut self, button: GamepadButton) {
        let bit = 1 << (button as u32);
        if self.buttons & bit != 0 {
            self.buttons &= !bit;
            self.events.push_back(GamepadEvent::ButtonUp(button));
        }
    }

    /// Check if a button is pressed.
    pub fn is_button_pressed(&self, button: GamepadButton) -> bool {
        self.buttons & (1 << (button as u32)) != 0
    }

    /// Set an axis value.
    pub fn set_axis(&mut self, axis: GamepadAxis, value: f32) {
        let idx = axis as usize;
        let clamped = value.clamp(-1.0, 1.0);
        
        if (self.axes[idx] - clamped).abs() > 0.001 {
            self.axes[idx] = clamped;
            self.events.push_back(GamepadEvent::AxisMotion {
                axis,
                value: clamped,
            });
        }
    }

    /// Get an axis value.
    pub fn get_axis(&self, axis: GamepadAxis) -> f32 {
        self.axes[axis as usize]
    }

    /// Get the next event.
    pub fn poll_event(&mut self) -> Option<GamepadEvent> {
        self.events.pop_front()
    }

    /// Drain all events.
    pub fn drain_events(&mut self) -> Vec<GamepadEvent> {
        self.events.drain(..).collect()
    }

    /// Apply deadzone to stick values.
    pub fn apply_deadzone(value: f32, deadzone: f32) -> f32 {
        if value.abs() < deadzone {
            0.0
        } else {
            let sign = value.signum();
            let normalized = (value.abs() - deadzone) / (1.0 - deadzone);
            sign * normalized
        }
    }
}

impl Default for GamepadState {
    fn default() -> Self {
        Self::new()
    }
}

/// Gamepad manager for multiple controllers.
pub struct GamepadManager {
    /// Gamepads by index.
    gamepads: RwLock<[GamepadState; 4]>,
}

impl GamepadManager {
    /// Create a new gamepad manager.
    pub fn new() -> Self {
        Self {
            gamepads: RwLock::new([
                GamepadState::new(),
                GamepadState::new(),
                GamepadState::new(),
                GamepadState::new(),
            ]),
        }
    }

    /// Get a gamepad by index.
    pub fn get(&self, index: usize) -> Option<GamepadState> {
        let gamepads = self.gamepads.read();
        gamepads.get(index).cloned()
    }

    /// Update a gamepad.
    pub fn update<F>(&self, index: usize, f: F)
    where
        F: FnOnce(&mut GamepadState),
    {
        let mut gamepads = self.gamepads.write();
        if let Some(gamepad) = gamepads.get_mut(index) {
            f(gamepad);
        }
    }
}

impl Default for GamepadManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamepad_state() {
        let mut state = GamepadState::new();
        state.connect();
        assert!(state.is_connected());
        
        state.button_down(GamepadButton::A);
        assert!(state.is_button_pressed(GamepadButton::A));
        
        state.button_up(GamepadButton::A);
        assert!(!state.is_button_pressed(GamepadButton::A));
    }

    #[test]
    fn test_axis() {
        let mut state = GamepadState::new();
        
        state.set_axis(GamepadAxis::LeftX, 0.5);
        assert!((state.get_axis(GamepadAxis::LeftX) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_deadzone() {
        assert!((GamepadState::apply_deadzone(0.05, 0.1)).abs() < 0.001);
        assert!(GamepadState::apply_deadzone(0.5, 0.1) > 0.4);
    }
}
