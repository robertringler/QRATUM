//! Keyboard input emulation.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};

/// Key codes (Android KeyEvent codes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum KeyCode {
    /// Unknown key.
    Unknown = 0,
    /// Soft left.
    SoftLeft = 1,
    /// Soft right.
    SoftRight = 2,
    /// Home button.
    Home = 3,
    /// Back button.
    Back = 4,
    /// Call button.
    Call = 5,
    /// End call button.
    Endcall = 6,
    /// 0-9 keys.
    Key0 = 7,
    Key1 = 8,
    Key2 = 9,
    Key3 = 10,
    Key4 = 11,
    Key5 = 12,
    Key6 = 13,
    Key7 = 14,
    Key8 = 15,
    Key9 = 16,
    /// Star key.
    Star = 17,
    /// Pound key.
    Pound = 18,
    /// D-pad up.
    DpadUp = 19,
    /// D-pad down.
    DpadDown = 20,
    /// D-pad left.
    DpadLeft = 21,
    /// D-pad right.
    DpadRight = 22,
    /// D-pad center.
    DpadCenter = 23,
    /// Volume up.
    VolumeUp = 24,
    /// Volume down.
    VolumeDown = 25,
    /// Power button.
    Power = 26,
    /// Camera button.
    Camera = 27,
    /// Clear key.
    Clear = 28,
    /// A-Z keys.
    A = 29, B = 30, C = 31, D = 32, E = 33, F = 34, G = 35,
    H = 36, I = 37, J = 38, K = 39, L = 40, M = 41, N = 42,
    O = 43, P = 44, Q = 45, R = 46, S = 47, T = 48, U = 49,
    V = 50, W = 51, X = 52, Y = 53, Z = 54,
    /// Comma.
    Comma = 55,
    /// Period.
    Period = 56,
    /// Alt left.
    AltLeft = 57,
    /// Alt right.
    AltRight = 58,
    /// Shift left.
    ShiftLeft = 59,
    /// Shift right.
    ShiftRight = 60,
    /// Tab.
    Tab = 61,
    /// Space.
    Space = 62,
    /// Enter.
    Enter = 66,
    /// Delete (backspace).
    Del = 67,
    /// Grave (`).
    Grave = 68,
    /// Minus.
    Minus = 69,
    /// Equals.
    Equals = 70,
    /// Left bracket.
    LeftBracket = 71,
    /// Right bracket.
    RightBracket = 72,
    /// Backslash.
    Backslash = 73,
    /// Semicolon.
    Semicolon = 74,
    /// Apostrophe.
    Apostrophe = 75,
    /// Slash.
    Slash = 76,
    /// At (@).
    At = 77,
    /// Menu.
    Menu = 82,
    /// Search.
    Search = 84,
    /// Play/pause.
    MediaPlayPause = 85,
    /// Stop.
    MediaStop = 86,
    /// Next.
    MediaNext = 87,
    /// Previous.
    MediaPrevious = 88,
    /// Page up.
    PageUp = 92,
    /// Page down.
    PageDown = 93,
    /// Escape.
    Escape = 111,
    /// Forward delete.
    ForwardDel = 112,
    /// Control left.
    CtrlLeft = 113,
    /// Control right.
    CtrlRight = 114,
    /// Caps lock.
    CapsLock = 115,
    /// Scroll lock.
    ScrollLock = 116,
    /// Meta left.
    MetaLeft = 117,
    /// Meta right.
    MetaRight = 118,
    /// Function.
    Function = 119,
    /// Print screen.
    Sysrq = 120,
    /// Break.
    Break = 121,
    /// Move home.
    MoveHome = 122,
    /// Move end.
    MoveEnd = 123,
    /// Insert.
    Insert = 124,
    /// F1-F12.
    F1 = 131, F2 = 132, F3 = 133, F4 = 134, F5 = 135, F6 = 136,
    F7 = 137, F8 = 138, F9 = 139, F10 = 140, F11 = 141, F12 = 142,
    /// Numpad keys.
    NumLock = 143,
    Numpad0 = 144, Numpad1 = 145, Numpad2 = 146, Numpad3 = 147,
    Numpad4 = 148, Numpad5 = 149, Numpad6 = 150, Numpad7 = 151,
    Numpad8 = 152, Numpad9 = 153,
    NumpadDivide = 154,
    NumpadMultiply = 155,
    NumpadSubtract = 156,
    NumpadAdd = 157,
    NumpadDot = 158,
    NumpadEnter = 160,
}

/// Key event action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyAction {
    /// Key pressed down.
    Down,
    /// Key released.
    Up,
    /// Key repeated (held).
    Repeat,
}

/// A keyboard event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyEvent {
    /// Key code.
    pub key_code: KeyCode,
    /// Action.
    pub action: KeyAction,
    /// Timestamp in nanoseconds.
    pub timestamp_ns: u64,
    /// Modifier flags.
    pub modifiers: u32,
    /// Repeat count.
    pub repeat_count: u32,
}

impl KeyEvent {
    /// Create a key down event.
    pub fn down(key_code: KeyCode) -> Self {
        Self {
            key_code,
            action: KeyAction::Down,
            timestamp_ns: 0,
            modifiers: 0,
            repeat_count: 0,
        }
    }

    /// Create a key up event.
    pub fn up(key_code: KeyCode) -> Self {
        Self {
            key_code,
            action: KeyAction::Up,
            timestamp_ns: 0,
            modifiers: 0,
            repeat_count: 0,
        }
    }
}

/// Keyboard state manager.
pub struct KeyboardState {
    /// Currently pressed keys.
    pressed_keys: RwLock<HashSet<KeyCode>>,
    /// Event queue.
    events: RwLock<VecDeque<KeyEvent>>,
}

impl KeyboardState {
    /// Create a new keyboard state.
    pub fn new() -> Self {
        Self {
            pressed_keys: RwLock::new(HashSet::new()),
            events: RwLock::new(VecDeque::new()),
        }
    }

    /// Process a key down.
    pub fn key_down(&self, key_code: KeyCode) {
        let mut pressed = self.pressed_keys.write();
        if pressed.insert(key_code) {
            self.events.write().push_back(KeyEvent::down(key_code));
        }
    }

    /// Process a key up.
    pub fn key_up(&self, key_code: KeyCode) {
        let mut pressed = self.pressed_keys.write();
        if pressed.remove(&key_code) {
            self.events.write().push_back(KeyEvent::up(key_code));
        }
    }

    /// Check if a key is pressed.
    pub fn is_pressed(&self, key_code: KeyCode) -> bool {
        self.pressed_keys.read().contains(&key_code)
    }

    /// Get the next event from the queue.
    pub fn poll_event(&self) -> Option<KeyEvent> {
        self.events.write().pop_front()
    }

    /// Get all pending events.
    pub fn drain_events(&self) -> Vec<KeyEvent> {
        self.events.write().drain(..).collect()
    }

    /// Get all pressed keys.
    pub fn pressed_keys(&self) -> Vec<KeyCode> {
        self.pressed_keys.read().iter().copied().collect()
    }

    /// Check if any modifier is pressed.
    pub fn has_modifier(&self, modifier: KeyCode) -> bool {
        self.is_pressed(modifier)
    }
}

impl Default for KeyboardState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_press_release() {
        let state = KeyboardState::new();
        
        state.key_down(KeyCode::A);
        assert!(state.is_pressed(KeyCode::A));
        
        state.key_up(KeyCode::A);
        assert!(!state.is_pressed(KeyCode::A));
        
        let events = state.drain_events();
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_multiple_keys() {
        let state = KeyboardState::new();
        
        state.key_down(KeyCode::ShiftLeft);
        state.key_down(KeyCode::A);
        
        assert!(state.is_pressed(KeyCode::ShiftLeft));
        assert!(state.is_pressed(KeyCode::A));
        
        let pressed = state.pressed_keys();
        assert_eq!(pressed.len(), 2);
    }
}
