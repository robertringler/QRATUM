//! Touch screen emulation.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Touch event type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TouchEventType {
    /// Touch down (new touch).
    Down,
    /// Touch move.
    Move,
    /// Touch up (release).
    Up,
    /// Touch cancelled.
    Cancel,
}

/// A single touch point.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TouchPoint {
    /// Touch ID (for multi-touch tracking).
    pub id: u32,
    /// X coordinate (0.0 - 1.0, normalized).
    pub x: f32,
    /// Y coordinate (0.0 - 1.0, normalized).
    pub y: f32,
    /// Pressure (0.0 - 1.0).
    pub pressure: f32,
    /// Touch size (0.0 - 1.0).
    pub size: f32,
}

impl TouchPoint {
    /// Create a new touch point.
    pub fn new(id: u32, x: f32, y: f32) -> Self {
        Self {
            id,
            x,
            y,
            pressure: 1.0,
            size: 0.1,
        }
    }

    /// Create with pressure.
    pub fn with_pressure(mut self, pressure: f32) -> Self {
        self.pressure = pressure.clamp(0.0, 1.0);
        self
    }
}

/// A touch event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TouchEvent {
    /// Event type.
    pub event_type: TouchEventType,
    /// Timestamp in nanoseconds.
    pub timestamp_ns: u64,
    /// Touch points.
    pub points: Vec<TouchPoint>,
}

impl TouchEvent {
    /// Create a new touch event.
    pub fn new(event_type: TouchEventType, points: Vec<TouchPoint>) -> Self {
        Self {
            event_type,
            timestamp_ns: 0,
            points,
        }
    }

    /// Create a touch down event.
    pub fn down(point: TouchPoint) -> Self {
        Self::new(TouchEventType::Down, vec![point])
    }

    /// Create a touch move event.
    pub fn move_event(point: TouchPoint) -> Self {
        Self::new(TouchEventType::Move, vec![point])
    }

    /// Create a touch up event.
    pub fn up(point: TouchPoint) -> Self {
        Self::new(TouchEventType::Up, vec![point])
    }
}

/// Touch screen state manager.
pub struct TouchState {
    /// Currently active touch points.
    active_points: RwLock<Vec<TouchPoint>>,
    /// Event queue.
    events: RwLock<VecDeque<TouchEvent>>,
    /// Next touch ID.
    next_id: RwLock<u32>,
    /// Screen width.
    width: u32,
    /// Screen height.
    height: u32,
}

impl TouchState {
    /// Create a new touch state.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            active_points: RwLock::new(Vec::new()),
            events: RwLock::new(VecDeque::new()),
            next_id: RwLock::new(0),
            width,
            height,
        }
    }

    /// Process a touch down at screen coordinates.
    pub fn touch_down(&self, x: i32, y: i32) -> u32 {
        let mut next = self.next_id.write();
        let id = *next;
        *next += 1;

        let point = TouchPoint::new(
            id,
            x as f32 / self.width as f32,
            y as f32 / self.height as f32,
        );

        self.active_points.write().push(point);
        self.events.write().push_back(TouchEvent::down(point));

        id
    }

    /// Process a touch move.
    pub fn touch_move(&self, id: u32, x: i32, y: i32) {
        let mut points = self.active_points.write();
        if let Some(point) = points.iter_mut().find(|p| p.id == id) {
            point.x = x as f32 / self.width as f32;
            point.y = y as f32 / self.height as f32;
            self.events.write().push_back(TouchEvent::move_event(*point));
        }
    }

    /// Process a touch up.
    pub fn touch_up(&self, id: u32) {
        let mut points = self.active_points.write();
        if let Some(pos) = points.iter().position(|p| p.id == id) {
            let point = points.remove(pos);
            self.events.write().push_back(TouchEvent::up(point));
        }
    }

    /// Get the next event from the queue.
    pub fn poll_event(&self) -> Option<TouchEvent> {
        self.events.write().pop_front()
    }

    /// Get all pending events.
    pub fn drain_events(&self) -> Vec<TouchEvent> {
        self.events.write().drain(..).collect()
    }

    /// Get active touch points.
    pub fn active_points(&self) -> Vec<TouchPoint> {
        self.active_points.read().clone()
    }

    /// Get the number of active touches.
    pub fn touch_count(&self) -> usize {
        self.active_points.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_touch_down_up() {
        let state = TouchState::new(1080, 1920);
        
        let id = state.touch_down(540, 960);
        assert_eq!(state.touch_count(), 1);
        
        state.touch_up(id);
        assert_eq!(state.touch_count(), 0);
        
        // Should have 2 events: down and up
        let events = state.drain_events();
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_multi_touch() {
        let state = TouchState::new(1080, 1920);
        
        let id1 = state.touch_down(100, 100);
        let id2 = state.touch_down(200, 200);
        
        assert_eq!(state.touch_count(), 2);
        
        state.touch_up(id1);
        assert_eq!(state.touch_count(), 1);
    }
}
