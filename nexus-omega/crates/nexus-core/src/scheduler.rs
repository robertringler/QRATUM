//! Task scheduling primitives

use crate::error::{CoreError, Result};
use crate::time::Timestamp;
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Task priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Priority(pub u8);

/// Deadline for a task
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Deadline(pub Timestamp);

impl PartialOrd for Deadline {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Deadline {
    fn cmp(&self, other: &Self) -> Ordering {
        // Earlier deadlines have higher priority (reverse order)
        other.0.cmp(&self.0)
    }
}

/// Schedulable task
#[derive(Debug, Clone)]
pub struct SchedulableTask {
    /// Task ID
    pub id: u64,
    /// Task priority
    pub priority: Priority,
    /// Task deadline
    pub deadline: Deadline,
    /// Worst-case execution time in nanoseconds
    pub wcet_ns: u64,
}

impl PartialEq for SchedulableTask {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for SchedulableTask {}

impl PartialOrd for SchedulableTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SchedulableTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // EDF: order by deadline, then priority, then ID
        self.deadline
            .cmp(&other.deadline)
            .then(self.priority.cmp(&other.priority))
            .then(self.id.cmp(&other.id))
    }
}

/// Earliest Deadline First scheduler
#[derive(Debug)]
pub struct Scheduler {
    /// Task queue (min-heap by deadline)
    queue: BinaryHeap<SchedulableTask>,
    /// Next task ID
    next_id: u64,
}

impl Scheduler {
    /// Create a new scheduler
    pub fn new() -> Self {
        Self {
            queue: BinaryHeap::new(),
            next_id: 0,
        }
    }

    /// Schedule a task
    pub fn schedule(
        &mut self,
        priority: Priority,
        deadline: Deadline,
        wcet_ns: u64,
    ) -> Result<u64> {
        let id = self.next_id;
        self.next_id += 1;

        let task = SchedulableTask {
            id,
            priority,
            deadline,
            wcet_ns,
        };

        self.queue.push(task);
        Ok(id)
    }

    /// Get the next task to execute
    pub fn next(&mut self) -> Option<SchedulableTask> {
        self.queue.pop()
    }

    /// Get the number of pending tasks
    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    /// Check if the scheduler is empty
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_edf() {
        let mut scheduler = Scheduler::new();

        // Schedule tasks with different deadlines
        scheduler
            .schedule(
                Priority(1),
                Deadline(Timestamp::new(1000)),
                100,
            )
            .unwrap();
        scheduler
            .schedule(
                Priority(1),
                Deadline(Timestamp::new(500)),
                100,
            )
            .unwrap();
        scheduler
            .schedule(
                Priority(1),
                Deadline(Timestamp::new(1500)),
                100,
            )
            .unwrap();

        // Tasks should be executed in deadline order
        let task1 = scheduler.next().unwrap();
        assert_eq!(task1.deadline.0.as_nanos(), 500);

        let task2 = scheduler.next().unwrap();
        assert_eq!(task2.deadline.0.as_nanos(), 1000);

        let task3 = scheduler.next().unwrap();
        assert_eq!(task3.deadline.0.as_nanos(), 1500);

        assert!(scheduler.is_empty());
    }
}
