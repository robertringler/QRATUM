//! Task execution with WCET checking

use crate::error::{CoreError, Result};
use crate::scheduler::{SchedulableTask, Scheduler};
use crate::time::{MonotonicClock, Timestamp};
use std::sync::Arc;

/// Task executor with WCET checking
#[derive(Debug)]
pub struct Executor {
    /// Scheduler
    scheduler: Scheduler,
    /// Clock
    clock: Arc<dyn MonotonicClock>,
    /// Enable WCET checking
    check_wcet: bool,
}

impl Executor {
    /// Create a new executor
    pub fn new(clock: Arc<dyn MonotonicClock>, check_wcet: bool) -> Self {
        Self {
            scheduler: Scheduler::new(),
            clock,
            check_wcet,
        }
    }

    /// Get a reference to the scheduler
    pub fn scheduler_mut(&mut self) -> &mut Scheduler {
        &mut self.scheduler
    }

    /// Execute the next task
    pub async fn execute_next<F, Fut>(&mut self, f: F) -> Result<Option<u64>>
    where
        F: FnOnce(SchedulableTask) -> Fut,
        Fut: std::future::Future<Output = Result<()>>,
    {
        let task = match self.scheduler.next() {
            Some(task) => task,
            None => return Ok(None),
        };

        let start_time = self.clock.now();

        // Check if we've already missed the deadline
        if start_time > task.deadline.0 {
            return Err(CoreError::Executor(format!(
                "Task {} missed deadline: start={} deadline={}",
                task.id,
                start_time.as_nanos(),
                task.deadline.0.as_nanos()
            )));
        }

        // Execute the task
        f(task.clone()).await?;

        let end_time = self.clock.now();
        let execution_time = end_time.as_nanos() - start_time.as_nanos();

        // Check WCET
        if self.check_wcet && execution_time > task.wcet_ns {
            return Err(CoreError::Executor(format!(
                "Task {} exceeded WCET: {}ns > {}ns",
                task.id, execution_time, task.wcet_ns
            )));
        }

        Ok(Some(task.id))
    }

    /// Get the number of pending tasks
    pub fn pending(&self) -> usize {
        self.scheduler.pending()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::{Deadline, Priority};
    use crate::time::SimulatedClock;

    #[tokio::test]
    async fn test_executor() {
        let clock = Arc::new(SimulatedClock::new());
        let mut executor = Executor::new(clock.clone(), false);

        // Schedule a task
        executor
            .scheduler_mut()
            .schedule(
                Priority(1),
                Deadline(Timestamp::new(1000)),
                100,
            )
            .unwrap();

        // Execute the task
        let result = executor
            .execute_next(|_task| async { Ok(()) })
            .await
            .unwrap();

        assert_eq!(result, Some(0));
        assert_eq!(executor.pending(), 0);
    }
}
