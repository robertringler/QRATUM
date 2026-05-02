/**
 * @file execution_scheduler.hpp
 * @brief Multi-threaded scheduling with priority-based dispatch.
 *
 * Maps CIIR constraint weights λ_i to task priorities and manages
 * concurrent execution of evolution steps, measurements, and
 * visualization updates.
 */

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>
#include <atomic>
#include <future>

namespace ciir_sim {

/**
 * @brief Task types for the execution scheduler.
 */
enum class TaskType : uint8_t {
    EVOLUTION     = 0, ///< State evolution step
    MEASUREMENT   = 1, ///< Observer measurement
    VISUALIZATION = 2, ///< Mesh/texture update
    EXPORT        = 3, ///< Screenshot/log export
};

/**
 * @brief A schedulable task with priority.
 */
struct Task {
    std::string id;
    TaskType    type;
    float       priority = 1.0f; ///< Higher = more urgent
    std::function<void()> execute;

    /// Priority comparison for the queue (max-heap).
    bool operator<(const Task& other) const { return priority < other.priority; }
};

/**
 * @brief Multi-threaded task scheduler for CIIR simulation.
 *
 * Uses a thread pool with a priority queue. Higher-priority tasks
 * (e.g., evolution steps, measurements) execute before lower-priority
 * tasks (e.g., visualization updates, exports).
 */
class ExecutionScheduler {
public:
    /**
     * @brief Construct scheduler with given thread count.
     * @param n_threads Number of worker threads (0 = hardware concurrency).
     */
    explicit ExecutionScheduler(uint32_t n_threads = 0);
    ~ExecutionScheduler();

    // Non-copyable
    ExecutionScheduler(const ExecutionScheduler&) = delete;
    ExecutionScheduler& operator=(const ExecutionScheduler&) = delete;

    /// Start the thread pool.
    void start();

    /// Stop the thread pool (waits for pending tasks).
    void stop();

    /// Submit a task to the queue.
    void submit(Task task);

    /// Submit and get a future for completion.
    std::future<void> submit_async(Task task);

    /// Wait for all pending tasks to complete.
    void wait_all();

    /// Number of pending tasks in the queue.
    [[nodiscard]] uint32_t pending_count() const;

    /// Number of tasks completed so far.
    [[nodiscard]] uint32_t completed_count() const { return completed_.load(); }

    /// Whether the scheduler is running.
    [[nodiscard]] bool is_running() const { return running_.load(); }

private:
    void worker_loop();

    uint32_t n_threads_;
    std::vector<std::thread> workers_;
    std::priority_queue<Task> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> running_{false};
    std::atomic<uint32_t> completed_{0};
    std::atomic<uint32_t> active_{0};
    std::condition_variable idle_cv_;
};

} // namespace ciir_sim
