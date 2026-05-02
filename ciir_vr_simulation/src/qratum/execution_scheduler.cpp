/**
 * @file execution_scheduler.cpp
 * @brief Multi-threaded priority task scheduler implementation.
 */

#include "qratum/execution_scheduler.hpp"

#include <algorithm>

namespace ciir_sim {

ExecutionScheduler::ExecutionScheduler(uint32_t n_threads)
    : n_threads_(n_threads == 0
          ? std::max(1u, std::thread::hardware_concurrency())
          : n_threads) {}

ExecutionScheduler::~ExecutionScheduler() {
    stop();
}

void ExecutionScheduler::start() {
    if (running_.load()) return;
    running_.store(true);
    workers_.reserve(n_threads_);
    for (uint32_t i = 0; i < n_threads_; ++i) {
        workers_.emplace_back(&ExecutionScheduler::worker_loop, this);
    }
}

void ExecutionScheduler::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_.load()) return;
        running_.store(false);
    }
    cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
    workers_.clear();
}

void ExecutionScheduler::submit(Task task) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(task));
    }
    cv_.notify_one();
}

std::future<void> ExecutionScheduler::submit_async(Task task) {
    auto pt = std::make_shared<std::packaged_task<void()>>(std::move(task.execute));
    auto future = pt->get_future();
    task.execute = [pt]() { (*pt)(); };
    submit(std::move(task));
    return future;
}

void ExecutionScheduler::wait_all() {
    std::unique_lock<std::mutex> lock(mutex_);
    idle_cv_.wait(lock, [this]() {
        return queue_.empty() && active_.load() == 0;
    });
}

uint32_t ExecutionScheduler::pending_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<uint32_t>(queue_.size());
}

void ExecutionScheduler::worker_loop() {
    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() {
                return !queue_.empty() || !running_.load();
            });
            if (!running_.load() && queue_.empty()) return;
            task = std::move(const_cast<Task&>(queue_.top()));
            queue_.pop();
            active_.fetch_add(1);
        }

        if (task.execute) {
            task.execute();
        }

        completed_.fetch_add(1);
        active_.fetch_sub(1);
        idle_cv_.notify_all();
    }
}

} // namespace ciir_sim
