// SPDX-License-Identifier: Apache-2.0
#include "quasim.hpp"
#include <stdexcept>

namespace quasim {

Scheduler::Scheduler(Config config)
    : config_(std::move(config)), queue_() {}

void Scheduler::submit(const TensorJob &job)
{
    if (job.data.empty()) {
        throw std::invalid_argument("TensorJob data must not be empty");
    }
    queue_.push_back(job);
}

std::size_t Scheduler::pending() const
{
    return queue_.size();
}

} // namespace quasim
