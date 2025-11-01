// SPDX-License-Identifier: Apache-2.0
#include "quasim.hpp"

namespace quasim {

Telemetry::Telemetry() : samples_() {}

void Telemetry::record_latency(double ms)
{
    samples_.push_back(ms);
}

double Telemetry::average_latency() const
{
    if (samples_.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (double sample : samples_) {
        total += sample;
    }
    return total / static_cast<double>(samples_.size());
}

} // namespace quasim
