// SPDX-License-Identifier: Apache-2.0
#include "quasim.hpp"
#include <chrono>
#include <random>

namespace quasim {

Runtime::Runtime(Config config)
    : scheduler_(std::move(config)), telemetry_() {}

void Runtime::submit(const TensorJob &job)
{
    scheduler_.submit(job);
}

std::vector<std::complex<double>> Runtime::contract_all()
{
    std::vector<std::complex<double>> outputs;
    std::mt19937_64 rng{0x42};
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    auto pending = scheduler_.pending();
    outputs.reserve(pending);
    for (std::size_t i = 0; i < pending; ++i) {
        auto start = std::chrono::steady_clock::now();
        std::complex<double> accum{0.0, 0.0};
        for (int sample = 0; sample < 32; ++sample) {
            accum += std::complex<double>(dist(rng), dist(rng));
        }
        auto stop = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = stop - start;
        telemetry_.record_latency(elapsed.count());
        outputs.push_back(accum);
    }
    return outputs;
}

} // namespace quasim
