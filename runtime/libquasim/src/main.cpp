// SPDX-License-Identifier: Apache-2.0
#include "quasim.hpp"
#include <iostream>

int main()
{
    quasim::Config cfg;
    cfg.precision = "fp8";
    quasim::Runtime runtime(cfg);

    quasim::TensorJob job{{4, 4}, {1.0, 2.0, 3.0, 4.0}, "demo"};
    runtime.submit(job);
    auto outputs = runtime.contract_all();
    std::cout << "Processed " << outputs.size() << " tensor job(s)\n";
    std::cout << "Average latency: " << runtime.telemetry().average_latency() << " ms\n";
    return 0;
}
