// SPDX-License-Identifier: Apache-2.0
#include "quasim.hpp"
#include <cassert>
#include <complex>
#include <iostream>

int main()
{
    quasim::Runtime runtime;
    quasim::TensorJob job{{2, 2}, {1.0, 2.0, 3.0, 4.0}, "test"};
    runtime.submit(job);
    auto outputs = runtime.contract_all();
    assert(outputs.size() == 1);
    std::cout << "Average latency: " << runtime.telemetry().average_latency() << std::endl;
    return 0;
}
