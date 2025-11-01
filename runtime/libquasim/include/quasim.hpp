// SPDX-License-Identifier: Apache-2.0
#ifndef QUASIM_HPP
#define QUASIM_HPP

#include <complex>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace quasim {

struct TensorJob {
    std::vector<std::size_t> shape;
    std::vector<std::complex<double>> data;
    std::string tag;
};

struct Config {
    std::string precision = "fp8";
    std::size_t max_workspace_mb = 16384;
};

class Telemetry {
public:
    Telemetry();
    void record_latency(double ms);
    [[nodiscard]] double average_latency() const;
private:
    std::vector<double> samples_;
};

class Scheduler {
public:
    explicit Scheduler(Config config);
    void submit(const TensorJob &job);
    [[nodiscard]] std::size_t pending() const;
private:
    Config config_;
    std::vector<TensorJob> queue_;
};

class Runtime {
public:
    explicit Runtime(Config config = {});
    void submit(const TensorJob &job);
    std::vector<std::complex<double>> contract_all();
    [[nodiscard]] const Telemetry &telemetry() const noexcept { return telemetry_; }
private:
    Scheduler scheduler_;
    Telemetry telemetry_;
};

} // namespace quasim

#endif
