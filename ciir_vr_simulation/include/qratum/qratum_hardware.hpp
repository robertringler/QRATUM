/**
 * @file qratum_hardware.hpp
 * @brief QRATUM hardware abstraction: maps tensor ops to virtualized GPU/CPU cores.
 *
 * QRATUM Subsystem Mappings:
 *   HCAL  → hardware constraint injection Π_C
 *   qstack → constraint weighting λ_i (priority scheduling)
 *   qnx   → distributed evolution executor
 *
 * Tracks device-level metrics for the performance dashboard:
 *   - Step latency
 *   - Tensor norms
 *   - Gradient magnitudes
 *   - Convergence rates
 *   - Memory utilization
 */

#pragma once

#include "ciir/ciir_state.hpp"
#include "quasim/quasim_tensor.hpp"

#include <cstdint>
#include <vector>
#include <string>
#include <chrono>

namespace ciir_sim {

/**
 * @brief Per-step metrics snapshot for dashboard display.
 */
struct HardwareMetrics {
    uint32_t step;
    float    loss;
    float    grad_norm;
    float    step_time_ms;
    float    cumulative_time_ms;
    float    throughput_steps_per_s;
    float    memory_usage_mb;
    std::vector<float> constraint_violations;
    std::vector<float> observer_expectations;
};

/**
 * @brief QRATUM hardware abstraction layer.
 *
 * Virtualizes hardware layers and provides:
 * - Hardware-aware constraint projection (HCAL bridge)
 * - Precision enforcement (fp32/fp64)
 * - Memory budget management
 * - Per-step performance metrics collection
 */
class QRATUMHardware {
public:
    /**
     * @brief Construct hardware abstraction.
     * @param precision Device precision mode.
     * @param memory_budget_mb Available device memory in MB.
     */
    QRATUMHardware(Precision precision = Precision::FP32,
                   float memory_budget_mb = 1024.0f);

    // ---- HCAL bridge: hardware constraint injection ----

    /// Apply hardware-aware constraint projection.
    void inject_constraints(std::vector<Matrix>& rho_batch) const;

    /// Maximum batch size within memory budget.
    [[nodiscard]] uint32_t max_batch_size(uint32_t dim) const;

    /// Estimated memory for a batch of density matrices.
    [[nodiscard]] float state_memory_mb(uint32_t batch_size, uint32_t dim) const;

    // ---- Metrics recording ----

    /// Record metrics for a completed step.
    void record_step(uint32_t step, float loss, float grad_norm,
                     const std::vector<float>& constraint_violations,
                     const std::vector<float>& observer_expectations,
                     float step_time_ms);

    /// Access metrics history.
    [[nodiscard]] const std::vector<HardwareMetrics>& metrics_log() const { return metrics_; }

    /// Clear metrics history.
    void clear_metrics() { metrics_.clear(); total_time_ms_ = 0.0f; }

    // ---- Device info ----
    [[nodiscard]] Precision precision() const { return precision_; }
    [[nodiscard]] float memory_budget_mb() const { return memory_budget_mb_; }
    [[nodiscard]] float total_time_ms() const { return total_time_ms_; }

    struct DeviceSummary {
        std::string precision_str;
        float memory_budget_mb;
        uint32_t total_steps;
        float total_time_s;
        float avg_step_ms;
        float throughput;
    };
    [[nodiscard]] DeviceSummary summary() const;

private:
    Precision precision_;
    float memory_budget_mb_;
    float total_time_ms_ = 0.0f;
    std::vector<HardwareMetrics> metrics_;
};

} // namespace ciir_sim
