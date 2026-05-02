/**
 * @file qratum_hardware.cpp
 * @brief QRATUM hardware abstraction implementation.
 */

#include "qratum/qratum_hardware.hpp"

#include <algorithm>
#include <cmath>

namespace ciir_sim {

QRATUMHardware::QRATUMHardware(Precision precision, float memory_budget_mb)
    : precision_(precision), memory_budget_mb_(memory_budget_mb) {}

void QRATUMHardware::inject_constraints(std::vector<Matrix>& rho_batch) const {
    for (auto& rho : rho_batch) {
        // Symmetrise to enforce Hermiticity
        rho.symmetrise();

        // Eigenvalue clipping: clamp negative eigenvalues to zero
        auto eigs = rho.eigenvalues();
        bool needs_fix = false;
        for (float e : eigs) {
            if (e < 0.0f) { needs_fix = true; break; }
        }
        if (needs_fix) {
            // Simple fix: add a small positive shift to make PSD
            float min_eig = *std::min_element(eigs.begin(), eigs.end());
            if (min_eig < 0.0f) {
                for (uint32_t i = 0; i < rho.dim; ++i) {
                    rho(i, i) -= min_eig;
                }
            }
        }

        // Trace normalization: Tr(ρ) = 1
        float tr = rho.trace();
        if (tr > 1e-12f) {
            rho.scale(1.0f / tr);
        }
    }
}

uint32_t QRATUMHardware::max_batch_size(uint32_t dim) const {
    float per_matrix_mb = state_memory_mb(1, dim);
    if (per_matrix_mb <= 0.0f) return 1;
    return static_cast<uint32_t>(memory_budget_mb_ / per_matrix_mb);
}

float QRATUMHardware::state_memory_mb(uint32_t batch_size, uint32_t dim) const {
    size_t elements = static_cast<size_t>(batch_size) * dim * dim;
    size_t bytes_per_elem = (precision_ == Precision::FP64) ? 8 : 4;
    return static_cast<float>(elements * bytes_per_elem) / (1024.0f * 1024.0f);
}

void QRATUMHardware::record_step(uint32_t step, float loss, float grad_norm,
                                  const std::vector<float>& constraint_violations,
                                  const std::vector<float>& observer_expectations,
                                  float step_time_ms) {
    total_time_ms_ += step_time_ms;
    float throughput = (total_time_ms_ > 0.0f)
        ? (static_cast<float>(step + 1) / (total_time_ms_ * 0.001f))
        : 0.0f;

    HardwareMetrics m{};
    m.step = step;
    m.loss = loss;
    m.grad_norm = grad_norm;
    m.step_time_ms = step_time_ms;
    m.cumulative_time_ms = total_time_ms_;
    m.throughput_steps_per_s = throughput;
    m.memory_usage_mb = 0.0f; // estimated externally
    m.constraint_violations = constraint_violations;
    m.observer_expectations = observer_expectations;
    metrics_.push_back(std::move(m));
}

QRATUMHardware::DeviceSummary QRATUMHardware::summary() const {
    DeviceSummary s{};
    s.precision_str = (precision_ == Precision::FP64) ? "FP64" : "FP32";
    s.memory_budget_mb = memory_budget_mb_;
    s.total_steps = static_cast<uint32_t>(metrics_.size());
    s.total_time_s = total_time_ms_ * 0.001f;
    s.avg_step_ms = (s.total_steps > 0)
        ? (total_time_ms_ / static_cast<float>(s.total_steps))
        : 0.0f;
    s.throughput = (s.total_time_s > 0.0f)
        ? (static_cast<float>(s.total_steps) / s.total_time_s)
        : 0.0f;
    return s;
}

} // namespace ciir_sim
