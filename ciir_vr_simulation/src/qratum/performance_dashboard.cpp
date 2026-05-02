/**
 * @file performance_dashboard.cpp
 * @brief Text-based performance dashboard (no Dear ImGui dependency).
 */

#include "qratum/performance_dashboard.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>

namespace ciir_sim {

// ---- TimeSeriesBuffer ----

float TimeSeriesBuffer::min_val() const {
    if (data_.empty()) return 0.0f;
    return *std::min_element(data_.begin(), data_.end());
}

float TimeSeriesBuffer::max_val() const {
    if (data_.empty()) return 0.0f;
    return *std::max_element(data_.begin(), data_.end());
}

// ---- PerformanceDashboard ----

PerformanceDashboard::PerformanceDashboard() = default;

void PerformanceDashboard::update(const HardwareMetrics& metrics,
                                   const ContractionLog* contraction) {
    latest_metrics_ = metrics;
    ++total_updates_;

    loss_history_.push(metrics.loss);
    grad_norm_history_.push(metrics.grad_norm);
    step_time_history_.push(metrics.step_time_ms);
    if (metrics.step_time_ms > 0.0f) {
        fps_history_.push(1000.0f / metrics.step_time_ms);
    }

    // Expand constraint histories if needed
    while (constraint_histories_.size() < metrics.constraint_violations.size()) {
        constraint_histories_.emplace_back(500);
    }
    for (size_t i = 0; i < metrics.constraint_violations.size(); ++i) {
        constraint_histories_[i].push(metrics.constraint_violations[i]);
    }

    (void)contraction; // reserved for future use
}

void PerformanceDashboard::render() const {
    std::printf("╔══════════════════════════════════════════╗\n");
    std::printf("║       QRATUM Performance Dashboard      ║\n");
    std::printf("╚══════════════════════════════════════════╝\n");

    if (show_loss_plot)     render_loss_panel();
    if (show_gradient_plot) render_gradient_panel();
    if (show_constraints)   render_constraint_panel();
    if (show_timing)        render_timing_panel();
    if (show_hardware)      render_hardware_panel();

    std::printf("──────────────────────────────────────────\n\n");
}

void PerformanceDashboard::render_vr_hud() const {
    std::printf("[HUD] Step %u | Loss %.4f | GradNorm %.4f | %.1f steps/s\n",
                latest_metrics_.step,
                latest_metrics_.loss,
                latest_metrics_.grad_norm,
                latest_metrics_.throughput_steps_per_s);
}

void PerformanceDashboard::export_json(const std::string& path) const {
    std::ofstream out(path);
    if (!out.is_open()) return;

    out << "{\n";

    // Helper lambda to write a buffer
    auto write_buffer = [&](const char* name, const TimeSeriesBuffer& buf, bool trailing_comma) {
        out << "  \"" << name << "\": [";
        const auto& d = buf.data();
        for (size_t i = 0; i < d.size(); ++i) {
            if (i > 0) out << ", ";
            out << d[i];
        }
        out << "]" << (trailing_comma ? "," : "") << "\n";
    };

    write_buffer("loss", loss_history_, true);
    write_buffer("grad_norm", grad_norm_history_, true);
    write_buffer("fps", fps_history_, true);
    write_buffer("step_time_ms", step_time_history_, true);

    out << "  \"constraints\": [\n";
    for (size_t c = 0; c < constraint_histories_.size(); ++c) {
        out << "    [";
        const auto& d = constraint_histories_[c].data();
        for (size_t i = 0; i < d.size(); ++i) {
            if (i > 0) out << ", ";
            out << d[i];
        }
        out << "]" << ((c + 1 < constraint_histories_.size()) ? "," : "") << "\n";
    }
    out << "  ],\n";

    out << "  \"total_updates\": " << total_updates_ << "\n";
    out << "}\n";
}

// ---- Individual panels ----

void PerformanceDashboard::render_loss_panel() const {
    std::printf("  Loss:      %.6f  (min %.6f, max %.6f)\n",
                loss_history_.latest(),
                loss_history_.min_val(),
                loss_history_.max_val());
}

void PerformanceDashboard::render_gradient_panel() const {
    std::printf("  GradNorm:  %.6f  (min %.6f, max %.6f)\n",
                grad_norm_history_.latest(),
                grad_norm_history_.min_val(),
                grad_norm_history_.max_val());
}

void PerformanceDashboard::render_constraint_panel() const {
    for (size_t i = 0; i < constraint_histories_.size(); ++i) {
        std::printf("  C[%zu]:      %.6f\n", i, constraint_histories_[i].latest());
    }
}

void PerformanceDashboard::render_timing_panel() const {
    std::printf("  StepTime:  %.2f ms  |  FPS: %.1f\n",
                step_time_history_.latest(),
                fps_history_.latest());
}

void PerformanceDashboard::render_hardware_panel() const {
    std::printf("  Throughput: %.1f steps/s  |  Memory: %.1f MB\n",
                latest_metrics_.throughput_steps_per_s,
                latest_metrics_.memory_usage_mb);
}

} // namespace ciir_sim
