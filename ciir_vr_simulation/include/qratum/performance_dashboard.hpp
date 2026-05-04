/**
 * @file performance_dashboard.hpp
 * @brief Real-time performance dashboard with Dear ImGui integration.
 *
 * Displays:
 *   - FPS counter and frame timing
 *   - Loss convergence plot
 *   - Constraint violation norms
 *   - Gradient magnitude
 *   - Tensor norms
 *   - Hardware latency and throughput
 *   - VR HUD overlay (when enabled)
 */

#pragma once

#include "qratum/qratum_hardware.hpp"
#include "quasim/tensor_runtime.hpp"

#include <cstdint>
#include <vector>
#include <string>
#include <deque>
#include <array>

namespace ciir_sim {

/**
 * @brief Ring buffer for time-series data (used for live plots).
 */
class TimeSeriesBuffer {
public:
    explicit TimeSeriesBuffer(size_t capacity = 500)
        : capacity_(capacity) {}

    void push(float value) {
        if (data_.size() >= capacity_) data_.pop_front();
        data_.push_back(value);
    }

    [[nodiscard]] const std::deque<float>& data() const { return data_; }
    [[nodiscard]] size_t size() const { return data_.size(); }
    [[nodiscard]] float latest() const { return data_.empty() ? 0.0f : data_.back(); }
    [[nodiscard]] float min_val() const;
    [[nodiscard]] float max_val() const;

    void clear() { data_.clear(); }

private:
    size_t capacity_;
    std::deque<float> data_;
};

/**
 * @brief Performance dashboard that generates Dear ImGui draw commands.
 *
 * Call update() each frame with the latest metrics, then render()
 * to emit ImGui draw commands (requires ImGui context to be active).
 *
 * In VR mode, the dashboard is rendered to a texture that's composited
 * as a HUD overlay in the headset view.
 */
class PerformanceDashboard {
public:
    PerformanceDashboard();

    /**
     * @brief Update dashboard with latest metrics.
     * @param metrics Hardware metrics from the current step.
     * @param contraction Optional tensor contraction log entry.
     */
    void update(const HardwareMetrics& metrics,
                const ContractionLog* contraction = nullptr);

    /**
     * @brief Render the dashboard using Dear ImGui.
     *
     * Must be called between ImGui::NewFrame() and ImGui::Render().
     * If ImGui is not available, this is a no-op.
     */
    void render() const;

    /**
     * @brief Render minimal VR HUD overlay.
     *
     * Shows only the most critical metrics (FPS, loss, gradient norm)
     * in a compact format suitable for VR headset display.
     */
    void render_vr_hud() const;

    /**
     * @brief Export dashboard data to JSON.
     * @param path Output file path.
     */
    void export_json(const std::string& path) const;

    // ---- Configuration ----
    bool show_loss_plot     = true;
    bool show_gradient_plot = true;
    bool show_constraints   = true;
    bool show_timing        = true;
    bool show_hardware      = true;
    float plot_height       = 80.0f; ///< ImGui plot height in pixels

private:
    void render_loss_panel() const;
    void render_gradient_panel() const;
    void render_constraint_panel() const;
    void render_timing_panel() const;
    void render_hardware_panel() const;

    TimeSeriesBuffer loss_history_;
    TimeSeriesBuffer grad_norm_history_;
    TimeSeriesBuffer fps_history_;
    TimeSeriesBuffer step_time_history_;
    std::vector<TimeSeriesBuffer> constraint_histories_;

    // Latest snapshot
    HardwareMetrics latest_metrics_{};
    uint32_t total_updates_ = 0;
};

} // namespace ciir_sim
