/**
 * @file simulation_engine.hpp
 * @brief Core simulation engine: orchestrates CIIR → QuASIM → QRATUM pipeline.
 *
 * The SimulationEngine owns all subsystem modules and drives the main loop:
 *
 *   1. Initialize Vulkan + OpenXR + GLFW
 *   2. Create CIIR states and QuASIM tensors on GPU
 *   3. Each frame: CIIR → QuASIM tensor map → QRATUM scheduling → Render
 *   4. Update metrics and export logs/screenshots
 *
 * Mathematical pipeline per frame:
 *   X_t ∈ R^{B×R×D} → ρ = X^T X / Tr(X^T X)
 *   L(ρ) = Σ λ_i C_i(ρ)² + Σ μ_j |⟨O_j⟩ - y_j|² + γ·Ω(ρ)
 *   ρ_{t+1} = Π_C(ρ_t - η ∇L(ρ_t))
 */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <atomic>

#include "ciir/ciir_state.hpp"
#include "ciir/ciir_observer.hpp"
#include "ciir/ciir_visualizer.hpp"
#include "quasim/quasim_tensor.hpp"
#include "quasim/tensor_runtime.hpp"
#include "quasim/tensor_visualizer.hpp"
#include "qratum/qratum_hardware.hpp"
#include "qratum/execution_scheduler.hpp"
#include "qratum/performance_dashboard.hpp"
#include "renderer/vulkan_context.hpp"
#include "renderer/vr_context.hpp"
#include "renderer/camera.hpp"
#include "renderer/mesh.hpp"

namespace ciir_sim {

/**
 * @brief Configuration for the simulation engine.
 */
struct SimulationConfig {
    // State manifold dimensions
    uint32_t rank       = 4;     ///< Ontic dimension R (D3.1)
    uint32_t rep_dim    = 8;     ///< Cognitive representation dimension D (D3.11)
    uint32_t batch_size = 4;     ///< Number of parallel states

    // Theory structure
    uint32_t n_constraints = 3;  ///< Number of constraint operators (D5.2)
    uint32_t n_observers   = 2;  ///< Number of observer operators (D3.20)

    // Evolution parameters (Ch7)
    uint32_t n_steps       = 200;    ///< Maximum evolution steps
    float    learning_rate = 0.01f;  ///< Learning rate η
    float    entropy_weight = 0.01f; ///< Regularisation γ
    float    convergence_tol = 1e-6f;

    // Rendering
    uint32_t window_width  = 1920;
    uint32_t window_height = 1080;
    bool     enable_vr     = false;
    bool     enable_validation = true;
    bool     fullscreen    = false;
    float    target_fps    = 90.0f;  ///< 90 FPS for VR

    // Export
    std::string output_dir      = "ciir_output";
    bool        auto_screenshot  = false;
    uint32_t    screenshot_interval = 50; ///< Steps between auto-screenshots
    bool        log_tensors     = true;

    uint32_t seed = 42;
};

/**
 * @brief Metrics snapshot for a single simulation step.
 */
struct StepMetrics {
    uint32_t step;
    float    total_loss;
    float    constraint_loss;
    float    observer_loss;
    float    entropy_loss;
    float    gradient_norm;
    float    step_time_ms;
    float    fps;
    std::vector<float> constraint_violations;
    std::vector<float> observer_expectations;
};

/**
 * @brief Core simulation engine orchestrating the full CIIR → QuASIM → QRATUM pipeline.
 *
 * Lifecycle:
 *   1. init()        — Create all subsystems and GPU resources
 *   2. run()         — Enter main loop (blocks until exit)
 *   3. shutdown()    — Release all resources
 *
 * Or for headless / step-by-step:
 *   1. init()
 *   2. step() in a loop
 *   3. shutdown()
 */
class SimulationEngine {
public:
    explicit SimulationEngine(const SimulationConfig& config);
    ~SimulationEngine();

    // Non-copyable, movable
    SimulationEngine(const SimulationEngine&) = delete;
    SimulationEngine& operator=(const SimulationEngine&) = delete;
    SimulationEngine(SimulationEngine&&) noexcept;
    SimulationEngine& operator=(SimulationEngine&&) noexcept;

    /**
     * @brief Initialize all subsystems: Vulkan, VR, CIIR theory, tensors.
     * @return true on success.
     */
    bool init();

    /**
     * @brief Enter the main simulation loop (blocks until window close or VR exit).
     */
    void run();

    /**
     * @brief Execute a single simulation step (for headless / external control).
     * @return Step metrics.
     */
    StepMetrics step();

    /**
     * @brief Shut down all subsystems and release GPU resources.
     */
    void shutdown();

    /**
     * @brief Take a screenshot of the current frame.
     * @param path Output file path (PNG).
     */
    void screenshot(const std::string& path) const;

    /**
     * @brief Export tensor log to file.
     * @param path Output file path (JSON).
     */
    void export_tensor_log(const std::string& path) const;

    /**
     * @brief Export all collected metrics to CSV.
     * @param path Output file path.
     */
    void export_metrics_csv(const std::string& path) const;

    // Accessors
    [[nodiscard]] bool is_running() const { return running_.load(); }
    [[nodiscard]] uint32_t current_step() const { return current_step_; }
    [[nodiscard]] const SimulationConfig& config() const { return config_; }
    [[nodiscard]] const std::vector<StepMetrics>& metrics_history() const { return metrics_; }

    // Interactive parameter control
    void set_learning_rate(float lr);
    void set_constraint_weight(uint32_t idx, float weight);
    void set_observer_target(uint32_t idx, float target);
    void set_entropy_weight(float gamma);
    void trigger_measurement(uint32_t observer_idx);

private:
    void init_ciir_theory();
    void init_gpu_resources();
    void init_vr();
    void process_input();
    void update_simulation();
    void render_frame();
    void update_dashboard();
    void record_metrics();

    SimulationConfig config_;
    std::atomic<bool> running_{false};
    uint32_t current_step_ = 0;

    // Subsystem modules
    std::unique_ptr<CIIRState>             ciir_state_;
    std::unique_ptr<CIIRObserver>          ciir_observer_;
    std::unique_ptr<CIIRVisualizer>        ciir_viz_;
    std::unique_ptr<QuASIMTensor>          state_tensor_;
    std::unique_ptr<TensorRuntime>         tensor_runtime_;
    std::unique_ptr<TensorVisualizer>      tensor_viz_;
    std::unique_ptr<QRATUMHardware>        hardware_;
    std::unique_ptr<ExecutionScheduler>    scheduler_;
    std::unique_ptr<PerformanceDashboard>  dashboard_;

    // Renderer
    std::unique_ptr<VulkanContext>  vulkan_ctx_;
    std::unique_ptr<VRContext>      vr_ctx_;
    std::unique_ptr<Camera>         camera_;
    std::vector<Mesh>               scene_meshes_;

    // Metrics
    std::vector<StepMetrics> metrics_;

    // Interactive state
    std::vector<float> constraint_weights_;
    std::vector<float> observer_targets_;
};

} // namespace ciir_sim
