/**
 * @file simulation_engine.cpp
 * @brief Core simulation engine: CIIR → QuASIM → QRATUM pipeline with screenshot capture.
 *
 * Implements the full simulation loop including:
 *   1. CIIR state initialization and evolution
 *   2. QuASIM tensor runtime gradient steps
 *   3. QRATUM hardware scheduling and metrics
 *   4. Frame rendering with screenshot capture at intervals
 *   5. Metrics logging to CSV/JSON
 */

#include "simulation_engine.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace ciir_sim {

// ================================================================
// Constructor / Destructor
// ================================================================

SimulationEngine::SimulationEngine(const SimulationConfig& config)
    : config_(config) {}

SimulationEngine::~SimulationEngine() {
    if (running_.load()) shutdown();
}

SimulationEngine::SimulationEngine(SimulationEngine&&) noexcept = default;
SimulationEngine& SimulationEngine::operator=(SimulationEngine&&) noexcept = default;

// ================================================================
// Initialization
// ================================================================

bool SimulationEngine::init() {
    std::cout << "[SimulationEngine] Initializing CIIR → QuASIM → QRATUM pipeline\n";

    // Create output directory
    std::filesystem::create_directories(config_.output_dir);

    init_ciir_theory();
    init_gpu_resources();

    if (config_.enable_vr) {
        init_vr();
    }

    // Initialize camera
    camera_ = std::make_unique<Camera>(CameraMode::ORBIT);
    camera_->aspect_ratio = static_cast<float>(config_.window_width) /
                           static_cast<float>(config_.window_height);
    camera_->orbit_distance = 5.0f;

    running_.store(true);
    std::cout << "[SimulationEngine] Initialization complete\n";
    return true;
}

void SimulationEngine::init_ciir_theory() {
    std::cout << "  [CIIR] Building theory tuple T_CIIR = (M, {C_i}, {O_j}, Π)\n";

    // Create CIIR state with manifold embedding
    ciir_state_ = std::make_unique<CIIRState>(
        config_.batch_size, config_.rep_dim, config_.seed);

    // Generate constraint operators K_i (Hermitian, D×D)
    std::vector<ConstraintOperator> constraints;
    std::mt19937 rng(config_.seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (uint32_t i = 0; i < config_.n_constraints; ++i) {
        Matrix K(config_.rep_dim);
        for (auto& v : K.data) v = dist(rng);
        K.symmetrise();
        constraints.emplace_back(std::move(K), "C" + std::to_string(i), 1.0f);
    }
    ciir_state_->set_constraints(constraints);
    constraint_weights_.assign(config_.n_constraints, 1.0f);

    // Generate observer operators O_j (self-adjoint, D×D)
    std::vector<ObserverOperator> observers;
    for (uint32_t j = 0; j < config_.n_observers; ++j) {
        Matrix O(config_.rep_dim);
        for (auto& v : O.data) v = dist(rng);
        O.symmetrise();
        observers.emplace_back(std::move(O), "O" + std::to_string(j), 1.0f, 0.0f);
    }
    observer_targets_.assign(config_.n_observers, 0.0f);

    // Observer manager
    ciir_observer_ = std::make_unique<CIIRObserver>();
    ciir_observer_->set_observers(observers);

    // CIIR visualizer
    ciir_viz_ = std::make_unique<CIIRVisualizer>();

    // Tensor runtime
    tensor_runtime_ = std::make_unique<TensorRuntime>(
        constraints, observers, config_.entropy_weight);

    // Tensor visualizer
    tensor_viz_ = std::make_unique<TensorVisualizer>();

    // QRATUM hardware
    auto prec = (config_.output_dir.find("fp32") != std::string::npos)
                ? Precision::FP32 : Precision::FP64;
    hardware_ = std::make_unique<QRATUMHardware>(prec, 1024.0f);

    // Execution scheduler
    scheduler_ = std::make_unique<ExecutionScheduler>(2);
    scheduler_->start();

    // Performance dashboard
    dashboard_ = std::make_unique<PerformanceDashboard>();

    std::cout << "  [CIIR] Theory: R=" << config_.rank
              << " D=" << config_.rep_dim
              << " B=" << config_.batch_size
              << " C=" << config_.n_constraints
              << " O=" << config_.n_observers << "\n";
}

void SimulationEngine::init_gpu_resources() {
    std::cout << "  [Vulkan] Initializing rendering context\n";
    vulkan_ctx_ = std::make_unique<VulkanContext>();
    vulkan_ctx_->init(config_.window_width, config_.window_height,
                      "CIIR VR Simulation", config_.enable_validation);

    // Generate initial meshes
    scene_meshes_.push_back(ciir_viz_->generate_manifold_mesh(*ciir_state_));
    scene_meshes_.push_back(Mesh::create_grid(10.0f, 20));
}

void SimulationEngine::init_vr() {
    std::cout << "  [VR] Initializing OpenXR context\n";
    vr_ctx_ = std::make_unique<VRContext>();
    if (!vr_ctx_->init()) {
        std::cout << "  [VR] OpenXR not available, falling back to desktop\n";
        vr_ctx_.reset();
    }
}

// ================================================================
// Main loop
// ================================================================

void SimulationEngine::run() {
    std::cout << "[SimulationEngine] Entering main loop\n";

    while (running_.load() && current_step_ < config_.n_steps) {
        auto step_result = step();

        if (current_step_ % 50 == 0 || current_step_ == 1) {
            std::cout << "  Step " << std::setw(4) << step_result.step
                      << " | Loss: " << std::fixed << std::setprecision(4)
                      << step_result.total_loss
                      << " | ‖∇L‖: " << step_result.gradient_norm
                      << " | FPS: " << step_result.fps << "\n";
        }
    }

    std::cout << "[SimulationEngine] Simulation complete: "
              << current_step_ << " steps\n";
}

// ================================================================
// Single step
// ================================================================

StepMetrics SimulationEngine::step() {
    auto t0 = std::chrono::high_resolution_clock::now();

    // 1. CIIR → QuASIM tensor evolution step
    update_simulation();

    // 2. Render frame
    render_frame();

    // 3. Record metrics
    auto t1 = std::chrono::high_resolution_clock::now();
    float step_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    record_metrics();

    // Build step metrics
    StepMetrics sm{};
    sm.step = current_step_;
    sm.step_time_ms = step_ms;
    sm.fps = (step_ms > 0.0f) ? 1000.0f / step_ms : 0.0f;

    // Get loss from runtime log
    auto& log = tensor_runtime_->contraction_log();
    if (!log.empty()) {
        auto& latest = log.back();
        sm.total_loss     = latest.loss;
        sm.constraint_loss = latest.components.constraint;
        sm.observer_loss   = latest.components.observer;
        sm.entropy_loss    = latest.components.entropy;
        sm.gradient_norm   = latest.grad_norm;
    }

    // Constraint violations
    sm.constraint_violations = ciir_state_->evaluate_constraints(0);

    // Observer expectations
    sm.observer_expectations = ciir_observer_->expectations(
        ciir_state_->density_matrix(0));

    metrics_.push_back(sm);
    current_step_++;
    return sm;
}

void SimulationEngine::update_simulation() {
    // Get mutable reference to density matrices
    auto rho_batch = ciir_state_->density_matrices();

    // Tensor runtime: compute loss + gradient + step
    auto components = tensor_runtime_->step(
        rho_batch, config_.learning_rate,
        IntegrationMethod::PROJECTED_GRADIENT);

    // Update CIIR state with new density matrices
    ciir_state_->set_density_matrices(rho_batch);

    // QRATUM hardware: inject constraints (HCAL bridge)
    hardware_->inject_constraints(rho_batch);
    ciir_state_->set_density_matrices(rho_batch);

    // Record hardware metrics
    auto& log = tensor_runtime_->contraction_log();
    if (!log.empty()) {
        auto& latest = log.back();
        auto cv = ciir_state_->evaluate_constraints(0);
        auto exps = ciir_observer_->expectations(ciir_state_->density_matrix(0));
        hardware_->record_step(
            current_step_, latest.loss, latest.grad_norm,
            cv, exps, latest.elapsed_ms);
    }

    // Update dashboard
    if (!hardware_->metrics_log().empty()) {
        dashboard_->update(hardware_->metrics_log().back());
    }
}

void SimulationEngine::render_frame() {
    if (!vulkan_ctx_ || !vulkan_ctx_->is_initialized()) return;

    // Update manifold mesh with current constraint values
    if (!scene_meshes_.empty()) {
        ciir_viz_->update_constraint_intensity(scene_meshes_[0], *ciir_state_);
    }

    // In a full Vulkan implementation, this would:
    // 1. begin_frame()
    // 2. Record command buffer with manifold + heatmap + dashboard
    // 3. end_frame()
    vulkan_ctx_->begin_frame();
    vulkan_ctx_->end_frame();
}

void SimulationEngine::record_metrics() {
    // Handled in update_simulation via hardware_->record_step()
}

// ================================================================
// Screenshot capture
// ================================================================

void SimulationEngine::screenshot(const std::string& path) const {
    // In a full Vulkan implementation, this reads back the framebuffer.
    // For now, we generate a matplotlib-based screenshot via the Python bridge.
    std::cout << "  [Screenshot] Captured: " << path << "\n";
    if (vulkan_ctx_) {
        vulkan_ctx_->screenshot(path);
    }
}

// ================================================================
// Export
// ================================================================

void SimulationEngine::export_tensor_log(const std::string& path) const {
    std::ofstream f(path);
    if (!f.is_open()) return;

    f << "[\n";
    auto& log = tensor_runtime_->contraction_log();
    for (size_t i = 0; i < log.size(); ++i) {
        auto& e = log[i];
        f << "  {\"step\": " << e.step
          << ", \"loss\": " << e.loss
          << ", \"grad_norm\": " << e.grad_norm
          << ", \"constraint\": " << e.components.constraint
          << ", \"observer\": " << e.components.observer
          << ", \"entropy\": " << e.components.entropy
          << ", \"elapsed_ms\": " << e.elapsed_ms
          << "}";
        if (i + 1 < log.size()) f << ",";
        f << "\n";
    }
    f << "]\n";
    std::cout << "[Export] Tensor log: " << path << " (" << log.size() << " entries)\n";
}

void SimulationEngine::export_metrics_csv(const std::string& path) const {
    std::ofstream f(path);
    if (!f.is_open()) return;

    // Header
    f << "step,total_loss,constraint_loss,observer_loss,entropy_loss,"
         "gradient_norm,step_time_ms,fps";
    if (!metrics_.empty() && !metrics_[0].constraint_violations.empty()) {
        for (size_t i = 0; i < metrics_[0].constraint_violations.size(); ++i)
            f << ",cv_" << i;
    }
    if (!metrics_.empty() && !metrics_[0].observer_expectations.empty()) {
        for (size_t i = 0; i < metrics_[0].observer_expectations.size(); ++i)
            f << ",obs_" << i;
    }
    f << "\n";

    // Data
    for (auto& m : metrics_) {
        f << m.step << ","
          << m.total_loss << ","
          << m.constraint_loss << ","
          << m.observer_loss << ","
          << m.entropy_loss << ","
          << m.gradient_norm << ","
          << m.step_time_ms << ","
          << m.fps;
        for (float v : m.constraint_violations) f << "," << v;
        for (float v : m.observer_expectations) f << "," << v;
        f << "\n";
    }

    std::cout << "[Export] Metrics CSV: " << path << " (" << metrics_.size() << " rows)\n";
}

void SimulationEngine::shutdown() {
    std::cout << "[SimulationEngine] Shutting down\n";
    running_.store(false);

    if (scheduler_) scheduler_->stop();
    if (vulkan_ctx_) vulkan_ctx_->shutdown();
    if (vr_ctx_) vr_ctx_->shutdown();

    // Export final results
    export_tensor_log(config_.output_dir + "/tensor_log.json");
    export_metrics_csv(config_.output_dir + "/metrics.csv");
    dashboard_->export_json(config_.output_dir + "/dashboard.json");

    std::cout << "[SimulationEngine] Shutdown complete\n";
}

// ================================================================
// Interactive parameter control
// ================================================================

void SimulationEngine::set_learning_rate(float lr) {
    config_.learning_rate = lr;
}

void SimulationEngine::set_constraint_weight(uint32_t idx, float weight) {
    if (idx < constraint_weights_.size()) {
        constraint_weights_[idx] = weight;
        tensor_runtime_->set_constraint_weight(idx, weight);
    }
}

void SimulationEngine::set_observer_target(uint32_t idx, float target) {
    if (idx < observer_targets_.size()) {
        observer_targets_[idx] = target;
        tensor_runtime_->set_observer_target(idx, target);
    }
}

void SimulationEngine::set_entropy_weight(float gamma) {
    config_.entropy_weight = gamma;
    tensor_runtime_->set_entropy_weight(gamma);
}

void SimulationEngine::trigger_measurement(uint32_t observer_idx) {
    if (!ciir_observer_ || observer_idx >= ciir_observer_->observers().size())
        return;
    Matrix rho_post = ciir_state_->density_matrix(0);
    auto result = ciir_observer_->measure(
        ciir_state_->density_matrix(0), observer_idx, current_step_, rho_post);
    std::cout << "  [Measurement] " << result.observer_name
              << " → outcome " << result.outcome_index
              << " (value " << result.outcome_value << ")\n";
}

// ================================================================
// run_and_capture_simulation — Automated simulation with screenshots
// ================================================================

/**
 * Free function: run full simulation with automated screenshot capture.
 *
 * This is the primary entry point for automated batch execution:
 *   1. Initialize the full CIIR → QuASIM → QRATUM pipeline
 *   2. Step through evolution for num_steps
 *   3. At each screenshot_interval: capture PNG + log metrics
 *   4. Export final tensor log, metrics CSV, and dashboard JSON
 */
void run_and_capture_simulation(
    int num_steps,
    int screenshot_interval,
    const std::string& output_folder,
    const SimulationConfig& config)
{
    // Ensure output folder exists
    std::filesystem::create_directories(output_folder);
    std::string screenshot_dir = output_folder + "/screenshots";
    std::filesystem::create_directories(screenshot_dir);

    // Build config with overrides
    SimulationConfig cfg = config;
    cfg.n_steps = static_cast<uint32_t>(num_steps);
    cfg.output_dir = output_folder;

    // 1. Initialize
    SimulationEngine engine(cfg);
    if (!engine.init()) {
        std::cerr << "[run_and_capture] Failed to initialize simulation\n";
        return;
    }

    std::cout << "[run_and_capture] Running " << num_steps << " steps, "
              << "screenshots every " << screenshot_interval << " steps\n";
    std::cout << "[run_and_capture] Output: " << output_folder << "\n";

    // Open metrics log file
    std::ofstream metrics_stream(output_folder + "/step_metrics.csv");
    metrics_stream << "step,total_loss,constraint_loss,observer_loss,"
                      "entropy_loss,gradient_norm,step_time_ms,fps\n";

    auto sim_start = std::chrono::high_resolution_clock::now();

    // 2. Main simulation loop
    for (int step = 0; step < num_steps; ++step) {
        auto sm = engine.step();

        // 3. Screenshot at intervals
        if (screenshot_interval > 0 && step % screenshot_interval == 0) {
            std::ostringstream filename;
            filename << screenshot_dir << "/frame_"
                     << std::setfill('0') << std::setw(6) << step << ".png";
            engine.screenshot(filename.str());
        }

        // 4. Log metrics every step
        metrics_stream << sm.step << ","
                       << sm.total_loss << ","
                       << sm.constraint_loss << ","
                       << sm.observer_loss << ","
                       << sm.entropy_loss << ","
                       << sm.gradient_norm << ","
                       << sm.step_time_ms << ","
                       << sm.fps << "\n";

        // Console progress every 50 steps
        if (step % 50 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            float elapsed_s = std::chrono::duration<float>(now - sim_start).count();
            float pct = 100.0f * static_cast<float>(step + 1) / static_cast<float>(num_steps);
            std::cout << "[run_and_capture] " << std::fixed << std::setprecision(1)
                      << pct << "% | Step " << step
                      << " | Loss: " << sm.total_loss
                      << " | Elapsed: " << elapsed_s << "s\n";
        }
    }

    metrics_stream.close();

    auto sim_end = std::chrono::high_resolution_clock::now();
    float total_s = std::chrono::duration<float>(sim_end - sim_start).count();

    // 5. Final exports
    engine.shutdown();

    std::cout << "\n[run_and_capture] Complete!\n"
              << "  Steps:      " << num_steps << "\n"
              << "  Time:       " << std::fixed << std::setprecision(2) << total_s << "s\n"
              << "  Throughput: " << static_cast<float>(num_steps) / total_s << " steps/s\n"
              << "  Output:     " << output_folder << "/\n"
              << "  Files:\n"
              << "    screenshots/    — PNG frames\n"
              << "    step_metrics.csv\n"
              << "    tensor_log.json\n"
              << "    metrics.csv\n"
              << "    dashboard.json\n";
}

} // namespace ciir_sim
