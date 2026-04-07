/**
 * @file main.cpp
 * @brief Entry point for CIIR → QuASIM → QRATUM VR Simulation.
 *
 * Usage:
 *   ciir_vr_sim [options]
 *
 * Options:
 *   --steps N            Number of evolution steps (default: 200)
 *   --screenshot-interval N  Capture screenshot every N steps (default: 50)
 *   --output DIR         Output directory (default: ciir_output)
 *   --batch B            Batch size (default: 4)
 *   --dim D              Representation dimension (default: 8)
 *   --constraints C      Number of constraints (default: 3)
 *   --observers O        Number of observers (default: 2)
 *   --lr RATE            Learning rate (default: 0.01)
 *   --entropy GAMMA      Entropy weight (default: 0.01)
 *   --seed S             Random seed (default: 42)
 *   --vr                 Enable VR mode
 *   --width W            Window width (default: 1920)
 *   --height H           Window height (default: 1080)
 *   --help               Show this help
 */

#include "simulation_engine.hpp"

#include <cstring>
#include <iostream>
#include <string>

using namespace ciir_sim;

namespace {

void print_usage(const char* prog) {
    std::cout
        << "CIIR → QuASIM → QRATUM VR Simulation v0.1.0\n"
        << "=============================================\n\n"
        << "Usage: " << prog << " [options]\n\n"
        << "Simulation:\n"
        << "  --steps N              Evolution steps (default: 200)\n"
        << "  --screenshot-interval N  Screenshot every N steps (default: 50, 0=disable)\n"
        << "  --output DIR           Output directory (default: ciir_output)\n"
        << "  --batch B              Batch size (default: 4)\n"
        << "  --dim D                Representation dimension D (default: 8)\n"
        << "  --rank R               Ontic dimension R (default: 4)\n"
        << "  --constraints C        Number of constraints (default: 3)\n"
        << "  --observers O          Number of observers (default: 2)\n"
        << "  --lr RATE              Learning rate η (default: 0.01)\n"
        << "  --entropy GAMMA        Entropy weight γ (default: 0.01)\n"
        << "  --seed S               Random seed (default: 42)\n\n"
        << "Rendering:\n"
        << "  --vr                   Enable VR mode (OpenXR)\n"
        << "  --width W              Window width (default: 1920)\n"
        << "  --height H             Window height (default: 1080)\n"
        << "  --no-validation        Disable Vulkan validation layers\n\n"
        << "  --help                 Show this help\n";
}

bool match_arg(const char* arg, const char* name) {
    return std::strcmp(arg, name) == 0;
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    SimulationConfig config;
    int screenshot_interval = 50;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        if (match_arg(argv[i], "--help") || match_arg(argv[i], "-h")) {
            print_usage(argv[0]);
            return 0;
        }
        else if (match_arg(argv[i], "--steps") && i + 1 < argc)
            config.n_steps = static_cast<uint32_t>(std::stoi(argv[++i]));
        else if (match_arg(argv[i], "--screenshot-interval") && i + 1 < argc)
            screenshot_interval = std::stoi(argv[++i]);
        else if (match_arg(argv[i], "--output") && i + 1 < argc)
            config.output_dir = argv[++i];
        else if (match_arg(argv[i], "--batch") && i + 1 < argc)
            config.batch_size = static_cast<uint32_t>(std::stoi(argv[++i]));
        else if (match_arg(argv[i], "--dim") && i + 1 < argc)
            config.rep_dim = static_cast<uint32_t>(std::stoi(argv[++i]));
        else if (match_arg(argv[i], "--rank") && i + 1 < argc)
            config.rank = static_cast<uint32_t>(std::stoi(argv[++i]));
        else if (match_arg(argv[i], "--constraints") && i + 1 < argc)
            config.n_constraints = static_cast<uint32_t>(std::stoi(argv[++i]));
        else if (match_arg(argv[i], "--observers") && i + 1 < argc)
            config.n_observers = static_cast<uint32_t>(std::stoi(argv[++i]));
        else if (match_arg(argv[i], "--lr") && i + 1 < argc)
            config.learning_rate = std::stof(argv[++i]);
        else if (match_arg(argv[i], "--entropy") && i + 1 < argc)
            config.entropy_weight = std::stof(argv[++i]);
        else if (match_arg(argv[i], "--seed") && i + 1 < argc)
            config.seed = static_cast<uint32_t>(std::stoi(argv[++i]));
        else if (match_arg(argv[i], "--vr"))
            config.enable_vr = true;
        else if (match_arg(argv[i], "--width") && i + 1 < argc)
            config.window_width = static_cast<uint32_t>(std::stoi(argv[++i]));
        else if (match_arg(argv[i], "--height") && i + 1 < argc)
            config.window_height = static_cast<uint32_t>(std::stoi(argv[++i]));
        else if (match_arg(argv[i], "--no-validation"))
            config.enable_validation = false;
        else {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Banner
    std::cout << "\n"
              << "╔══════════════════════════════════════════════════╗\n"
              << "║  CIIR → QuASIM → QRATUM  VR Simulation  v0.1   ║\n"
              << "╠══════════════════════════════════════════════════╣\n"
              << "║  Manifold:  R=" << config.rank << " D=" << config.rep_dim
              << " B=" << config.batch_size << "                       ║\n"
              << "║  Theory:    C=" << config.n_constraints
              << " O=" << config.n_observers
              << " η=" << config.learning_rate
              << " γ=" << config.entropy_weight << "         ║\n"
              << "║  Steps:     " << config.n_steps
              << "  Screenshots: every " << screenshot_interval << "     ║\n"
              << "║  VR:        " << (config.enable_vr ? "Enabled " : "Disabled")
              << "                              ║\n"
              << "║  Output:    " << config.output_dir
              << "                          ║\n"
              << "╚══════════════════════════════════════════════════╝\n\n";

    // Run automated simulation with screenshot capture
    run_and_capture_simulation(
        static_cast<int>(config.n_steps),
        screenshot_interval,
        config.output_dir,
        config);

    return 0;
}
