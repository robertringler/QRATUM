/**
 * @file vr_context.hpp
 * @brief OpenXR VR/AR integration with controller input and haptics.
 *
 * Provides:
 *   - OpenXR session management
 *   - Controller tracking (position, orientation, buttons)
 *   - Teleportation and object manipulation
 *   - Gaze-based selection
 *   - Haptic feedback based on tensor gradient magnitude
 *   - HUD overlay rendering
 */

#pragma once

#include <cstdint>
#include <array>
#include <string>
#include <vector>

namespace ciir_sim {

/**
 * @brief 3D pose (position + orientation).
 */
struct Pose {
    std::array<float, 3> position    = {0.0f, 0.0f, 0.0f};
    std::array<float, 4> orientation = {0.0f, 0.0f, 0.0f, 1.0f}; ///< Quaternion (x,y,z,w)
};

/**
 * @brief VR controller state.
 */
struct ControllerState {
    Pose pose;
    bool trigger_pressed  = false;
    bool grip_pressed     = false;
    bool menu_pressed     = false;
    float trigger_value   = 0.0f;  ///< Analog trigger [0, 1]
    std::array<float, 2> thumbstick = {0.0f, 0.0f};
    bool active = false;
};

/**
 * @brief VR gaze ray for selection.
 */
struct GazeRay {
    std::array<float, 3> origin;
    std::array<float, 3> direction;
};

/**
 * @brief OpenXR VR context.
 *
 * Lifecycle:
 *   1. init() — Create OpenXR instance and session
 *   2. begin_frame() / end_frame() in render loop
 *   3. shutdown() — Destroy session
 *
 * When VR is not available, falls back to desktop mouse/keyboard.
 */
class VRContext {
public:
    VRContext() = default;
    ~VRContext();

    VRContext(const VRContext&) = delete;
    VRContext& operator=(const VRContext&) = delete;

    /**
     * @brief Initialize OpenXR session.
     * @return true on success, false if VR not available.
     */
    bool init();

    /// Shut down VR session.
    void shutdown();

    /// Whether VR is active.
    [[nodiscard]] bool is_active() const { return active_; }

    // ---- Frame lifecycle ----

    /// Begin VR frame (wait for poses).
    bool begin_frame();

    /// End VR frame (submit layers).
    void end_frame();

    // ---- Tracking ----

    /// Head pose in world space.
    [[nodiscard]] const Pose& head_pose() const { return head_pose_; }

    /// Left controller state.
    [[nodiscard]] const ControllerState& left_controller() const { return left_; }

    /// Right controller state.
    [[nodiscard]] const ControllerState& right_controller() const { return right_; }

    /// Gaze ray from head pose.
    [[nodiscard]] GazeRay gaze_ray() const;

    // ---- Interaction ----

    /**
     * @brief Teleport to a world position (triggered by controller).
     * @param target World position to teleport to.
     */
    void teleport(const std::array<float, 3>& target);

    /**
     * @brief Check if a world-space point is being grabbed by a controller.
     * @param point World position to test.
     * @param grab_radius Maximum grab distance.
     * @return -1 if not grabbed, 0 for left, 1 for right controller.
     */
    int check_grab(const std::array<float, 3>& point, float grab_radius = 0.1f) const;

    /**
     * @brief Trigger haptic feedback on a controller.
     * @param controller 0 = left, 1 = right.
     * @param intensity Vibration intensity [0, 1].
     * @param duration_ms Duration in milliseconds.
     */
    void haptic_pulse(int controller, float intensity, float duration_ms = 50.0f);

    // ---- View matrices ----

    /// Get stereo view matrices for left and right eyes.
    struct StereoView {
        std::array<float, 16> left_view;
        std::array<float, 16> right_view;
        std::array<float, 16> left_projection;
        std::array<float, 16> right_projection;
    };
    [[nodiscard]] StereoView stereo_view() const;

    /// World offset (for teleportation).
    std::array<float, 3> world_offset = {0.0f, 0.0f, 0.0f};

private:
    bool active_ = false;
    Pose head_pose_;
    ControllerState left_;
    ControllerState right_;
};

} // namespace ciir_sim
