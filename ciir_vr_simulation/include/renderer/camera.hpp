/**
 * @file camera.hpp
 * @brief 3D camera with FPS, orbit, and VR modes.
 *
 * Supports:
 *   - FPS-style free camera (WASD + mouse)
 *   - Orbit camera (rotate around target)
 *   - VR passthrough (uses head tracking from VRContext)
 */

#pragma once

#include <array>
#include <cmath>
#include <cstdint>

namespace ciir_sim {

/**
 * @brief Camera projection mode.
 */
enum class CameraMode : uint8_t {
    FPS   = 0, ///< Free-flying first person
    ORBIT = 1, ///< Orbit around a target point
    VR    = 2, ///< Head-tracked VR
};

/**
 * @brief 3D camera for rendering.
 */
class Camera {
public:
    Camera() = default;
    explicit Camera(CameraMode mode);

    // ---- Configuration ----
    CameraMode mode = CameraMode::ORBIT;
    float fov_y  = 60.0f; ///< Vertical field of view in degrees
    float z_near = 0.01f;
    float z_far  = 100.0f;
    float aspect_ratio = 16.0f / 9.0f;

    // ---- Position ----
    std::array<float, 3> position = {0.0f, 2.0f, 5.0f};
    std::array<float, 3> target   = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> up       = {0.0f, 1.0f, 0.0f};

    // ---- Orbit parameters ----
    float orbit_distance = 5.0f;
    float orbit_yaw      = 0.0f;   ///< Radians
    float orbit_pitch    = 0.3f;   ///< Radians

    // ---- FPS parameters ----
    float yaw   = -90.0f; ///< Degrees
    float pitch = 0.0f;   ///< Degrees
    float move_speed   = 3.0f;
    float mouse_sensitivity = 0.1f;

    // ---- Update ----

    /// Process mouse movement (dx, dy in pixels).
    void process_mouse(float dx, float dy);

    /// Process keyboard movement: forward/backward/left/right/up/down.
    void process_movement(float forward, float right, float up_amount, float dt);

    /// Process scroll (zoom for orbit mode).
    void process_scroll(float delta);

    /// Update orbit camera position from yaw/pitch/distance.
    void update_orbit();

    // ---- Matrices ----

    /// Get 4×4 view matrix (column-major).
    [[nodiscard]] std::array<float, 16> view_matrix() const;

    /// Get 4×4 projection matrix (column-major).
    [[nodiscard]] std::array<float, 16> projection_matrix() const;

    /// Get combined view-projection matrix.
    [[nodiscard]] std::array<float, 16> view_projection_matrix() const;

    /// Get camera forward direction.
    [[nodiscard]] std::array<float, 3> forward() const;

    /// Get camera right direction.
    [[nodiscard]] std::array<float, 3> right_dir() const;
};

} // namespace ciir_sim
