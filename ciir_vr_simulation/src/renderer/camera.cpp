/**
 * @file camera.cpp
 * @brief 3D camera implementation (FPS, orbit, VR modes).
 */

#include "renderer/camera.hpp"

#include <algorithm>

namespace ciir_sim {

namespace {

constexpr float PI = 3.14159265358979323846f;

inline float deg2rad(float d) { return d * (PI / 180.0f); }

struct Vec3 { float x, y, z; };

inline Vec3 normalize(Vec3 v) {
    float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len > 1e-8f) { v.x /= len; v.y /= len; v.z /= len; }
    return v;
}

inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

inline float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 sub(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

} // anonymous namespace

Camera::Camera(CameraMode m) : mode(m) {}

// ---- Input processing ----

void Camera::process_mouse(float dx, float dy) {
    if (mode == CameraMode::VR) return;

    if (mode == CameraMode::ORBIT) {
        orbit_yaw   += dx * mouse_sensitivity * (PI / 180.0f);
        orbit_pitch += dy * mouse_sensitivity * (PI / 180.0f);
        // Clamp pitch to avoid gimbal lock
        orbit_pitch = std::clamp(orbit_pitch, -1.5f, 1.5f);
        update_orbit();
    } else {
        yaw   += dx * mouse_sensitivity;
        pitch -= dy * mouse_sensitivity;
        pitch = std::clamp(pitch, -89.0f, 89.0f);
    }
}

void Camera::process_movement(float fwd, float right_amount,
                                float up_amount, float dt) {
    if (mode != CameraMode::FPS) return;

    auto f = forward();
    auto r = right_dir();

    float speed = move_speed * dt;
    position[0] += (f[0] * fwd + r[0] * right_amount) * speed;
    position[1] += up_amount * speed;
    position[2] += (f[2] * fwd + r[2] * right_amount) * speed;
}

void Camera::process_scroll(float delta) {
    if (mode == CameraMode::ORBIT) {
        orbit_distance -= delta * 0.5f;
        orbit_distance = std::max(0.1f, orbit_distance);
        update_orbit();
    }
}

void Camera::update_orbit() {
    float cp = std::cos(orbit_pitch), sp = std::sin(orbit_pitch);
    float cy = std::cos(orbit_yaw),   sy = std::sin(orbit_yaw);

    position[0] = target[0] + orbit_distance * cp * sy;
    position[1] = target[1] + orbit_distance * sp;
    position[2] = target[2] + orbit_distance * cp * cy;
}

// ---- Direction vectors ----

std::array<float, 3> Camera::forward() const {
    if (mode == CameraMode::ORBIT) {
        Vec3 d = normalize(sub(
            {target[0], target[1], target[2]},
            {position[0], position[1], position[2]}));
        return {d.x, d.y, d.z};
    }
    float yr = deg2rad(yaw), pr = deg2rad(pitch);
    return {std::cos(pr) * std::cos(yr),
            std::sin(pr),
            std::cos(pr) * std::sin(yr)};
}

std::array<float, 3> Camera::right_dir() const {
    auto f = forward();
    Vec3 r = normalize(cross({f[0], f[1], f[2]}, {up[0], up[1], up[2]}));
    return {r.x, r.y, r.z};
}

// ---- Matrices (column-major) ----

std::array<float, 16> Camera::view_matrix() const {
    Vec3 eye = {position[0], position[1], position[2]};
    auto fwd = forward();
    Vec3 f = {fwd[0], fwd[1], fwd[2]};
    Vec3 u = {up[0], up[1], up[2]};

    Vec3 r = normalize(cross(f, u));
    Vec3 real_up = cross(r, f);

    // Column-major lookAt matrix
    std::array<float, 16> m{};
    m[0]  =  r.x;      m[1]  =  real_up.x; m[2]  = -f.x; m[3]  = 0.0f;
    m[4]  =  r.y;      m[5]  =  real_up.y; m[6]  = -f.y; m[7]  = 0.0f;
    m[8]  =  r.z;      m[9]  =  real_up.z; m[10] = -f.z; m[11] = 0.0f;
    m[12] = -dot(r, eye); m[13] = -dot(real_up, eye); m[14] = dot(f, eye); m[15] = 1.0f;
    return m;
}

std::array<float, 16> Camera::projection_matrix() const {
    float fov_rad = deg2rad(fov_y);
    float tan_half = std::tan(fov_rad * 0.5f);

    std::array<float, 16> m{};
    m[0]  = 1.0f / (aspect_ratio * tan_half);
    m[5]  = 1.0f / tan_half;
    m[10] = -(z_far + z_near) / (z_far - z_near);
    m[11] = -1.0f;
    m[14] = -(2.0f * z_far * z_near) / (z_far - z_near);
    // m[15] = 0.0f (already zero)
    return m;
}

std::array<float, 16> Camera::view_projection_matrix() const {
    auto v = view_matrix();
    auto p = projection_matrix();

    // Column-major 4×4 multiply: result = P * V
    std::array<float, 16> r{};
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += p[k * 4 + row] * v[col * 4 + k];
            }
            r[col * 4 + row] = sum;
        }
    }
    return r;
}

} // namespace ciir_sim
