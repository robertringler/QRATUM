/**
 * @file vr_context.cpp
 * @brief Stub VR context (OpenXR not available, provides safe defaults).
 */

#include "renderer/vr_context.hpp"

#include <cmath>

namespace ciir_sim {

VRContext::~VRContext() {
    shutdown();
}

bool VRContext::init() {
    // VR hardware not available in stub build
    active_ = false;
    return false;
}

void VRContext::shutdown() {
    active_ = false;
}

bool VRContext::begin_frame() {
    return active_;
}

void VRContext::end_frame() {
    // Stub: no-op
}

GazeRay VRContext::gaze_ray() const {
    // Forward direction from head orientation quaternion
    const auto& q = head_pose_.orientation;
    float qx = q[0], qy = q[1], qz = q[2], qw = q[3];

    // Rotate -Z by quaternion to get forward vector
    // v' = q * v * q^-1 with v = (0, 0, -1)
    float dx = 2.0f * (qx * qz + qy * qw);
    float dy = 2.0f * (qy * qz - qx * qw);
    float dz = -(1.0f - 2.0f * (qx * qx + qy * qy));

    float len = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (len > 1e-6f) {
        dx /= len; dy /= len; dz /= len;
    } else {
        dx = 0.0f; dy = 0.0f; dz = -1.0f;
    }

    return GazeRay{head_pose_.position, {dx, dy, dz}};
}

void VRContext::teleport(const std::array<float, 3>& target) {
    world_offset[0] = target[0] - head_pose_.position[0];
    world_offset[1] = target[1] - head_pose_.position[1];
    world_offset[2] = target[2] - head_pose_.position[2];
}

int VRContext::check_grab(const std::array<float, 3>& point, float grab_radius) const {
    auto dist = [](const std::array<float, 3>& a, const std::array<float, 3>& b) {
        float dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };

    if (left_.active && left_.grip_pressed &&
        dist(left_.pose.position, point) <= grab_radius) {
        return 0;
    }
    if (right_.active && right_.grip_pressed &&
        dist(right_.pose.position, point) <= grab_radius) {
        return 1;
    }
    return -1;
}

void VRContext::haptic_pulse(int /*controller*/, float /*intensity*/,
                              float /*duration_ms*/) {
    // Stub: no-op
}

VRContext::StereoView VRContext::stereo_view() const {
    // Return identity matrices
    constexpr std::array<float, 16> identity = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    return StereoView{identity, identity, identity, identity};
}

} // namespace ciir_sim
