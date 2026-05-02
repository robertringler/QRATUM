/**
 * @file ciir_visualizer.cpp
 * @brief Implementation of CIIRVisualizer — mesh generation for manifold, constraints,
 *        trajectories, and observer arrows.
 */

#include "ciir/ciir_visualizer.hpp"

#include <cmath>
#include <algorithm>

namespace ciir_sim {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static constexpr float PI = 3.14159265358979323846f;

/// Map a scalar in [0,1] to a heatmap color (blue → green → red).
static std::array<float, 4> violation_color(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    float r, g, b;
    if (t < 0.5f) {
        float s = t * 2.0f;
        r = 0.0f;
        g = s;
        b = 1.0f - s;
    } else {
        float s = (t - 0.5f) * 2.0f;
        r = s;
        g = 1.0f - s;
        b = 0.0f;
    }
    return {r, g, b, 1.0f};
}

// ---------------------------------------------------------------------------
// generate_manifold_mesh
// ---------------------------------------------------------------------------

Mesh CIIRVisualizer::generate_manifold_mesh(const CIIRState& state,
                                             uint32_t resolution) const {
    // Base sphere displaced by eigenvalue embedding.
    // For each batch element, the center is the 3D eigenvalue embedding.
    // We create a sphere around element 0 with per-vertex displacement
    // scaled by constraint violation.

    auto center = state.manifold_embedding_3d(0);
    float base_radius = 0.3f;
    float violation = state.total_violation(0);

    Mesh mesh;
    mesh.name = "ciir_manifold";
    mesh.topology = MeshTopology::TRIANGLES;

    const uint32_t slices = resolution;
    const uint32_t stacks = resolution / 2;

    // Generate vertices on a sphere displaced by eigenvalue embedding
    for (uint32_t i = 0; i <= stacks; ++i) {
        float phi = PI * static_cast<float>(i) / static_cast<float>(stacks);
        float sin_phi = std::sin(phi);
        float cos_phi = std::cos(phi);

        for (uint32_t j = 0; j <= slices; ++j) {
            float theta = 2.0f * PI * static_cast<float>(j) / static_cast<float>(slices);
            float sin_theta = std::sin(theta);
            float cos_theta = std::cos(theta);

            // Normal direction on unit sphere
            float nx = sin_phi * cos_theta;
            float ny = cos_phi;
            float nz = sin_phi * sin_theta;

            // Displace radius by a function of constraint violation
            float r = base_radius * (1.0f + 0.2f * violation * std::abs(nx * ny));

            Vertex v;
            v.position = {center[0] + r * nx,
                          center[1] + r * ny,
                          center[2] + r * nz};
            v.normal = {nx, ny, nz};
            v.constraint_value = violation;
            v.color = violation_color(std::clamp(violation, 0.0f, 1.0f));
            mesh.vertices.push_back(v);
        }
    }

    // Generate triangle indices
    for (uint32_t i = 0; i < stacks; ++i) {
        for (uint32_t j = 0; j < slices; ++j) {
            uint32_t a = i * (slices + 1) + j;
            uint32_t b = a + slices + 1;
            mesh.indices.push_back(a);
            mesh.indices.push_back(b);
            mesh.indices.push_back(a + 1);

            mesh.indices.push_back(a + 1);
            mesh.indices.push_back(b);
            mesh.indices.push_back(b + 1);
        }
    }

    return mesh;
}

// ---------------------------------------------------------------------------
// generate_constraint_surface
// ---------------------------------------------------------------------------

Mesh CIIRVisualizer::generate_constraint_surface(const CIIRState& state,
                                                  uint32_t n_samples) const {
    auto samples = state.constraint_surface_samples(state.constraints(), n_samples);

    // Find max violation for normalization
    float max_viol = 0.0f;
    for (const auto& s : samples)
        max_viol = std::max(max_viol, s.violation);
    if (max_viol < 1e-12f) max_viol = 1.0f;

    Mesh mesh;
    mesh.name = "constraint_surface";
    mesh.topology = MeshTopology::POINTS;

    for (const auto& s : samples) {
        Vertex v;
        v.position = s.position;
        v.normal = {0.0f, 1.0f, 0.0f};
        float t = s.violation / max_viol;
        v.color = violation_color(t);
        v.constraint_value = s.violation;
        mesh.vertices.push_back(v);
    }

    // Point topology: indices are just 0..n-1
    mesh.indices.resize(mesh.vertices.size());
    for (uint32_t i = 0; i < static_cast<uint32_t>(mesh.vertices.size()); ++i)
        mesh.indices[i] = i;

    return mesh;
}

// ---------------------------------------------------------------------------
// generate_trajectory_lines
// ---------------------------------------------------------------------------

Mesh CIIRVisualizer::generate_trajectory_lines(
    const std::vector<std::array<float, 3>>& trajectory) const
{
    Mesh mesh;
    mesh.name = "trajectory";
    mesh.topology = MeshTopology::LINES;

    if (trajectory.size() < 2) return mesh;

    const auto n = static_cast<uint32_t>(trajectory.size());

    for (uint32_t i = 0; i < n; ++i) {
        Vertex v;
        v.position = trajectory[i];
        v.normal = {0.0f, 1.0f, 0.0f};

        // Color gradient along trajectory (start=cyan, end=magenta)
        float t = static_cast<float>(i) / static_cast<float>(n - 1);
        v.color = {t, 0.2f, 1.0f - t, 1.0f};
        v.constraint_value = 0.0f;
        mesh.vertices.push_back(v);
    }

    // Line segment pairs: (0,1), (1,2), ..., (n-2, n-1)
    for (uint32_t i = 0; i + 1 < n; ++i) {
        mesh.indices.push_back(i);
        mesh.indices.push_back(i + 1);
    }

    return mesh;
}

// ---------------------------------------------------------------------------
// generate_observer_arrows
// ---------------------------------------------------------------------------

std::vector<Mesh> CIIRVisualizer::generate_observer_arrows(
    const CIIRObserver& observers,
    const std::array<float, 3>& center) const
{
    std::vector<Mesh> arrows;
    const auto& obs_list = observers.observers();

    // Predefined color palette for observers
    static const std::array<std::array<float, 4>, 6> palette = {{
        {1.0f, 0.2f, 0.2f, 1.0f},
        {0.2f, 1.0f, 0.2f, 1.0f},
        {0.2f, 0.2f, 1.0f, 1.0f},
        {1.0f, 1.0f, 0.2f, 1.0f},
        {1.0f, 0.2f, 1.0f, 1.0f},
        {0.2f, 1.0f, 1.0f, 1.0f}
    }};

    for (size_t idx = 0; idx < obs_list.size(); ++idx) {
        const auto& obs = obs_list[idx];
        auto color = palette[idx % palette.size()];

        // Direction from first 3 eigenvalues of the observer matrix
        auto eigs = obs.matrix.eigenvalues();
        std::array<float, 3> dir = {0.0f, 0.0f, 0.0f};
        for (uint32_t i = 0; i < std::min(static_cast<uint32_t>(eigs.size()), 3u); ++i)
            dir[i] = eigs[i];

        // Normalize direction
        float len = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
        if (len < 1e-8f) len = 1.0f;
        for (auto& d : dir) d /= len;

        float arrow_length = 0.4f;

        // Build a simple arrow as a line + cone tip approximation
        Mesh arrow;
        arrow.name = "observer_" + obs.name;
        arrow.topology = MeshTopology::LINES;

        // Shaft: from center to tip
        Vertex base_v;
        base_v.position = center;
        base_v.normal = {dir[0], dir[1], dir[2]};
        base_v.color = color;
        base_v.constraint_value = 0.0f;

        Vertex tip_v;
        tip_v.position = {center[0] + dir[0] * arrow_length,
                          center[1] + dir[1] * arrow_length,
                          center[2] + dir[2] * arrow_length};
        tip_v.normal = {dir[0], dir[1], dir[2]};
        tip_v.color = color;
        tip_v.constraint_value = 0.0f;

        arrow.vertices.push_back(base_v);
        arrow.vertices.push_back(tip_v);
        arrow.indices = {0, 1};

        // Arrowhead: small fan of lines from tip
        float head_size = 0.06f;
        // Find two perpendicular vectors to dir
        std::array<float, 3> perp1, perp2;
        if (std::abs(dir[0]) < 0.9f) {
            perp1 = {0.0f, -dir[2], dir[1]};
        } else {
            perp1 = {-dir[2], 0.0f, dir[0]};
        }
        float plen = std::sqrt(perp1[0] * perp1[0] + perp1[1] * perp1[1] + perp1[2] * perp1[2]);
        if (plen > 1e-8f) for (auto& p : perp1) p /= plen;

        perp2 = {dir[1] * perp1[2] - dir[2] * perp1[1],
                  dir[2] * perp1[0] - dir[0] * perp1[2],
                  dir[0] * perp1[1] - dir[1] * perp1[0]};

        constexpr uint32_t head_segments = 4;
        uint32_t tip_idx = 1;
        for (uint32_t h = 0; h < head_segments; ++h) {
            float angle = 2.0f * PI * static_cast<float>(h) / static_cast<float>(head_segments);
            float cs = std::cos(angle), sn = std::sin(angle);

            Vertex hv;
            hv.position = {tip_v.position[0] - dir[0] * head_size + (perp1[0] * cs + perp2[0] * sn) * head_size,
                           tip_v.position[1] - dir[1] * head_size + (perp1[1] * cs + perp2[1] * sn) * head_size,
                           tip_v.position[2] - dir[2] * head_size + (perp1[2] * cs + perp2[2] * sn) * head_size};
            hv.normal = {dir[0], dir[1], dir[2]};
            hv.color = color;
            hv.constraint_value = 0.0f;

            auto hv_idx = static_cast<uint32_t>(arrow.vertices.size());
            arrow.vertices.push_back(hv);
            arrow.indices.push_back(tip_idx);
            arrow.indices.push_back(hv_idx);
        }

        arrows.push_back(std::move(arrow));
    }

    return arrows;
}

// ---------------------------------------------------------------------------
// update_constraint_intensity
// ---------------------------------------------------------------------------

void CIIRVisualizer::update_constraint_intensity(Mesh& mesh,
                                                  const CIIRState& state) const {
    float violation = state.total_violation(0);
    float clamped = std::clamp(violation, 0.0f, 1.0f);

    for (auto& v : mesh.vertices) {
        v.constraint_value = violation;
        v.color = violation_color(clamped);
    }
}

// ---------------------------------------------------------------------------
// generate_commutator_lines
// ---------------------------------------------------------------------------

Mesh CIIRVisualizer::generate_commutator_lines(
    const CIIRObserver& observer_mgr,
    const std::vector<std::array<float, 3>>& positions,
    float threshold) const
{
    Mesh mesh;
    mesh.name = "commutator_lines";
    mesh.topology = MeshTopology::LINES;

    auto comm_mat = observer_mgr.commutator_matrix();
    const auto n = comm_mat.size();

    if (positions.size() < n) return mesh;

    // Build vertex list from positions
    for (size_t i = 0; i < n; ++i) {
        Vertex v;
        v.position = positions[i];
        v.normal = {0.0f, 1.0f, 0.0f};
        v.color = {1.0f, 1.0f, 1.0f, 0.5f};
        v.constraint_value = 0.0f;
        mesh.vertices.push_back(v);
    }

    // Find max commutator norm for color scaling
    float max_norm = 0.0f;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            max_norm = std::max(max_norm, comm_mat[i][j]);
    if (max_norm < 1e-12f) max_norm = 1.0f;

    // Connect non-commuting pairs
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (comm_mat[i][j] > threshold) {
                float intensity = comm_mat[i][j] / max_norm;
                auto color = violation_color(intensity);

                // Override vertex colors for these endpoints
                mesh.vertices[i].color = color;
                mesh.vertices[j].color = color;

                mesh.indices.push_back(static_cast<uint32_t>(i));
                mesh.indices.push_back(static_cast<uint32_t>(j));
            }
        }
    }

    return mesh;
}

} // namespace ciir_sim
