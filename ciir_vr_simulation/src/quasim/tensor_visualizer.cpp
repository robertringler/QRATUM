/**
 * @file tensor_visualizer.cpp
 * @brief Implementation of TensorVisualizer — 3D mesh generation for loss surfaces,
 *        gradient fields, spectrum bars, and heatmaps.
 */

#include "quasim/tensor_visualizer.hpp"

#include <cmath>
#include <algorithm>
#include <cassert>

namespace ciir_sim {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map a scalar in [0, 1] to a heatmap color (blue → green → red).
static std::array<float, 4> heatmap_color(float t) {
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

/// Compute finite-difference surface normal from a height grid.
static std::array<float, 3> grid_normal(
    const std::vector<std::vector<float>>& grid,
    uint32_t i, uint32_t j, float dx, float dz)
{
    auto rows = static_cast<uint32_t>(grid.size());
    auto cols = static_cast<uint32_t>(grid[0].size());

    float dhdx = 0.0f, dhdz = 0.0f;
    if (i > 0 && i + 1 < rows)
        dhdx = (grid[i + 1][j] - grid[i - 1][j]) / (2.0f * dx);
    else if (i + 1 < rows)
        dhdx = (grid[i + 1][j] - grid[i][j]) / dx;
    else if (i > 0)
        dhdx = (grid[i][j] - grid[i - 1][j]) / dx;

    if (j > 0 && j + 1 < cols)
        dhdz = (grid[i][j + 1] - grid[i][j - 1]) / (2.0f * dz);
    else if (j + 1 < cols)
        dhdz = (grid[i][j + 1] - grid[i][j]) / dz;
    else if (j > 0)
        dhdz = (grid[i][j] - grid[i][j - 1]) / dz;

    // n = (-dh/dx, 1, -dh/dz), then normalize
    float nx = -dhdx, ny = 1.0f, nz = -dhdz;
    float len = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (len > 1e-8f) { nx /= len; ny /= len; nz /= len; }
    return {nx, ny, nz};
}

// ---------------------------------------------------------------------------
// generate_loss_surface
// ---------------------------------------------------------------------------

Mesh TensorVisualizer::generate_loss_surface(
    const TensorRuntime& runtime,
    const Matrix& rho_center,
    uint32_t n_grid,
    float extent) const
{
    const uint32_t d = rho_center.dim;

    // Build two random symmetric perturbation directions
    Matrix dir1(d), dir2(d);
    for (uint32_t i = 0; i < d; ++i)
        for (uint32_t j = i; j < d; ++j) {
            float v1 = static_cast<float>((i * 7 + j * 13 + 1) % 17) / 17.0f - 0.5f;
            float v2 = static_cast<float>((i * 11 + j * 3 + 5) % 19) / 19.0f - 0.5f;
            dir1(i, j) = v1; dir1(j, i) = v1;
            dir2(i, j) = v2; dir2(j, i) = v2;
        }

    // Normalize directions
    float n1 = dir1.frobenius_norm();
    float n2 = dir2.frobenius_norm();
    if (n1 > 1e-8f) dir1.scale(1.0f / n1);
    if (n2 > 1e-8f) dir2.scale(1.0f / n2);

    auto grid = runtime.loss_landscape_2d(rho_center, dir1, dir2, n_grid, extent);

    // Find value range for color normalization
    float vmin = grid[0][0], vmax = grid[0][0];
    for (const auto& row : grid)
        for (float v : row) { vmin = std::min(vmin, v); vmax = std::max(vmax, v); }
    float vrange = (vmax - vmin > 1e-12f) ? (vmax - vmin) : 1.0f;

    // Build colored height grid for create_surface
    std::vector<std::vector<std::array<float, 4>>> colors(
        n_grid, std::vector<std::array<float, 4>>(n_grid));
    for (uint32_t i = 0; i < n_grid; ++i)
        for (uint32_t j = 0; j < n_grid; ++j)
            colors[i][j] = heatmap_color((grid[i][j] - vmin) / vrange);

    Mesh mesh = Mesh::create_surface(grid,
                                     -extent, extent,
                                     -extent, extent,
                                     colors);
    mesh.name = "loss_surface";
    return mesh;
}

// ---------------------------------------------------------------------------
// generate_gradient_field
// ---------------------------------------------------------------------------

std::vector<Mesh> TensorVisualizer::generate_gradient_field(
    const TensorRuntime& runtime,
    const Matrix& rho_center,
    uint32_t n_arrows) const
{
    const uint32_t d = rho_center.dim;

    // Build two perturbation directions (same deterministic scheme)
    Matrix dir1(d), dir2(d);
    for (uint32_t i = 0; i < d; ++i)
        for (uint32_t j = i; j < d; ++j) {
            float v1 = static_cast<float>((i * 7 + j * 13 + 1) % 17) / 17.0f - 0.5f;
            float v2 = static_cast<float>((i * 11 + j * 3 + 5) % 19) / 19.0f - 0.5f;
            dir1(i, j) = v1; dir1(j, i) = v1;
            dir2(i, j) = v2; dir2(j, i) = v2;
        }
    float n1 = dir1.frobenius_norm();
    float n2 = dir2.frobenius_norm();
    if (n1 > 1e-8f) dir1.scale(1.0f / n1);
    if (n2 > 1e-8f) dir2.scale(1.0f / n2);

    const float extent = 0.3f;
    std::vector<Mesh> arrows;
    arrows.reserve(n_arrows * n_arrows);

    for (uint32_t i = 0; i < n_arrows; ++i) {
        float alpha = -extent + 2.0f * extent * static_cast<float>(i) /
                      static_cast<float>(n_arrows - 1);
        for (uint32_t j = 0; j < n_arrows; ++j) {
            float beta = -extent + 2.0f * extent * static_cast<float>(j) /
                         static_cast<float>(n_arrows - 1);

            Matrix rho = rho_center;
            rho.add_scaled(dir1, alpha);
            rho.add_scaled(dir2, beta);

            std::vector<Matrix> batch = {rho};
            auto grads = runtime.compute_gradient(batch);
            const Matrix& g = grads[0];

            // Project gradient onto the two directions: (g · dir1, g · dir2)
            float gx = g.trace_product(dir1);
            float gy = g.trace_product(dir2);
            float gmag = std::sqrt(gx * gx + gy * gy);

            float arrow_len = std::min(gmag * 0.5f, 0.08f);
            if (arrow_len < 1e-6f) arrow_len = 1e-6f;

            // Normalize direction
            float inv_mag = (gmag > 1e-8f) ? (1.0f / gmag) : 0.0f;
            float dx = -gx * inv_mag;  // negative gradient direction
            float dz = -gy * inv_mag;

            float t = std::clamp(gmag * 2.0f, 0.0f, 1.0f);
            auto color = heatmap_color(t);

            Mesh arrow;
            arrow.name = "grad_arrow";
            arrow.topology = MeshTopology::LINES;

            Vertex base_v;
            base_v.position = {alpha, 0.0f, beta};
            base_v.normal   = {0.0f, 1.0f, 0.0f};
            base_v.color    = color;

            Vertex tip_v;
            tip_v.position = {alpha + dx * arrow_len, 0.0f, beta + dz * arrow_len};
            tip_v.normal   = {0.0f, 1.0f, 0.0f};
            tip_v.color    = color;

            arrow.vertices = {base_v, tip_v};
            arrow.indices  = {0, 1};
            arrows.push_back(std::move(arrow));
        }
    }
    return arrows;
}

// ---------------------------------------------------------------------------
// generate_spectrum_bars
// ---------------------------------------------------------------------------

Mesh TensorVisualizer::generate_spectrum_bars(const QuASIMTensor& tensor) const {
    // Interpret the tensor as a square matrix and compute eigenvalues.
    // For non-square tensors, use the raw data as a flat vector.
    const auto& shape = tensor.shape();
    uint32_t d = 0;
    bool is_square = (shape.size() == 2 && shape[0] == shape[1]);

    std::vector<float> eigs;
    if (is_square) {
        d = shape[0];
        Matrix mat(d, std::vector<float>(tensor.data(), tensor.data() + d * d));
        eigs = mat.eigenvalues();
    } else {
        // Use sorted absolute values of elements as "spectrum"
        eigs.resize(tensor.numel());
        for (size_t i = 0; i < tensor.numel(); ++i)
            eigs[i] = tensor[i];
        std::sort(eigs.begin(), eigs.end(), [](float a, float b) {
            return std::abs(a) > std::abs(b);
        });
        if (eigs.size() > 32) eigs.resize(32); // cap bar count
    }

    if (eigs.empty()) return Mesh{.name = "spectrum_bars"};

    float max_abs = 0.0f;
    for (float e : eigs) max_abs = std::max(max_abs, std::abs(e));
    if (max_abs < 1e-12f) max_abs = 1.0f;

    Mesh mesh;
    mesh.name = "spectrum_bars";
    mesh.topology = MeshTopology::TRIANGLES;

    const float bar_width  = 0.08f;
    const float bar_depth  = 0.08f;
    const float spacing    = 0.12f;
    const float height_scale = 1.0f / max_abs;

    for (size_t idx = 0; idx < eigs.size(); ++idx) {
        float h = eigs[idx] * height_scale;
        float x = static_cast<float>(idx) * spacing;
        float t = (eigs[idx] / max_abs + 1.0f) * 0.5f; // map [-1,1] → [0,1]
        auto color = heatmap_color(t);

        float y0 = 0.0f;
        float y1 = h;
        if (y1 < y0) std::swap(y0, y1);
        if (y1 - y0 < 1e-6f) y1 = y0 + 1e-6f;

        // 8 corners of a box
        float hx = bar_width * 0.5f;
        float hz = bar_depth * 0.5f;
        std::array<std::array<float, 3>, 8> corners = {{
            {x - hx, y0, -hz}, {x + hx, y0, -hz},
            {x + hx, y0,  hz}, {x - hx, y0,  hz},
            {x - hx, y1, -hz}, {x + hx, y1, -hz},
            {x + hx, y1,  hz}, {x - hx, y1,  hz}
        }};

        uint32_t base = static_cast<uint32_t>(mesh.vertices.size());

        // 6 faces × 4 vertices with appropriate normals
        struct Face { uint32_t a, b, c, d; std::array<float,3> n; };
        std::array<Face, 6> faces = {{
            {0, 1, 2, 3, {0, -1, 0}}, // bottom
            {4, 7, 6, 5, {0,  1, 0}}, // top
            {0, 4, 5, 1, {0, 0, -1}}, // front
            {2, 6, 7, 3, {0, 0,  1}}, // back
            {0, 3, 7, 4, {-1, 0, 0}}, // left
            {1, 5, 6, 2, { 1, 0, 0}}, // right
        }};

        for (const auto& f : faces) {
            uint32_t vi = static_cast<uint32_t>(mesh.vertices.size());
            for (uint32_t ci : {f.a, f.b, f.c, f.d}) {
                Vertex v;
                v.position = corners[ci];
                v.normal   = f.n;
                v.color    = color;
                v.constraint_value = eigs[idx];
                mesh.vertices.push_back(v);
            }
            mesh.indices.insert(mesh.indices.end(),
                {vi, vi + 1, vi + 2, vi, vi + 2, vi + 3});
        }
    }

    return mesh;
}

// ---------------------------------------------------------------------------
// generate_heatmap
// ---------------------------------------------------------------------------

Mesh TensorVisualizer::generate_heatmap(
    const std::vector<std::vector<float>>& grid,
    std::array<float, 2> x_range,
    std::array<float, 2> y_range) const
{
    if (grid.empty() || grid[0].empty()) return Mesh{.name = "heatmap"};

    auto rows = static_cast<uint32_t>(grid.size());
    auto cols = static_cast<uint32_t>(grid[0].size());

    // Value range for color normalization
    float vmin = grid[0][0], vmax = grid[0][0];
    for (const auto& row : grid)
        for (float v : row) { vmin = std::min(vmin, v); vmax = std::max(vmax, v); }
    float vrange = (vmax - vmin > 1e-12f) ? (vmax - vmin) : 1.0f;

    float dx = (x_range[1] - x_range[0]) / static_cast<float>(rows - 1);
    float dz = (y_range[1] - y_range[0]) / static_cast<float>(cols - 1);

    Mesh mesh;
    mesh.name = "heatmap";
    mesh.topology = MeshTopology::TRIANGLES;

    // Generate vertices
    for (uint32_t i = 0; i < rows; ++i) {
        float x = x_range[0] + static_cast<float>(i) * dx;
        for (uint32_t j = 0; j < cols; ++j) {
            float z = y_range[0] + static_cast<float>(j) * dz;
            float val = grid[i][j];
            float t = (val - vmin) / vrange;

            Vertex v;
            v.position = {x, val, z};
            v.normal   = grid_normal(grid, i, j, dx, dz);
            v.color    = heatmap_color(t);
            v.constraint_value = val;
            mesh.vertices.push_back(v);
        }
    }

    // Generate triangle indices for the grid
    for (uint32_t i = 0; i + 1 < rows; ++i) {
        for (uint32_t j = 0; j + 1 < cols; ++j) {
            uint32_t a = i * cols + j;
            uint32_t b = a + 1;
            uint32_t c = a + cols;
            uint32_t d = c + 1;
            mesh.indices.insert(mesh.indices.end(), {a, c, b, b, c, d});
        }
    }

    return mesh;
}

// ---------------------------------------------------------------------------
// update_heatmap_values
// ---------------------------------------------------------------------------

void TensorVisualizer::update_heatmap_values(Mesh& mesh,
                                              const std::vector<float>& values) const {
    size_t n = std::min(mesh.vertices.size(), values.size());
    if (n == 0) return;

    float vmin = values[0], vmax = values[0];
    for (size_t i = 0; i < n; ++i) {
        vmin = std::min(vmin, values[i]);
        vmax = std::max(vmax, values[i]);
    }
    float vrange = (vmax - vmin > 1e-12f) ? (vmax - vmin) : 1.0f;

    for (size_t i = 0; i < n; ++i) {
        mesh.vertices[i].constraint_value = values[i];
        float t = (values[i] - vmin) / vrange;
        mesh.vertices[i].color = heatmap_color(t);
    }
}

} // namespace ciir_sim
