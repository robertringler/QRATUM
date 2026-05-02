/**
 * @file mesh.cpp
 * @brief Mesh factory method implementations (sphere, box, arrow, grid, surface).
 */

#include "renderer/mesh.hpp"

#include <cmath>

namespace ciir_sim {

namespace {

constexpr float PI = 3.14159265358979323846f;

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

} // anonymous namespace

// ---- UV Sphere ----

Mesh Mesh::create_sphere(float radius, uint32_t slices, uint32_t stacks,
                          std::array<float, 4> color) {
    Mesh mesh;
    mesh.name = "sphere";
    mesh.topology = MeshTopology::TRIANGLES;

    for (uint32_t j = 0; j <= stacks; ++j) {
        float phi = PI * static_cast<float>(j) / static_cast<float>(stacks);
        float sp = std::sin(phi), cp = std::cos(phi);

        for (uint32_t i = 0; i <= slices; ++i) {
            float theta = 2.0f * PI * static_cast<float>(i) / static_cast<float>(slices);
            float st = std::sin(theta), ct = std::cos(theta);

            float nx = sp * ct, ny = cp, nz = sp * st;
            Vertex v;
            v.position = {radius * nx, radius * ny, radius * nz};
            v.normal   = {nx, ny, nz};
            v.color    = color;
            mesh.vertices.push_back(v);
        }
    }

    uint32_t cols = slices + 1;
    for (uint32_t j = 0; j < stacks; ++j) {
        for (uint32_t i = 0; i < slices; ++i) {
            uint32_t a = j * cols + i;
            uint32_t b = a + cols;
            mesh.indices.insert(mesh.indices.end(), {a, b, a + 1});
            mesh.indices.insert(mesh.indices.end(), {a + 1, b, b + 1});
        }
    }
    return mesh;
}

// ---- Box ----

Mesh Mesh::create_box(std::array<float, 3> h, std::array<float, 4> color) {
    Mesh mesh;
    mesh.name = "box";
    mesh.topology = MeshTopology::TRIANGLES;

    // Each face: 4 vertices, 6 indices (2 triangles)
    struct Face { Vec3 normal; Vec3 u; Vec3 v; };
    Face faces[6] = {
        {{ 0, 0, 1}, { 1, 0, 0}, {0, 1, 0}},  // +Z
        {{ 0, 0,-1}, {-1, 0, 0}, {0, 1, 0}},  // -Z
        {{ 1, 0, 0}, { 0, 0,-1}, {0, 1, 0}},  // +X
        {{-1, 0, 0}, { 0, 0, 1}, {0, 1, 0}},  // -X
        {{ 0, 1, 0}, { 1, 0, 0}, {0, 0,-1}},  // +Y
        {{ 0,-1, 0}, { 1, 0, 0}, {0, 0, 1}},  // -Y
    };

    for (auto& f : faces) {
        uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
        float signs[4][2] = {{-1,-1},{1,-1},{1,1},{-1,1}};
        for (auto& s : signs) {
            float px = f.normal.x * h[0] + f.u.x * h[0] * s[0] + f.v.x * h[0] * s[1];
            float py = f.normal.y * h[1] + f.u.y * h[1] * s[0] + f.v.y * h[1] * s[1];
            float pz = f.normal.z * h[2] + f.u.z * h[2] * s[0] + f.v.z * h[2] * s[1];

            Vertex vtx;
            vtx.position = {px, py, pz};
            vtx.normal   = {f.normal.x, f.normal.y, f.normal.z};
            vtx.color    = color;
            mesh.vertices.push_back(vtx);
        }
        mesh.indices.insert(mesh.indices.end(),
            {base, base + 1, base + 2, base, base + 2, base + 3});
    }
    return mesh;
}

// ---- Arrow (cylinder body + cone tip) ----

Mesh Mesh::create_arrow(float length, float radius,
                         std::array<float, 4> color) {
    Mesh mesh;
    mesh.name = "arrow";
    mesh.topology = MeshTopology::TRIANGLES;

    constexpr uint32_t segments = 16;
    float body_len = length * 0.75f;
    float cone_len = length * 0.25f;
    float cone_radius = radius * 2.0f;

    // Cylinder body along +Y from 0 to body_len
    for (uint32_t i = 0; i <= segments; ++i) {
        float theta = 2.0f * PI * static_cast<float>(i) / static_cast<float>(segments);
        float cx = std::cos(theta), cz = std::sin(theta);

        Vertex bottom, top;
        bottom.position = {radius * cx, 0.0f, radius * cz};
        bottom.normal   = {cx, 0.0f, cz};
        bottom.color    = color;

        top.position = {radius * cx, body_len, radius * cz};
        top.normal   = {cx, 0.0f, cz};
        top.color    = color;

        mesh.vertices.push_back(bottom);
        mesh.vertices.push_back(top);
    }
    for (uint32_t i = 0; i < segments; ++i) {
        uint32_t a = i * 2, b = a + 1, c = a + 2, d = a + 3;
        mesh.indices.insert(mesh.indices.end(), {a, c, b});
        mesh.indices.insert(mesh.indices.end(), {b, c, d});
    }

    // Cone tip from body_len to body_len + cone_len
    uint32_t tip_base = static_cast<uint32_t>(mesh.vertices.size());
    // Tip apex
    Vertex apex;
    apex.position = {0.0f, body_len + cone_len, 0.0f};
    apex.normal   = {0.0f, 1.0f, 0.0f};
    apex.color    = color;
    mesh.vertices.push_back(apex);

    for (uint32_t i = 0; i <= segments; ++i) {
        float theta = 2.0f * PI * static_cast<float>(i) / static_cast<float>(segments);
        float cx = std::cos(theta), cz = std::sin(theta);

        // Cone normal: outward + upward component
        Vec3 n = normalize({cx, cone_radius / cone_len, cz});
        Vertex v;
        v.position = {cone_radius * cx, body_len, cone_radius * cz};
        v.normal   = {n.x, n.y, n.z};
        v.color    = color;
        mesh.vertices.push_back(v);
    }
    for (uint32_t i = 0; i < segments; ++i) {
        uint32_t base_i = tip_base + 1 + i;
        mesh.indices.insert(mesh.indices.end(), {tip_base, base_i + 1, base_i});
    }

    return mesh;
}

// ---- Grid ----

Mesh Mesh::create_grid(float size, uint32_t divisions,
                        std::array<float, 4> color) {
    Mesh mesh;
    mesh.name = "grid";
    mesh.topology = MeshTopology::LINES;

    float half = size * 0.5f;
    float step = size / static_cast<float>(divisions);

    for (uint32_t i = 0; i <= divisions; ++i) {
        float t = -half + step * static_cast<float>(i);

        // Line parallel to Z
        Vertex a, b;
        a.position = {t, 0.0f, -half}; a.normal = {0, 1, 0}; a.color = color;
        b.position = {t, 0.0f,  half}; b.normal = {0, 1, 0}; b.color = color;
        uint32_t idx = static_cast<uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back(a);
        mesh.vertices.push_back(b);
        mesh.indices.push_back(idx);
        mesh.indices.push_back(idx + 1);

        // Line parallel to X
        Vertex c, d;
        c.position = {-half, 0.0f, t}; c.normal = {0, 1, 0}; c.color = color;
        d.position = { half, 0.0f, t}; d.normal = {0, 1, 0}; d.color = color;
        idx = static_cast<uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back(c);
        mesh.vertices.push_back(d);
        mesh.indices.push_back(idx);
        mesh.indices.push_back(idx + 1);
    }
    return mesh;
}

// ---- Surface from height field ----

Mesh Mesh::create_surface(const std::vector<std::vector<float>>& heights,
                           float x_min, float x_max,
                           float z_min, float z_max,
                           const std::vector<std::vector<std::array<float, 4>>>& colors) {
    Mesh mesh;
    mesh.name = "surface";
    mesh.topology = MeshTopology::TRIANGLES;

    if (heights.empty() || heights[0].empty()) return mesh;

    uint32_t rows = static_cast<uint32_t>(heights.size());
    uint32_t cols = static_cast<uint32_t>(heights[0].size());

    float dx = (cols > 1) ? (x_max - x_min) / static_cast<float>(cols - 1) : 0.0f;
    float dz = (rows > 1) ? (z_max - z_min) / static_cast<float>(rows - 1) : 0.0f;

    bool has_colors = !colors.empty() && colors.size() == rows &&
                      !colors[0].empty() && colors[0].size() == cols;

    // Generate vertices
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            float x = x_min + dx * static_cast<float>(c);
            float z = z_min + dz * static_cast<float>(r);
            float y = heights[r][c];

            // Compute normal via central differences
            float dydx = 0.0f, dydz = 0.0f;
            if (c > 0 && c < cols - 1) {
                dydx = (heights[r][c + 1] - heights[r][c - 1]) / (2.0f * dx);
            } else if (c > 0) {
                dydx = (heights[r][c] - heights[r][c - 1]) / dx;
            } else if (c < cols - 1) {
                dydx = (heights[r][c + 1] - heights[r][c]) / dx;
            }

            if (r > 0 && r < rows - 1) {
                dydz = (heights[r + 1][c] - heights[r - 1][c]) / (2.0f * dz);
            } else if (r > 0) {
                dydz = (heights[r][c] - heights[r - 1][c]) / dz;
            } else if (r < rows - 1) {
                dydz = (heights[r + 1][c] - heights[r][c]) / dz;
            }

            // Normal = normalize(cross(tangent_x, tangent_z))
            // tangent_x = (1, dydx, 0), tangent_z = (0, dydz, 1)
            Vec3 n = normalize(cross({1, dydx, 0}, {0, dydz, 1}));

            Vertex v;
            v.position = {x, y, z};
            v.normal   = {n.x, n.y, n.z};
            v.color    = has_colors ? colors[r][c] : std::array<float,4>{1,1,1,1};
            mesh.vertices.push_back(v);
        }
    }

    // Generate indices (two triangles per quad)
    for (uint32_t r = 0; r < rows - 1; ++r) {
        for (uint32_t c = 0; c < cols - 1; ++c) {
            uint32_t tl = r * cols + c;
            uint32_t tr = tl + 1;
            uint32_t bl = tl + cols;
            uint32_t br = bl + 1;
            mesh.indices.insert(mesh.indices.end(), {tl, bl, tr});
            mesh.indices.insert(mesh.indices.end(), {tr, bl, br});
        }
    }

    return mesh;
}

} // namespace ciir_sim
