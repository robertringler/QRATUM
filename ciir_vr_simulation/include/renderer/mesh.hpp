/**
 * @file mesh.hpp
 * @brief Renderable mesh with vertex attributes for Vulkan pipeline.
 *
 * Vertex layout:
 *   Position:          vec3 (location 0)
 *   Normal:            vec3 (location 1)
 *   Color:             vec4 (location 2)
 *   Constraint value:  float (location 3) — for shader-driven morphing/heatmap
 */

#pragma once

#include <cstdint>
#include <vector>
#include <array>
#include <string>

namespace ciir_sim {

/**
 * @brief Single vertex with all attributes.
 */
struct Vertex {
    std::array<float, 3> position  = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> normal    = {0.0f, 1.0f, 0.0f};
    std::array<float, 4> color     = {1.0f, 1.0f, 1.0f, 1.0f}; ///< RGBA
    float constraint_value         = 0.0f; ///< Shader uniform for morphing/heatmap
};

/**
 * @brief Mesh topology type.
 */
enum class MeshTopology : uint8_t {
    TRIANGLES = 0,
    LINES     = 1,
    POINTS    = 2,
};

/**
 * @brief Renderable mesh with vertex/index data.
 */
struct Mesh {
    std::string name;
    MeshTopology topology = MeshTopology::TRIANGLES;
    std::vector<Vertex>   vertices;
    std::vector<uint32_t> indices;

    // Transform
    std::array<float, 3> position = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> scale    = {1.0f, 1.0f, 1.0f};
    std::array<float, 4> rotation = {0.0f, 0.0f, 0.0f, 1.0f}; ///< Quaternion

    // Visibility
    bool visible = true;
    float opacity = 1.0f;

    // GPU handles
    uint32_t vbo = 0; ///< Vertex buffer object handle
    uint32_t ibo = 0; ///< Index buffer object handle

    /// Number of vertices.
    [[nodiscard]] uint32_t vertex_count() const {
        return static_cast<uint32_t>(vertices.size());
    }

    /// Number of indices.
    [[nodiscard]] uint32_t index_count() const {
        return static_cast<uint32_t>(indices.size());
    }

    // ---- Factory methods ----

    /// Generate a UV sphere.
    static Mesh create_sphere(float radius, uint32_t slices, uint32_t stacks,
                              std::array<float, 4> color = {1, 1, 1, 1});

    /// Generate a box.
    static Mesh create_box(std::array<float, 3> half_extents,
                           std::array<float, 4> color = {1, 1, 1, 1});

    /// Generate an arrow (cylinder + cone).
    static Mesh create_arrow(float length, float radius,
                             std::array<float, 4> color = {1, 0, 0, 1});

    /// Generate a grid plane.
    static Mesh create_grid(float size, uint32_t divisions,
                            std::array<float, 4> color = {0.3f, 0.3f, 0.3f, 1.0f});

    /// Generate a surface from a height field.
    static Mesh create_surface(const std::vector<std::vector<float>>& heights,
                               float x_min, float x_max,
                               float z_min, float z_max,
                               const std::vector<std::vector<std::array<float, 4>>>& colors);
};

} // namespace ciir_sim
