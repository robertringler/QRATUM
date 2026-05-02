/**
 * @file tensor_visualizer.hpp
 * @brief Shader-driven 3D tensor heatmaps and gradient field visualization.
 *
 * Generates renderable geometry for tensor transformations:
 * - 3D loss landscape surfaces
 * - Gradient vector fields
 * - Tensor spectrum bar charts
 * - Constraint/observer evaluation heatmaps
 */

#pragma once

#include "quasim/quasim_tensor.hpp"
#include "quasim/tensor_runtime.hpp"
#include "renderer/mesh.hpp"

#include <vector>

namespace ciir_sim {

/**
 * @brief 3D visualization of QuASIM tensor transformations.
 */
class TensorVisualizer {
public:
    TensorVisualizer() = default;

    /**
     * @brief Generate 3D loss landscape surface mesh.
     *
     * @param runtime Tensor runtime for loss evaluation.
     * @param rho_center Center density matrix.
     * @param n_grid Grid resolution.
     * @param extent Sampling extent.
     * @return Surface mesh colored by loss value.
     */
    Mesh generate_loss_surface(
        const TensorRuntime& runtime,
        const Matrix& rho_center,
        uint32_t n_grid = 25,
        float extent = 0.3f) const;

    /**
     * @brief Generate gradient vector field arrows.
     *
     * Creates arrow meshes at grid points showing gradient direction
     * and magnitude on the loss landscape.
     *
     * @param runtime Tensor runtime.
     * @param rho_center Center density matrix.
     * @param n_arrows Number of arrows per dimension.
     * @return Arrow meshes.
     */
    std::vector<Mesh> generate_gradient_field(
        const TensorRuntime& runtime,
        const Matrix& rho_center,
        uint32_t n_arrows = 8) const;

    /**
     * @brief Generate tensor spectrum visualization (bar chart in 3D).
     *
     * @param tensor The tensor to visualize.
     * @return Mesh with colored bars for each spectral value.
     */
    Mesh generate_spectrum_bars(const QuASIMTensor& tensor) const;

    /**
     * @brief Generate heatmap mesh from a 2D grid of values.
     *
     * @param grid n_x × n_y grid of values.
     * @param x_range Min/max for x axis.
     * @param y_range Min/max for y axis.
     * @return Colored surface mesh.
     */
    Mesh generate_heatmap(
        const std::vector<std::vector<float>>& grid,
        std::array<float, 2> x_range,
        std::array<float, 2> y_range) const;

    /**
     * @brief Update shader data for real-time tensor animation.
     *
     * Modifies vertex attributes based on current tensor values.
     *
     * @param mesh The mesh to update.
     * @param values Per-vertex scalar values from tensor computation.
     */
    void update_heatmap_values(Mesh& mesh, const std::vector<float>& values) const;
};

} // namespace ciir_sim
