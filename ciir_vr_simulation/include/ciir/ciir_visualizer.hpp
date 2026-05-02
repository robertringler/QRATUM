/**
 * @file ciir_visualizer.hpp
 * @brief 3D visualization of CIIR manifolds, constraints, and observer effects.
 *
 * Generates renderable geometry for the Vulkan pipeline:
 * - Manifold mesh: morphs in real-time based on constraints
 * - Constraint surface: heatmap coloring by violation intensity
 * - Observer trajectories: measurement collapse paths
 * - Vertex shader data: per-vertex constraint intensity for dynamic color mapping
 */

#pragma once

#include "ciir/ciir_state.hpp"
#include "ciir/ciir_observer.hpp"
#include "renderer/mesh.hpp"

#include <vector>

namespace ciir_sim {

/**
 * @brief Generates and updates 3D geometry for CIIR manifold visualization.
 *
 * The visualizer creates mesh data that the Vulkan renderer draws.
 * Manifold morphing is driven by the vertex shader using per-vertex
 * constraint intensity values passed as vertex attributes.
 */
class CIIRVisualizer {
public:
    CIIRVisualizer() = default;

    /**
     * @brief Generate manifold mesh from current CIIR state.
     *
     * Creates a 3D surface mesh embedded using the spectral embedding
     * φ(ρ) = (λ₁, λ₂, λ₃) where λ_i are the largest eigenvalues.
     *
     * @param state Current CIIR state.
     * @param resolution Grid resolution for surface tessellation.
     * @return Renderable mesh with constraint-intensity vertex colors.
     */
    Mesh generate_manifold_mesh(const CIIRState& state, uint32_t resolution = 32) const;

    /**
     * @brief Generate constraint surface mesh (isosurface of C_i(ρ) = 0).
     *
     * Samples the constraint landscape around the current state and
     * creates a mesh colored by total violation.
     *
     * @param state Current CIIR state.
     * @param n_samples Number of surface samples.
     * @return Mesh with heatmap coloring.
     */
    Mesh generate_constraint_surface(const CIIRState& state, uint32_t n_samples = 500) const;

    /**
     * @brief Generate trajectory lines from state history.
     *
     * @param trajectory History of CIIR states (manifold embeddings).
     * @return Line mesh for the trajectory.
     */
    Mesh generate_trajectory_lines(
        const std::vector<std::array<float, 3>>& trajectory) const;

    /**
     * @brief Generate observer basis vectors as arrow meshes.
     *
     * Each observer O_j is visualized as an arrow in eigenvalue space.
     *
     * @param observers Observer manager.
     * @param center Center point for arrows.
     * @return Arrow meshes.
     */
    std::vector<Mesh> generate_observer_arrows(
        const CIIRObserver& observers,
        const std::array<float, 3>& center) const;

    /**
     * @brief Update per-vertex constraint intensity for shader-driven morphing.
     *
     * Called each frame to update the vertex attribute that the shader
     * uses for dynamic deformation and color mapping.
     *
     * @param mesh The manifold mesh to update.
     * @param state Current CIIR state.
     */
    void update_constraint_intensity(Mesh& mesh, const CIIRState& state) const;

    /**
     * @brief Generate commutator visualization (connecting non-commuting observers).
     *
     * @param observer_mgr Observer manager.
     * @param positions Observer arrow tip positions.
     * @param threshold Minimum commutator norm to draw connection.
     * @return Line mesh showing non-commutativity structure.
     */
    Mesh generate_commutator_lines(
        const CIIRObserver& observer_mgr,
        const std::vector<std::array<float, 3>>& positions,
        float threshold = 0.01f) const;
};

} // namespace ciir_sim
