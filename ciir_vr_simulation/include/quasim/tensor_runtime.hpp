/**
 * @file tensor_runtime.hpp
 * @brief QuASIM tensor runtime: gradient evolution, einsum contraction, projected updates.
 *
 * Implements the tensor computation pipeline (§4–§5 of formalization):
 *
 *   Loss: L(ρ) = Σ λ_i C_i(ρ)² + Σ μ_j |⟨O_j⟩ − y_j|² + γ·Ω(ρ)
 *
 *   Gradient: ∇L = Σ 2λ_i C_i K_i + Σ 2μ_j (⟨O_j⟩ − y_j) O_j − γ(log ρ + I)
 *
 *   Update: ρ_{t+1} = Π_C(ρ_t − η ∇L(ρ_t))
 *
 * All operations are expressed as tensor contractions (einsum-style)
 * suitable for GPU compute shader execution.
 */

#pragma once

#include "ciir/ciir_state.hpp"
#include "ciir/ciir_observer.hpp"
#include "quasim/quasim_tensor.hpp"

#include <vector>
#include <string>
#include <functional>

namespace ciir_sim {

/**
 * @brief Integration method for state evolution.
 */
enum class IntegrationMethod : uint8_t {
    GRADIENT           = 0, ///< Plain gradient descent
    PROJECTED_GRADIENT = 1, ///< Projected gradient (Ax4.2 constraint satisfaction)
    STRANG_SPLITTING   = 2, ///< Strang splitting: unitary + dissipative (T7.3)
};

/**
 * @brief Loss decomposition for logging and visualization.
 */
struct LossComponents {
    float total      = 0.0f;
    float constraint = 0.0f;
    float observer   = 0.0f;
    float entropy    = 0.0f;
};

/**
 * @brief Contraction profiling entry.
 */
struct ContractionLog {
    uint32_t step;
    float    loss;
    float    grad_norm;
    LossComponents components;
    float    elapsed_ms;
};

/**
 * @brief QuASIM tensor runtime for CIIR evolution.
 *
 * Manages the full tensor computation pipeline:
 * 1. Batch constraint evaluation: C_i(ρ) = Tr(K_i ρ), einsum 'de,bde->b'
 * 2. Batch observer evaluation: ⟨O_j⟩ = Tr(O_j ρ), einsum 'de,bde->b'
 * 3. Loss computation: L = Σ λ_i C_i² + Σ μ_j residual² − γ S(ρ)
 * 4. Gradient computation: ∇L (tensor contraction)
 * 5. Projected gradient step: Π_C(ρ − η ∇L)
 */
class TensorRuntime {
public:
    /**
     * @brief Construct tensor runtime for given CIIR configuration.
     * @param constraints Constraint operators with kernels K_i.
     * @param observers Observer operators.
     * @param entropy_weight Regularisation coefficient γ.
     */
    TensorRuntime(const std::vector<ConstraintOperator>& constraints,
                  const std::vector<ObserverOperator>& observers,
                  float entropy_weight = 0.01f);

    // ---- Loss computation ----

    /**
     * @brief Compute full CIIR loss for a batch of density matrices.
     * @param rho_batch Batch of density matrices.
     * @return Loss components.
     */
    LossComponents compute_loss(const std::vector<Matrix>& rho_batch) const;

    /**
     * @brief Compute loss gradient ∂L/∂ρ for a batch.
     * @param rho_batch Batch of density matrices.
     * @return Per-batch gradients.
     */
    std::vector<Matrix> compute_gradient(const std::vector<Matrix>& rho_batch) const;

    // ---- Evolution ----

    /**
     * @brief Perform one evolution step.
     * @param rho_batch Current batch of density matrices (modified in-place).
     * @param lr Learning rate η.
     * @param method Integration method.
     * @return Loss components for this step.
     */
    LossComponents step(std::vector<Matrix>& rho_batch, float lr,
                        IntegrationMethod method = IntegrationMethod::PROJECTED_GRADIENT);

    // ---- Tensor evaluation ----

    /// Batch constraint eval: returns (B, n_c) values.
    [[nodiscard]] std::vector<std::vector<float>> eval_constraints(
        const std::vector<Matrix>& rho_batch) const;

    /// Batch observer eval: returns (B, n_o) values.
    [[nodiscard]] std::vector<std::vector<float>> eval_observers(
        const std::vector<Matrix>& rho_batch) const;

    // ---- Loss landscape sampling ----

    /**
     * @brief Sample 2D loss landscape slice.
     * @param rho_center Center density matrix.
     * @param dir1 First perturbation direction (symmetric).
     * @param dir2 Second perturbation direction (symmetric).
     * @param n_grid Grid resolution per dimension.
     * @param extent Sampling range.
     * @return Grid of loss values (n_grid × n_grid).
     */
    std::vector<std::vector<float>> loss_landscape_2d(
        const Matrix& rho_center,
        const Matrix& dir1, const Matrix& dir2,
        uint32_t n_grid = 25, float extent = 0.5f) const;

    // ---- Profiling ----
    [[nodiscard]] const std::vector<ContractionLog>& contraction_log() const { return log_; }
    void clear_log() { log_.clear(); }

    // ---- Parameter control ----
    void set_constraint_weight(uint32_t idx, float weight);
    void set_observer_target(uint32_t idx, float target);
    void set_entropy_weight(float gamma) { entropy_weight_ = gamma; }
    [[nodiscard]] float entropy_weight() const { return entropy_weight_; }

private:
    /// Constraint loss: Σ λ_i C_i(ρ)²
    float constraint_loss(const Matrix& rho) const;
    Matrix constraint_gradient(const Matrix& rho) const;

    /// Observer loss: Σ μ_j |⟨O_j⟩ − y_j|²
    float observer_loss(const Matrix& rho) const;
    Matrix observer_gradient(const Matrix& rho) const;

    /// Entropy: −γ S(ρ) where S = −Tr(ρ log ρ)
    float entropy_loss(const Matrix& rho) const;
    Matrix entropy_gradient(const Matrix& rho) const;

    /// Project onto density operator cone.
    static void project_density(Matrix& rho);

    std::vector<ConstraintOperator> constraints_;
    std::vector<ObserverOperator> observers_;
    float entropy_weight_;
    mutable std::vector<ContractionLog> log_;
    mutable uint32_t step_counter_ = 0;
};

} // namespace ciir_sim
