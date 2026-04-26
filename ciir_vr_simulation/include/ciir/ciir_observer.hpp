/**
 * @file ciir_observer.hpp
 * @brief Non-commutative observer operators with sequential measurement effects.
 *
 * Implements the CIIR observer algebra from D3.20, T5.1, T8.2–T8.7:
 *
 *   [O_i, O_j] = O_i O_j − O_j O_i ≠ 0  (non-commutativity)
 *
 *   Born rule (T8.2):  p(o) = Tr(P_o ρ)
 *   Measurement update (T8.3):  ρ → P_o ρ P_o / Tr(P_o ρ)
 *   Interference (T8.6):  E_{ψ±}[O] = ½(E₁ + E₂) ± Re⟨ψ₁|O|ψ₂⟩
 *   Contextuality (T8.7): measurement order matters for non-commuting O
 */

#pragma once

#include "ciir/ciir_state.hpp"

#include <string>
#include <vector>
#include <array>

namespace ciir_sim {

/**
 * @brief Observer operator O_j ∈ A(H_C).
 *
 * Self-adjoint operator on H_C implementing the observable algebra (D3.20).
 * Non-commutativity follows from T5.1 (holonomy of constraint curvature).
 */
struct ObserverOperator {
    Matrix matrix;      ///< Self-adjoint observable matrix O_j (D×D)
    std::string name;
    float weight = 1.0f; ///< Observer loss weight μ_j
    float target = 0.0f; ///< Target value y_j

    ObserverOperator() = default;
    ObserverOperator(Matrix m, std::string n, float w = 1.0f, float t = 0.0f)
        : matrix(std::move(m)), name(std::move(n)), weight(w), target(t) {
        matrix.symmetrise(); // Enforce self-adjointness (D3.20)
    }

    /// Expectation value ⟨O_j⟩_ρ = Tr(O_j ρ)
    [[nodiscard]] float expectation(const Matrix& rho) const {
        return matrix.trace_product(rho);
    }

    /// Residual: ⟨O_j⟩ − y_j
    [[nodiscard]] float residual(const Matrix& rho) const {
        return expectation(rho) - target;
    }

    /// Observer loss: μ_j |⟨O_j⟩ − y_j|²
    [[nodiscard]] float loss(const Matrix& rho) const {
        float r = residual(rho);
        return weight * r * r;
    }

    /// ∂L_O/∂ρ = 2 μ_j (⟨O_j⟩ − y_j) O_j
    [[nodiscard]] Matrix gradient(const Matrix& rho) const {
        float r = residual(rho);
        Matrix g = matrix;
        g.scale(2.0f * weight * r);
        return g;
    }

    /// Commutator [O_i, O_j] = O_i O_j − O_j O_i
    [[nodiscard]] Matrix commutator(const ObserverOperator& other) const {
        Matrix ab = matrix.multiply(other.matrix);
        Matrix ba = other.matrix.multiply(matrix);
        for (size_t i = 0; i < ab.data.size(); ++i)
            ab.data[i] -= ba.data[i];
        return ab;
    }

    /// ‖[O_i, O_j]‖_F — non-commutativity measure
    [[nodiscard]] float commutator_norm(const ObserverOperator& other) const {
        return commutator(other).frobenius_norm();
    }
};

/**
 * @brief Projection-Valued Measure from spectral decomposition (T8.1).
 */
struct PVM {
    std::vector<float>  eigenvalues;   ///< Distinct eigenvalues {a_o}
    std::vector<Matrix> projectors;    ///< Orthogonal projectors {P_o}

    /// Construct PVM from an observer's spectral decomposition.
    static PVM from_observable(const ObserverOperator& obs);

    /// Born probability: p(o) = Tr(P_o ρ)
    [[nodiscard]] std::vector<float> born_distribution(const Matrix& rho) const;

    /// Most probable outcome index.
    [[nodiscard]] uint32_t most_probable(const Matrix& rho) const;
};

/**
 * @brief Measurement result from a single observation.
 */
struct MeasurementResult {
    std::string observer_name;
    uint32_t    step;
    uint32_t    outcome_index;
    float       outcome_value;
    std::vector<float> probabilities;
    float       pre_entropy;
    float       post_entropy;
};

/**
 * @brief CIIR Observer manager with sequential measurement tracking.
 *
 * Tracks measurement history, computes commutator matrices, and
 * implements sequential measurement effects where order matters.
 */
class CIIRObserver {
public:
    CIIRObserver() = default;

    /// Set the observer operators.
    void set_observers(std::vector<ObserverOperator> observers);

    /// Perform measurement on a single density matrix (collapses state).
    MeasurementResult measure(const Matrix& rho, uint32_t observer_index,
                              uint32_t step, Matrix& rho_post) const;

    /// Sequential measurement: apply observers in given order.
    std::vector<MeasurementResult> sequential_measure(
        const Matrix& rho, const std::vector<uint32_t>& observer_indices,
        uint32_t step, Matrix& rho_final) const;

    /// Compute commutator norm matrix ‖[O_i, O_j]‖ for all pairs.
    [[nodiscard]] std::vector<std::vector<float>> commutator_matrix() const;

    /// Expectation values ⟨O_j⟩_ρ for all observers.
    [[nodiscard]] std::vector<float> expectations(const Matrix& rho) const;

    /// Access observers
    [[nodiscard]] const std::vector<ObserverOperator>& observers() const { return observers_; }
    [[nodiscard]] const std::vector<MeasurementResult>& history() const { return history_; }

    /// Clear measurement history.
    void clear_history() { history_.clear(); }

private:
    std::vector<ObserverOperator> observers_;
    mutable std::vector<MeasurementResult> history_;
};

} // namespace ciir_sim
