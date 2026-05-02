/**
 * @file ciir_state.hpp
 * @brief GPU-resident CIIR manifold state with dynamic constraints.
 *
 * Mathematical Objects (Ch3–Ch5 of CIIR monograph):
 *
 *   State manifold M ⊆ R^n as tensor space T^{r,d}:
 *     X ∈ R^{B × R × D}, ρ = X^T X / Tr(X^T X)
 *
 *   Density operator: ρ ∈ D(H_C) with ρ ≥ 0, Tr(ρ) = 1
 *
 *   Constraint surface:
 *     S_C = {ρ ∈ D(H_C) : C_i(ρ) = 0 ∀i}
 *     where C_i(ρ) = Tr(K_i · ρ) for Hermitian K_i
 *
 *   Interface map: Π: M → I with contraction bound (T8.4)
 */

#pragma once

#include <cstdint>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>
#include <cassert>

namespace ciir_sim {

/**
 * @brief Dense symmetric matrix stored row-major, dimension D×D.
 *
 * Used for constraint kernels K_i, observer matrices O_j, and
 * density matrices ρ.
 */
struct Matrix {
    uint32_t dim = 0;
    std::vector<float> data; ///< Row-major D×D

    Matrix() = default;
    explicit Matrix(uint32_t d) : dim(d), data(d * d, 0.0f) {}
    Matrix(uint32_t d, const std::vector<float>& vals) : dim(d), data(vals) {
        assert(vals.size() == static_cast<size_t>(d) * d);
    }

    float& operator()(uint32_t r, uint32_t c) { return data[r * dim + c]; }
    float  operator()(uint32_t r, uint32_t c) const { return data[r * dim + c]; }

    /// Frobenius norm: √(Σ a²_ij)
    [[nodiscard]] float frobenius_norm() const {
        float sum = 0.0f;
        for (float v : data) sum += v * v;
        return std::sqrt(sum);
    }

    /// Trace: Σ a_ii
    [[nodiscard]] float trace() const {
        float t = 0.0f;
        for (uint32_t i = 0; i < dim; ++i) t += (*this)(i, i);
        return t;
    }

    /// Symmetrise in-place: A ← (A + A^T) / 2
    void symmetrise() {
        for (uint32_t i = 0; i < dim; ++i)
            for (uint32_t j = i + 1; j < dim; ++j) {
                float avg = 0.5f * ((*this)(i, j) + (*this)(j, i));
                (*this)(i, j) = avg;
                (*this)(j, i) = avg;
            }
    }

    /// Matrix multiply: C = A · B (both D×D)
    [[nodiscard]] Matrix multiply(const Matrix& other) const {
        assert(dim == other.dim);
        Matrix result(dim);
        for (uint32_t i = 0; i < dim; ++i)
            for (uint32_t j = 0; j < dim; ++j) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < dim; ++k)
                    sum += (*this)(i, k) * other(k, j);
                result(i, j) = sum;
            }
        return result;
    }

    /// Tr(A · B)
    [[nodiscard]] float trace_product(const Matrix& other) const {
        assert(dim == other.dim);
        float t = 0.0f;
        for (uint32_t i = 0; i < dim; ++i)
            for (uint32_t k = 0; k < dim; ++k)
                t += (*this)(i, k) * other(k, i);
        return t;
    }

    /// Eigenvalues via Jacobi iteration (for small symmetric matrices)
    [[nodiscard]] std::vector<float> eigenvalues() const;

    /// Scale in-place: A ← α·A
    void scale(float alpha) {
        for (float& v : data) v *= alpha;
    }

    /// A ← A + α·B
    void add_scaled(const Matrix& other, float alpha) {
        assert(dim == other.dim);
        for (size_t i = 0; i < data.size(); ++i)
            data[i] += alpha * other.data[i];
    }
};

/**
 * @brief Constraint operator C_i: D(H_C) → R.
 *
 * Implements the constraint operator algebra K(H_C) (D5.2).
 * C_i(ρ) = Tr(K_i · ρ) for Hermitian K_i.
 */
struct ConstraintOperator {
    Matrix kernel;     ///< Hermitian constraint kernel K_i (D×D)
    std::string name;
    float weight = 1.0f; ///< Lagrange multiplier λ_i

    ConstraintOperator() = default;
    ConstraintOperator(Matrix k, std::string n, float w = 1.0f)
        : kernel(std::move(k)), name(std::move(n)), weight(w) {
        kernel.symmetrise(); // Enforce Hermiticity (Ax4.2)
    }

    /// C_i(ρ) = Tr(K_i · ρ)
    [[nodiscard]] float evaluate(const Matrix& rho) const {
        return kernel.trace_product(rho);
    }

    /// C_i(ρ)²
    [[nodiscard]] float violation(const Matrix& rho) const {
        float c = evaluate(rho);
        return c * c;
    }

    /// ∂(C_i²)/∂ρ = 2 C_i(ρ) · K_i
    [[nodiscard]] Matrix gradient(const Matrix& rho) const {
        float c = evaluate(rho);
        Matrix g = kernel;
        g.scale(2.0f * c);
        return g;
    }

    /// Commutator [K_i, K_j] = K_i K_j − K_j K_i
    [[nodiscard]] Matrix commutator(const ConstraintOperator& other) const {
        Matrix ab = kernel.multiply(other.kernel);
        Matrix ba = other.kernel.multiply(kernel);
        for (size_t i = 0; i < ab.data.size(); ++i)
            ab.data[i] -= ba.data[i];
        return ab;
    }
};

/**
 * @brief GPU-resident CIIR manifold state.
 *
 * Manages a batch of density matrices ρ ∈ R^{B×D×D} with:
 * - Dynamic constraint evaluation
 * - Eigenvalue caching for 3D embedding
 * - Purity and entropy tracking
 * - Constraint surface sampling for visualization
 */
class CIIRState {
public:
    /**
     * @brief Construct CIIR state with given dimensions.
     * @param batch_size Number of parallel states B.
     * @param rep_dim Cognitive representation dimension D.
     * @param seed Random seed.
     */
    CIIRState(uint32_t batch_size, uint32_t rep_dim, uint32_t seed = 42);

    /// Reinitialise with random density matrices on the unit sphere.
    void randomise(uint32_t seed);

    /// Update density matrices (called after evolution step).
    void set_density_matrices(const std::vector<Matrix>& rho_batch);

    /// Project onto valid density operator cone D(H_C).
    void project_density();

    // ---- Accessors ----
    [[nodiscard]] uint32_t batch_size() const { return batch_size_; }
    [[nodiscard]] uint32_t dim() const { return rep_dim_; }
    [[nodiscard]] const std::vector<Matrix>& density_matrices() const { return rho_; }
    [[nodiscard]] Matrix& density_matrix(uint32_t b) { return rho_[b]; }
    [[nodiscard]] const Matrix& density_matrix(uint32_t b) const { return rho_[b]; }

    /// Purity Tr(ρ²) for batch element b.
    [[nodiscard]] float purity(uint32_t b) const;

    /// Von Neumann entropy S(ρ) = −Tr(ρ log ρ) for batch element b.
    [[nodiscard]] float entropy(uint32_t b) const;

    /// 3D manifold embedding (first 3 eigenvalues) for batch element b.
    [[nodiscard]] std::array<float, 3> manifold_embedding_3d(uint32_t b) const;

    /// Sample constraint surface points around batch element 0.
    struct SurfaceSample {
        std::array<float, 3> position;
        float violation;
    };
    [[nodiscard]] std::vector<SurfaceSample> constraint_surface_samples(
        const std::vector<ConstraintOperator>& constraints,
        uint32_t n_samples = 100,
        uint32_t step_seed = 0) const;

    /// Add constraints for tracking
    void set_constraints(std::vector<ConstraintOperator> constraints);
    [[nodiscard]] const std::vector<ConstraintOperator>& constraints() const { return constraints_; }

    /// Evaluate all constraints for batch element b. Returns per-constraint values.
    [[nodiscard]] std::vector<float> evaluate_constraints(uint32_t b) const;

    /// Total constraint violation Σ C_i(ρ)² for batch element b.
    [[nodiscard]] float total_violation(uint32_t b) const;

    /// GPU buffer handle (Vulkan buffer index, or 0 if not uploaded).
    uint32_t gpu_buffer = 0;

private:
    uint32_t batch_size_;
    uint32_t rep_dim_;
    std::vector<Matrix> rho_;
    std::vector<ConstraintOperator> constraints_;
    mutable std::mt19937 rng_;
};

} // namespace ciir_sim
