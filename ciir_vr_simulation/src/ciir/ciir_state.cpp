/**
 * @file ciir_state.cpp
 * @brief Implementation of CIIRState — density matrices, constraints, and manifold embedding.
 */

#include "ciir/ciir_state.hpp"

#include <cmath>
#include <limits>

namespace ciir_sim {

// ---------------------------------------------------------------------------
// Matrix::eigenvalues — Jacobi eigenvalue algorithm for small symmetric matrices
// ---------------------------------------------------------------------------

std::vector<float> Matrix::eigenvalues() const {
    constexpr uint32_t max_sweeps = 50;
    constexpr float    eps        = 1e-8f;

    const uint32_t n = dim;
    if (n == 0) return {};

    // Work on a mutable copy
    std::vector<float> a = data;

    auto A = [&](uint32_t r, uint32_t c) -> float& { return a[r * n + c]; };

    for (uint32_t sweep = 0; sweep < max_sweeps; ++sweep) {
        // Compute off-diagonal norm
        float off_norm = 0.0f;
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t j = i + 1; j < n; ++j)
                off_norm += A(i, j) * A(i, j);

        if (off_norm < eps * eps) break;

        for (uint32_t p = 0; p < n; ++p) {
            for (uint32_t q = p + 1; q < n; ++q) {
                float apq = A(p, q);
                if (std::abs(apq) < eps) continue;

                float app = A(p, p);
                float aqq = A(q, q);
                float tau = (aqq - app) / (2.0f * apq);
                float t   = (tau >= 0.0f ? 1.0f : -1.0f) /
                            (std::abs(tau) + std::sqrt(1.0f + tau * tau));
                float c   = 1.0f / std::sqrt(1.0f + t * t);
                float s   = t * c;

                // Update diagonal elements
                A(p, p) = app - t * apq;
                A(q, q) = aqq + t * apq;
                A(p, q) = 0.0f;
                A(q, p) = 0.0f;

                // Rotate remaining rows/columns
                for (uint32_t r = 0; r < n; ++r) {
                    if (r == p || r == q) continue;
                    float arp = A(r, p);
                    float arq = A(r, q);
                    A(r, p) = c * arp - s * arq;
                    A(p, r) = A(r, p);
                    A(r, q) = s * arp + c * arq;
                    A(q, r) = A(r, q);
                }
            }
        }
    }

    std::vector<float> eigs(n);
    for (uint32_t i = 0; i < n; ++i) eigs[i] = A(i, i);
    std::sort(eigs.begin(), eigs.end(), std::greater<float>());
    return eigs;
}

// ---------------------------------------------------------------------------
// CIIRState
// ---------------------------------------------------------------------------

static Matrix random_density(uint32_t d, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    Matrix x(d);
    for (uint32_t i = 0; i < d; ++i)
        for (uint32_t j = 0; j < d; ++j)
            x(i, j) = normal(rng);

    // ρ = X^T X / Tr(X^T X)
    Matrix rho(d);
    for (uint32_t i = 0; i < d; ++i)
        for (uint32_t j = 0; j < d; ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < d; ++k)
                sum += x(k, i) * x(k, j);
            rho(i, j) = sum;
        }

    float tr = rho.trace();
    if (tr > 0.0f) rho.scale(1.0f / tr);
    return rho;
}

CIIRState::CIIRState(uint32_t batch_size, uint32_t rep_dim, uint32_t seed)
    : batch_size_(batch_size), rep_dim_(rep_dim), rng_(seed)
{
    rho_.reserve(batch_size_);
    for (uint32_t b = 0; b < batch_size_; ++b)
        rho_.push_back(random_density(rep_dim_, rng_));
}

void CIIRState::randomise(uint32_t seed) {
    rng_.seed(seed);
    rho_.clear();
    rho_.reserve(batch_size_);
    for (uint32_t b = 0; b < batch_size_; ++b)
        rho_.push_back(random_density(rep_dim_, rng_));
}

void CIIRState::set_density_matrices(const std::vector<Matrix>& rho_batch) {
    assert(rho_batch.size() == batch_size_);
    rho_ = rho_batch;
}

void CIIRState::project_density() {
    for (auto& rho : rho_) {
        rho.symmetrise();

        // Eigenvalue clipping: replace negative eigenvalues with 0
        // then reconstruct via spectral projection.
        // For small matrices we do an in-place fixup using Jacobi.
        const uint32_t d = rho.dim;

        // Full Jacobi to get eigenvalues and eigenvectors
        constexpr uint32_t max_sweeps = 50;
        constexpr float    eps        = 1e-8f;

        std::vector<float> a = rho.data;
        // Eigenvector matrix V (initially identity)
        std::vector<float> v(d * d, 0.0f);
        for (uint32_t i = 0; i < d; ++i) v[i * d + i] = 1.0f;

        auto A = [&](uint32_t r, uint32_t c) -> float& { return a[r * d + c]; };
        auto V = [&](uint32_t r, uint32_t c) -> float& { return v[r * d + c]; };

        for (uint32_t sweep = 0; sweep < max_sweeps; ++sweep) {
            float off_norm = 0.0f;
            for (uint32_t i = 0; i < d; ++i)
                for (uint32_t j = i + 1; j < d; ++j)
                    off_norm += A(i, j) * A(i, j);
            if (off_norm < eps * eps) break;

            for (uint32_t p = 0; p < d; ++p) {
                for (uint32_t q = p + 1; q < d; ++q) {
                    float apq = A(p, q);
                    if (std::abs(apq) < eps) continue;

                    float tau = (A(q, q) - A(p, p)) / (2.0f * apq);
                    float t   = (tau >= 0.0f ? 1.0f : -1.0f) /
                                (std::abs(tau) + std::sqrt(1.0f + tau * tau));
                    float c   = 1.0f / std::sqrt(1.0f + t * t);
                    float s   = t * c;

                    A(p, p) -= t * apq;
                    A(q, q) += t * apq;
                    A(p, q) = 0.0f;
                    A(q, p) = 0.0f;

                    for (uint32_t r = 0; r < d; ++r) {
                        if (r != p && r != q) {
                            float arp = A(r, p), arq = A(r, q);
                            A(r, p) = c * arp - s * arq;
                            A(p, r) = A(r, p);
                            A(r, q) = s * arp + c * arq;
                            A(q, r) = A(r, q);
                        }
                        // Accumulate eigenvectors
                        float vrp = V(r, p), vrq = V(r, q);
                        V(r, p) = c * vrp - s * vrq;
                        V(r, q) = s * vrp + c * vrq;
                    }
                }
            }
        }

        // Clip negative eigenvalues, reconstruct: ρ = V diag(λ_clipped) V^T
        float trace_sum = 0.0f;
        for (uint32_t i = 0; i < d; ++i) {
            if (A(i, i) < 0.0f) A(i, i) = 0.0f;
            trace_sum += A(i, i);
        }

        if (trace_sum < 1e-12f) {
            // Degenerate — reset to maximally mixed state
            rho = Matrix(d);
            for (uint32_t i = 0; i < d; ++i) rho(i, i) = 1.0f / static_cast<float>(d);
            continue;
        }

        // Normalize eigenvalues so they sum to 1
        for (uint32_t i = 0; i < d; ++i) A(i, i) /= trace_sum;

        // Reconstruct: ρ_ij = Σ_k λ_k V_{ik} V_{jk}
        for (uint32_t i = 0; i < d; ++i)
            for (uint32_t j = 0; j < d; ++j) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < d; ++k)
                    sum += A(k, k) * V(i, k) * V(j, k);
                rho(i, j) = sum;
            }
    }
}

float CIIRState::purity(uint32_t b) const {
    return rho_[b].trace_product(rho_[b]);
}

float CIIRState::entropy(uint32_t b) const {
    auto eigs = rho_[b].eigenvalues();
    float s = 0.0f;
    for (float lam : eigs) {
        if (lam > 1e-12f)
            s -= lam * std::log(lam);
    }
    return s;
}

std::array<float, 3> CIIRState::manifold_embedding_3d(uint32_t b) const {
    auto eigs = rho_[b].eigenvalues();
    std::array<float, 3> emb = {0.0f, 0.0f, 0.0f};
    for (uint32_t i = 0; i < std::min(static_cast<uint32_t>(eigs.size()), 3u); ++i)
        emb[i] = eigs[i];
    return emb;
}

std::vector<CIIRState::SurfaceSample> CIIRState::constraint_surface_samples(
    const std::vector<ConstraintOperator>& constraints,
    uint32_t n_samples,
    uint32_t step_seed) const
{
    std::vector<SurfaceSample> samples;
    samples.reserve(n_samples);

    auto local_rng = rng_;
    if (step_seed != 0) local_rng.seed(step_seed);
    std::normal_distribution<float> normal(0.0f, 0.05f);

    const Matrix& base = rho_[0];

    for (uint32_t s = 0; s < n_samples; ++s) {
        Matrix perturbed = base;
        for (auto& v : perturbed.data) v += normal(local_rng);
        perturbed.symmetrise();

        // Normalize trace to 1
        float tr = perturbed.trace();
        if (std::abs(tr) > 1e-12f) perturbed.scale(1.0f / tr);

        // Compute embedding
        auto eigs = perturbed.eigenvalues();
        std::array<float, 3> pos = {0.0f, 0.0f, 0.0f};
        for (uint32_t i = 0; i < std::min(static_cast<uint32_t>(eigs.size()), 3u); ++i)
            pos[i] = eigs[i];

        // Compute total violation
        float violation = 0.0f;
        for (const auto& c : constraints)
            violation += c.violation(perturbed);

        samples.push_back({pos, violation});
    }
    return samples;
}

void CIIRState::set_constraints(std::vector<ConstraintOperator> constraints) {
    constraints_ = std::move(constraints);
}

std::vector<float> CIIRState::evaluate_constraints(uint32_t b) const {
    std::vector<float> vals;
    vals.reserve(constraints_.size());
    for (const auto& c : constraints_)
        vals.push_back(c.evaluate(rho_[b]));
    return vals;
}

float CIIRState::total_violation(uint32_t b) const {
    float total = 0.0f;
    for (const auto& c : constraints_)
        total += c.violation(rho_[b]);
    return total;
}

} // namespace ciir_sim
