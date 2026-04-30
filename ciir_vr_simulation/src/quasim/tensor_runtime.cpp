/**
 * @file tensor_runtime.cpp
 * @brief Implementation of TensorRuntime — loss computation, gradient evolution,
 *        and projected density updates for the QuASIM tensor pipeline.
 */

#include "quasim/tensor_runtime.hpp"

#include <cassert>
#include <cmath>
#include <chrono>
#include <algorithm>

namespace ciir_sim {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

TensorRuntime::TensorRuntime(const std::vector<ConstraintOperator>& constraints,
                             const std::vector<ObserverOperator>& observers,
                             float entropy_weight)
    : constraints_(constraints), observers_(observers),
      entropy_weight_(entropy_weight)
{}

// ---------------------------------------------------------------------------
// Jacobi eigendecomposition (eigenvalues + eigenvectors) for small symmetric
// matrices.  Returns (eigenvalues, column-major eigenvector flat array).
// ---------------------------------------------------------------------------

static std::pair<std::vector<float>, std::vector<float>>
jacobi_eigen(const Matrix& mat) {
    constexpr uint32_t max_sweeps = 50;
    constexpr float    eps        = 1e-8f;

    const uint32_t d = mat.dim;
    if (d == 0) return {{}, {}};

    std::vector<float> a = mat.data;                      // work copy
    std::vector<float> v(d * d, 0.0f);                    // eigenvectors (V)
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
                    float vrp = V(r, p), vrq = V(r, q);
                    V(r, p) = c * vrp - s * vrq;
                    V(r, q) = s * vrp + c * vrq;
                }
            }
        }
    }

    std::vector<float> eigs(d);
    for (uint32_t i = 0; i < d; ++i) eigs[i] = A(i, i);
    return {eigs, v};
}

// ---------------------------------------------------------------------------
// Loss terms
// ---------------------------------------------------------------------------

float TensorRuntime::constraint_loss(const Matrix& rho) const {
    float loss = 0.0f;
    for (const auto& c : constraints_)
        loss += c.weight * c.violation(rho);
    return loss;
}

float TensorRuntime::observer_loss(const Matrix& rho) const {
    float loss = 0.0f;
    for (const auto& o : observers_)
        loss += o.loss(rho);
    return loss;
}

float TensorRuntime::entropy_loss(const Matrix& rho) const {
    // S(ρ) = -Tr(ρ log ρ) computed via eigendecomposition
    auto eigs = rho.eigenvalues();
    float s = 0.0f;
    for (float lam : eigs) {
        if (lam > 1e-12f)
            s -= lam * std::log(lam);
    }
    // Entropy loss = -γ S(ρ)  (we want to *maximise* entropy, so loss is negative)
    return -entropy_weight_ * s;
}

// ---------------------------------------------------------------------------
// Gradient terms
// ---------------------------------------------------------------------------

Matrix TensorRuntime::constraint_gradient(const Matrix& rho) const {
    Matrix grad(rho.dim);
    for (const auto& c : constraints_) {
        Matrix g = c.gradient(rho);       // 2 C_i(ρ) K_i
        grad.add_scaled(g, c.weight);
    }
    return grad;
}

Matrix TensorRuntime::observer_gradient(const Matrix& rho) const {
    Matrix grad(rho.dim);
    for (const auto& o : observers_)
        grad.add_scaled(o.gradient(rho), 1.0f);
    return grad;
}

Matrix TensorRuntime::entropy_gradient(const Matrix& rho) const {
    // ∂(-γ S)/∂ρ = γ (log ρ + I)
    // Compute log ρ via eigendecomposition: log ρ = V diag(log λ) V^T
    const uint32_t d = rho.dim;
    auto [eigs, vflat] = jacobi_eigen(rho);

    auto V = [&](uint32_t r, uint32_t c) -> float { return vflat[r * d + c]; };

    // Build log_rho = V diag(log(max(λ, ε))) V^T
    Matrix log_rho(d);
    for (uint32_t i = 0; i < d; ++i) {
        float log_lam = (eigs[i] > 1e-12f) ? std::log(eigs[i]) : std::log(1e-12f);
        for (uint32_t r = 0; r < d; ++r)
            for (uint32_t c = 0; c < d; ++c)
                log_rho(r, c) += log_lam * V(r, i) * V(c, i);
    }

    // Gradient = γ (log ρ + I)
    Matrix grad = log_rho;
    for (uint32_t i = 0; i < d; ++i)
        grad(i, i) += 1.0f;
    grad.scale(entropy_weight_);
    return grad;
}

// ---------------------------------------------------------------------------
// Combined loss and gradient
// ---------------------------------------------------------------------------

LossComponents TensorRuntime::compute_loss(const std::vector<Matrix>& rho_batch) const {
    LossComponents lc;
    for (const auto& rho : rho_batch) {
        lc.constraint += constraint_loss(rho);
        lc.observer   += observer_loss(rho);
        lc.entropy    += entropy_loss(rho);
    }
    float inv_b = 1.0f / static_cast<float>(rho_batch.size());
    lc.constraint *= inv_b;
    lc.observer   *= inv_b;
    lc.entropy    *= inv_b;
    lc.total = lc.constraint + lc.observer + lc.entropy;
    return lc;
}

std::vector<Matrix> TensorRuntime::compute_gradient(
    const std::vector<Matrix>& rho_batch) const
{
    std::vector<Matrix> grads;
    grads.reserve(rho_batch.size());
    for (const auto& rho : rho_batch) {
        Matrix g = constraint_gradient(rho);
        g.add_scaled(observer_gradient(rho), 1.0f);
        g.add_scaled(entropy_gradient(rho), 1.0f);
        grads.push_back(std::move(g));
    }
    return grads;
}

// ---------------------------------------------------------------------------
// Evolution step
// ---------------------------------------------------------------------------

LossComponents TensorRuntime::step(std::vector<Matrix>& rho_batch, float lr,
                                   IntegrationMethod method) {
    auto t0 = std::chrono::steady_clock::now();

    auto grads = compute_gradient(rho_batch);
    LossComponents lc = compute_loss(rho_batch);

    float grad_norm = 0.0f;
    for (size_t b = 0; b < rho_batch.size(); ++b) {
        // ρ ← ρ − η ∇L
        rho_batch[b].add_scaled(grads[b], -lr);
        grad_norm += grads[b].frobenius_norm();

        if (method == IntegrationMethod::PROJECTED_GRADIENT ||
            method == IntegrationMethod::STRANG_SPLITTING) {
            project_density(rho_batch[b]);
        }
    }
    grad_norm /= static_cast<float>(rho_batch.size());

    auto t1 = std::chrono::steady_clock::now();
    float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    log_.push_back(ContractionLog{
        step_counter_++,
        lc.total,
        grad_norm,
        lc,
        ms
    });

    return lc;
}

// ---------------------------------------------------------------------------
// Batch evaluation helpers
// ---------------------------------------------------------------------------

std::vector<std::vector<float>> TensorRuntime::eval_constraints(
    const std::vector<Matrix>& rho_batch) const
{
    std::vector<std::vector<float>> result;
    result.reserve(rho_batch.size());
    for (const auto& rho : rho_batch) {
        std::vector<float> vals;
        vals.reserve(constraints_.size());
        for (const auto& c : constraints_)
            vals.push_back(c.evaluate(rho));
        result.push_back(std::move(vals));
    }
    return result;
}

std::vector<std::vector<float>> TensorRuntime::eval_observers(
    const std::vector<Matrix>& rho_batch) const
{
    std::vector<std::vector<float>> result;
    result.reserve(rho_batch.size());
    for (const auto& rho : rho_batch) {
        std::vector<float> vals;
        vals.reserve(observers_.size());
        for (const auto& o : observers_)
            vals.push_back(o.expectation(rho));
        result.push_back(std::move(vals));
    }
    return result;
}

// ---------------------------------------------------------------------------
// 2D loss landscape sampling
// ---------------------------------------------------------------------------

std::vector<std::vector<float>> TensorRuntime::loss_landscape_2d(
    const Matrix& rho_center,
    const Matrix& dir1, const Matrix& dir2,
    uint32_t n_grid, float extent) const
{
    std::vector<std::vector<float>> grid(n_grid, std::vector<float>(n_grid, 0.0f));

    for (uint32_t i = 0; i < n_grid; ++i) {
        float alpha = -extent + 2.0f * extent * static_cast<float>(i) /
                      static_cast<float>(n_grid - 1);
        for (uint32_t j = 0; j < n_grid; ++j) {
            float beta = -extent + 2.0f * extent * static_cast<float>(j) /
                         static_cast<float>(n_grid - 1);

            Matrix rho = rho_center;
            rho.add_scaled(dir1, alpha);
            rho.add_scaled(dir2, beta);

            // Evaluate single-element batch
            std::vector<Matrix> batch = {rho};
            auto lc = compute_loss(batch);
            grid[i][j] = lc.total;
        }
    }
    return grid;
}

// ---------------------------------------------------------------------------
// Density projection (eigenvalue clipping)
// ---------------------------------------------------------------------------

void TensorRuntime::project_density(Matrix& rho) {
    rho.symmetrise();
    const uint32_t d = rho.dim;

    auto [eigs, vflat] = jacobi_eigen(rho);

    auto V = [&](uint32_t r, uint32_t c) -> float { return vflat[r * d + c]; };

    // Clip negative eigenvalues
    float trace_sum = 0.0f;
    for (uint32_t i = 0; i < d; ++i) {
        if (eigs[i] < 0.0f) eigs[i] = 0.0f;
        trace_sum += eigs[i];
    }

    if (trace_sum < 1e-12f) {
        // Degenerate — reset to maximally mixed state
        rho = Matrix(d);
        for (uint32_t i = 0; i < d; ++i) rho(i, i) = 1.0f / static_cast<float>(d);
        return;
    }

    // Normalize eigenvalues to sum to 1
    for (uint32_t i = 0; i < d; ++i) eigs[i] /= trace_sum;

    // Reconstruct: ρ_ij = Σ_k λ_k V_{ik} V_{jk}
    for (uint32_t r = 0; r < d; ++r)
        for (uint32_t c = 0; c < d; ++c) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < d; ++k)
                sum += eigs[k] * V(r, k) * V(c, k);
            rho(r, c) = sum;
        }
}

// ---------------------------------------------------------------------------
// Parameter control
// ---------------------------------------------------------------------------

void TensorRuntime::set_constraint_weight(uint32_t idx, float weight) {
    assert(idx < constraints_.size());
    constraints_[idx].weight = weight;
}

void TensorRuntime::set_observer_target(uint32_t idx, float target) {
    assert(idx < observers_.size());
    observers_[idx].target = target;
}

} // namespace ciir_sim
