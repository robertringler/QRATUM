/**
 * @file ciir_observer.cpp
 * @brief Implementation of CIIRObserver — PVM construction, Born rule, measurement update.
 */

#include "ciir/ciir_observer.hpp"

#include <cmath>
#include <algorithm>
#include <numeric>

namespace ciir_sim {

// ---------------------------------------------------------------------------
// PVM — Projection-Valued Measure via spectral decomposition
// ---------------------------------------------------------------------------

PVM PVM::from_observable(const ObserverOperator& obs) {
    const uint32_t d = obs.matrix.dim;

    // Full Jacobi eigendecomposition
    constexpr uint32_t max_sweeps = 50;
    constexpr float    eps        = 1e-8f;

    std::vector<float> a = obs.matrix.data;
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
                    float vrp = V(r, p), vrq = V(r, q);
                    V(r, p) = c * vrp - s * vrq;
                    V(r, q) = s * vrp + c * vrq;
                }
            }
        }
    }

    // Group degenerate eigenvalues (within tolerance) into projectors
    constexpr float degen_tol = 1e-5f;

    // Collect raw eigenvalues with column indices
    struct EigPair {
        float value;
        uint32_t col;
    };
    std::vector<EigPair> pairs(d);
    for (uint32_t i = 0; i < d; ++i) pairs[i] = {A(i, i), i};
    std::sort(pairs.begin(), pairs.end(),
              [](const EigPair& a, const EigPair& b) { return a.value > b.value; });

    PVM pvm;
    uint32_t i = 0;
    while (i < d) {
        float val = pairs[i].value;
        std::vector<uint32_t> group;
        group.push_back(pairs[i].col);
        uint32_t j = i + 1;
        while (j < d && std::abs(pairs[j].value - val) < degen_tol) {
            group.push_back(pairs[j].col);
            ++j;
        }

        // Projector P_o = Σ_{k in group} |v_k><v_k|
        Matrix proj(d);
        for (uint32_t col : group) {
            for (uint32_t r = 0; r < d; ++r)
                for (uint32_t c_idx = 0; c_idx < d; ++c_idx)
                    proj(r, c_idx) += V(r, col) * V(c_idx, col);
        }

        // Average eigenvalue for the group
        float avg_val = 0.0f;
        for (uint32_t k = i; k < j; ++k) avg_val += pairs[k].value;
        avg_val /= static_cast<float>(j - i);

        pvm.eigenvalues.push_back(avg_val);
        pvm.projectors.push_back(std::move(proj));
        i = j;
    }

    return pvm;
}

std::vector<float> PVM::born_distribution(const Matrix& rho) const {
    std::vector<float> probs;
    probs.reserve(projectors.size());
    for (const auto& proj : projectors)
        probs.push_back(proj.trace_product(rho));
    return probs;
}

uint32_t PVM::most_probable(const Matrix& rho) const {
    auto probs = born_distribution(rho);
    return static_cast<uint32_t>(
        std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())));
}

// ---------------------------------------------------------------------------
// CIIRObserver
// ---------------------------------------------------------------------------

void CIIRObserver::set_observers(std::vector<ObserverOperator> observers) {
    observers_ = std::move(observers);
}

MeasurementResult CIIRObserver::measure(const Matrix& rho, uint32_t observer_index,
                                         uint32_t step, Matrix& rho_post) const
{
    const auto& obs = observers_[observer_index];
    PVM pvm = PVM::from_observable(obs);
    auto probs = pvm.born_distribution(rho);

    // Select most probable outcome (deterministic collapse)
    uint32_t outcome = pvm.most_probable(rho);

    float p_outcome = probs[outcome];

    // Pre-measurement entropy
    auto pre_eigs = rho.eigenvalues();
    float pre_entropy = 0.0f;
    for (float lam : pre_eigs)
        if (lam > 1e-12f)
            pre_entropy -= lam * std::log(lam);

    // Measurement update: ρ_post = P_o ρ P_o / Tr(P_o ρ)
    const Matrix& proj = pvm.projectors[outcome];
    Matrix temp = proj.multiply(rho);
    rho_post = temp.multiply(proj);

    if (p_outcome > 1e-12f)
        rho_post.scale(1.0f / p_outcome);

    // Post-measurement entropy
    auto post_eigs = rho_post.eigenvalues();
    float post_entropy = 0.0f;
    for (float lam : post_eigs)
        if (lam > 1e-12f)
            post_entropy -= lam * std::log(lam);

    MeasurementResult result{
        obs.name,
        step,
        outcome,
        pvm.eigenvalues[outcome],
        probs,
        pre_entropy,
        post_entropy
    };

    history_.push_back(result);
    return result;
}

std::vector<MeasurementResult> CIIRObserver::sequential_measure(
    const Matrix& rho, const std::vector<uint32_t>& observer_indices,
    uint32_t step, Matrix& rho_final) const
{
    std::vector<MeasurementResult> results;
    results.reserve(observer_indices.size());

    Matrix current = rho;
    for (uint32_t idx : observer_indices) {
        Matrix next(current.dim);
        auto result = measure(current, idx, step, next);
        results.push_back(std::move(result));
        current = std::move(next);
    }
    rho_final = std::move(current);
    return results;
}

std::vector<std::vector<float>> CIIRObserver::commutator_matrix() const {
    const auto n = observers_.size();
    std::vector<std::vector<float>> mat(n, std::vector<float>(n, 0.0f));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j) {
            float norm = observers_[i].commutator_norm(observers_[j]);
            mat[i][j] = norm;
            mat[j][i] = norm;
        }
    return mat;
}

std::vector<float> CIIRObserver::expectations(const Matrix& rho) const {
    std::vector<float> vals;
    vals.reserve(observers_.size());
    for (const auto& obs : observers_)
        vals.push_back(obs.expectation(rho));
    return vals;
}

} // namespace ciir_sim
