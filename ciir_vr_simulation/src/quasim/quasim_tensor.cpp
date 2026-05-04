/**
 * @file quasim_tensor.cpp
 * @brief Implementation of QuASIMTensor — multi-rank tensor with row-major storage.
 */

#include "quasim/quasim_tensor.hpp"

#include <cassert>
#include <cmath>
#include <random>
#include <stdexcept>

namespace ciir_sim {

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

QuASIMTensor::QuASIMTensor(std::vector<uint32_t> shape, std::string name,
                           Precision precision)
    : shape_(std::move(shape)), name_(std::move(name)), precision_(precision)
{
    assert(shape_.size() <= MAX_TENSOR_RANK);
    size_t total = 1;
    for (uint32_t d : shape_) total *= d;
    data_.resize(total, 0.0f);
    compute_strides();
}

QuASIMTensor::QuASIMTensor(std::vector<uint32_t> shape, std::vector<float> data,
                           std::string name)
    : shape_(std::move(shape)), data_(std::move(data)), name_(std::move(name)),
      precision_(Precision::FP32)
{
    assert(shape_.size() <= MAX_TENSOR_RANK);
    size_t total = 1;
    for (uint32_t d : shape_) total *= d;
    assert(data_.size() == total);
    compute_strides();
}

// ---------------------------------------------------------------------------
// Stride computation (row-major)
// ---------------------------------------------------------------------------

void QuASIMTensor::compute_strides() {
    const auto r = shape_.size();
    strides_.resize(r);
    if (r == 0) return;

    strides_[r - 1] = 1;
    for (size_t i = r - 1; i > 0; --i)
        strides_[i - 1] = strides_[i] * shape_[i];
}

size_t QuASIMTensor::flat_index(const std::vector<uint32_t>& indices) const {
    assert(indices.size() == shape_.size());
    size_t idx = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        assert(indices[i] < shape_[i]);
        idx += static_cast<size_t>(indices[i]) * strides_[i];
    }
    return idx;
}

// ---------------------------------------------------------------------------
// Element access by multi-index
// ---------------------------------------------------------------------------

float& QuASIMTensor::at(const std::vector<uint32_t>& indices) {
    return data_[flat_index(indices)];
}

float QuASIMTensor::at(const std::vector<uint32_t>& indices) const {
    return data_[flat_index(indices)];
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

float QuASIMTensor::frobenius_norm() const {
    float acc = 0.0f;
    for (float v : data_) acc += v * v;
    return std::sqrt(acc);
}

float QuASIMTensor::sum() const {
    float acc = 0.0f;
    for (float v : data_) acc += v;
    return acc;
}

float QuASIMTensor::max_abs() const {
    float m = 0.0f;
    for (float v : data_) m = std::max(m, std::abs(v));
    return m;
}

// ---------------------------------------------------------------------------
// In-place operations
// ---------------------------------------------------------------------------

void QuASIMTensor::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

void QuASIMTensor::randomise(uint32_t seed, float mean, float stddev) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(mean, stddev);
    for (float& v : data_) v = dist(rng);
}

void QuASIMTensor::scale(float alpha) {
    for (float& v : data_) v *= alpha;
}

void QuASIMTensor::axpy(float alpha, const QuASIMTensor& other) {
    assert(data_.size() == other.data_.size());
    for (size_t i = 0; i < data_.size(); ++i)
        data_[i] += alpha * other.data_[i];
}

// ---------------------------------------------------------------------------
// Reshape
// ---------------------------------------------------------------------------

void QuASIMTensor::reshape(std::vector<uint32_t> new_shape) {
    size_t total = 1;
    for (uint32_t d : new_shape) total *= d;
    assert(total == data_.size());
    shape_ = std::move(new_shape);
    compute_strides();
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

size_t QuASIMTensor::bytes() const {
    size_t elem_size = (precision_ == Precision::FP64) ? sizeof(double) : sizeof(float);
    return data_.size() * elem_size;
}

} // namespace ciir_sim
