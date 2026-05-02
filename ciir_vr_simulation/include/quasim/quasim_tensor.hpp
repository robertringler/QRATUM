/**
 * @file quasim_tensor.hpp
 * @brief GPU-resident multi-rank tensor with Vulkan buffer management.
 *
 * Tensor Layout Convention (§4 of formalization):
 *   State tensor:     X ∈ R^{B × R × D}
 *   Density tensor:   ρ ∈ R^{B × D × D}
 *   Constraint tensor: K_i ∈ R^{D × D}
 *   Observer tensor:  O_j ∈ R^{D × D}
 *
 * Supports rank-1 through rank-8 tensors with:
 *   - GPU buffer allocation (Vulkan storage buffers)
 *   - Precision modes (fp32, fp64)
 *   - Efficient memory management and reuse
 *   - Einsum-compatible dimension labeling
 */

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <array>
#include <cassert>
#include <cmath>
#include <numeric>
#include <functional>
#include <algorithm>

namespace ciir_sim {

/// Maximum supported tensor rank.
constexpr uint32_t MAX_TENSOR_RANK = 8;

/**
 * @brief Numerical precision mode for tensor storage.
 */
enum class Precision : uint8_t {
    FP32 = 0, ///< 32-bit float (default)
    FP64 = 1, ///< 64-bit double (high precision)
};

/**
 * @brief Multi-rank tensor with GPU buffer management.
 *
 * Stores data in row-major contiguous layout. Supports up to rank-8.
 * GPU buffer handle is managed externally by VulkanContext.
 */
class QuASIMTensor {
public:
    QuASIMTensor() = default;

    /**
     * @brief Construct tensor with given shape.
     * @param shape Dimension sizes (up to MAX_TENSOR_RANK).
     * @param name Human-readable name for debugging/export.
     * @param precision Numerical precision mode.
     */
    QuASIMTensor(std::vector<uint32_t> shape, std::string name = "",
                 Precision precision = Precision::FP32);

    /// Construct from existing data (copies).
    QuASIMTensor(std::vector<uint32_t> shape, std::vector<float> data,
                 std::string name = "");

    // ---- Shape ----
    [[nodiscard]] uint32_t rank() const { return static_cast<uint32_t>(shape_.size()); }
    [[nodiscard]] const std::vector<uint32_t>& shape() const { return shape_; }
    [[nodiscard]] uint32_t dim(uint32_t axis) const { return shape_.at(axis); }
    [[nodiscard]] size_t numel() const { return data_.size(); }

    // ---- Data access ----
    [[nodiscard]] float* data() { return data_.data(); }
    [[nodiscard]] const float* data() const { return data_.data(); }
    [[nodiscard]] std::vector<float>& storage() { return data_; }
    [[nodiscard]] const std::vector<float>& storage() const { return data_; }

    /// Element access by flat index.
    float& operator[](size_t idx) { return data_[idx]; }
    float  operator[](size_t idx) const { return data_[idx]; }

    /// Element access by multi-index.
    float& at(const std::vector<uint32_t>& indices);
    float  at(const std::vector<uint32_t>& indices) const;

    // ---- Operations ----
    /// Frobenius norm: √(Σ x²)
    [[nodiscard]] float frobenius_norm() const;

    /// Sum of all elements.
    [[nodiscard]] float sum() const;

    /// Max absolute value.
    [[nodiscard]] float max_abs() const;

    /// Fill with constant value.
    void fill(float value);

    /// Fill with random normal values.
    void randomise(uint32_t seed, float mean = 0.0f, float stddev = 1.0f);

    /// Scale all elements: x ← α·x
    void scale(float alpha);

    /// Axpy: this ← this + α·other
    void axpy(float alpha, const QuASIMTensor& other);

    /// Reshape (must preserve total element count).
    void reshape(std::vector<uint32_t> new_shape);

    // ---- Metadata ----
    [[nodiscard]] const std::string& name() const { return name_; }
    [[nodiscard]] Precision precision() const { return precision_; }
    [[nodiscard]] size_t bytes() const;

    /// GPU buffer handle (0 if not uploaded).
    uint32_t gpu_buffer = 0;

    /// Dimension labels for einsum clarity.
    std::vector<std::string> dim_labels;

private:
    std::vector<uint32_t> shape_;
    std::vector<float> data_;
    std::string name_;
    Precision precision_ = Precision::FP32;
    std::vector<size_t> strides_; ///< Row-major strides

    void compute_strides();
    [[nodiscard]] size_t flat_index(const std::vector<uint32_t>& indices) const;
};

} // namespace ciir_sim
