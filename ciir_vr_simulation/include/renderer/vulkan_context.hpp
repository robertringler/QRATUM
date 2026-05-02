/**
 * @file vulkan_context.hpp
 * @brief Vulkan 1.3+ rendering context with compute shader support.
 *
 * Manages the Vulkan instance, device, swapchain, render passes,
 * and compute pipeline for tensor operations.
 *
 * Pipeline layout:
 *   1. Compute pass: tensor_contraction.comp, gradient_update.comp
 *   2. Graphics pass: manifold.vert/frag, heatmap.vert/frag
 *   3. UI pass: Dear ImGui overlay
 */

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <functional>

namespace ciir_sim {

/**
 * @brief Vulkan buffer handle with metadata.
 */
struct VulkanBuffer {
    uint32_t handle = 0;   ///< Opaque handle (VkBuffer internally)
    size_t   size   = 0;   ///< Buffer size in bytes
    bool     device_local = false;
};

/**
 * @brief Shader module descriptor.
 */
struct ShaderModule {
    std::string path;         ///< Path to SPIR-V file
    std::string entry_point = "main";
    enum class Stage : uint8_t { VERTEX, FRAGMENT, COMPUTE } stage;
};

/**
 * @brief Vulkan rendering context.
 *
 * Lifecycle:
 *   1. init(width, height, title) — Create instance, device, swapchain
 *   2. create_buffer() / load_shader() — Set up resources
 *   3. begin_frame() / end_frame() in render loop
 *   4. shutdown() — Destroy all resources
 */
class VulkanContext {
public:
    VulkanContext() = default;
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    /**
     * @brief Initialize Vulkan instance, device, and swapchain.
     * @param width Window width.
     * @param height Window height.
     * @param title Window title.
     * @param enable_validation Enable Vulkan validation layers.
     * @return true on success.
     */
    bool init(uint32_t width, uint32_t height,
              const std::string& title, bool enable_validation = true);

    /// Shut down and release all resources.
    void shutdown();

    /// Check if context is initialized.
    [[nodiscard]] bool is_initialized() const { return initialized_; }

    // ---- Resource management ----

    /// Create a GPU buffer for tensor storage.
    VulkanBuffer create_storage_buffer(size_t size_bytes);

    /// Upload data to a GPU buffer.
    void upload_buffer(const VulkanBuffer& buffer, const void* data, size_t size);

    /// Download data from a GPU buffer.
    void download_buffer(const VulkanBuffer& buffer, void* data, size_t size) const;

    /// Load a shader module from SPIR-V file.
    uint32_t load_shader(const ShaderModule& module);

    // ---- Render loop ----

    /// Begin a new frame. Returns false if window should close.
    bool begin_frame();

    /// Submit draw commands for the current frame.
    void end_frame();

    /// Get current framebuffer dimensions.
    [[nodiscard]] uint32_t width() const { return width_; }
    [[nodiscard]] uint32_t height() const { return height_; }

    // ---- Compute dispatch ----

    /// Dispatch a compute shader.
    void dispatch_compute(uint32_t shader_handle,
                         uint32_t group_x, uint32_t group_y, uint32_t group_z,
                         const std::vector<VulkanBuffer>& buffers);

    // ---- Screenshot ----

    /// Capture current framebuffer to file.
    bool screenshot(const std::string& path) const;

    // ---- Window events ----

    /// Poll window events. Returns false if close requested.
    bool poll_events();

    /// Check if a key is pressed.
    [[nodiscard]] bool key_pressed(int key) const;

    /// Get mouse position.
    [[nodiscard]] std::pair<float, float> mouse_position() const;

    /// Check if mouse button is pressed.
    [[nodiscard]] bool mouse_button(int button) const;

private:
    bool initialized_ = false;
    uint32_t width_ = 0, height_ = 0;

    // Internal Vulkan handles would be here in production.
    // For this implementation we store indices / mock handles.
    std::vector<VulkanBuffer> buffers_;
    std::vector<ShaderModule> shaders_;
};

} // namespace ciir_sim
