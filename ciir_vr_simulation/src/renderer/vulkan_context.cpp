/**
 * @file vulkan_context.cpp
 * @brief Stub Vulkan context (functional without actual Vulkan SDK).
 */

#include "renderer/vulkan_context.hpp"

namespace ciir_sim {

VulkanContext::~VulkanContext() {
    shutdown();
}

bool VulkanContext::init(uint32_t width, uint32_t height,
                          const std::string& /*title*/, bool /*enable_validation*/) {
    width_ = width;
    height_ = height;
    initialized_ = true;
    return true;
}

void VulkanContext::shutdown() {
    buffers_.clear();
    shaders_.clear();
    initialized_ = false;
    width_ = 0;
    height_ = 0;
}

VulkanBuffer VulkanContext::create_storage_buffer(size_t size_bytes) {
    VulkanBuffer buf;
    buf.handle = static_cast<uint32_t>(buffers_.size()) + 1;
    buf.size = size_bytes;
    buf.device_local = true;
    buffers_.push_back(buf);
    return buf;
}

void VulkanContext::upload_buffer(const VulkanBuffer& /*buffer*/,
                                   const void* /*data*/, size_t /*size*/) {
    // Stub: no-op without real Vulkan
}

void VulkanContext::download_buffer(const VulkanBuffer& /*buffer*/,
                                      void* /*data*/, size_t /*size*/) const {
    // Stub: no-op without real Vulkan
}

uint32_t VulkanContext::load_shader(const ShaderModule& module) {
    shaders_.push_back(module);
    return static_cast<uint32_t>(shaders_.size());
}

bool VulkanContext::begin_frame() {
    return initialized_;
}

void VulkanContext::end_frame() {
    // Stub: no-op
}

void VulkanContext::dispatch_compute(uint32_t /*shader_handle*/,
                                      uint32_t /*group_x*/, uint32_t /*group_y*/,
                                      uint32_t /*group_z*/,
                                      const std::vector<VulkanBuffer>& /*buffers*/) {
    // Stub: no-op
}

bool VulkanContext::screenshot(const std::string& /*path*/) const {
    return false;
}

bool VulkanContext::poll_events() {
    return initialized_;
}

bool VulkanContext::key_pressed(int /*key*/) const {
    return false;
}

std::pair<float, float> VulkanContext::mouse_position() const {
    return {0.0f, 0.0f};
}

bool VulkanContext::mouse_button(int /*button*/) const {
    return false;
}

} // namespace ciir_sim
