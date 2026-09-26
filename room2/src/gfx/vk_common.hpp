// room2 - common Vulkan includes, error checking and small helpers.
#pragma once

#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>

#include "core/log.hpp"

namespace room2::vk {

// ---------------------------------------------------------------- error handling
const char* resultString(VkResult r);
// Returns true when the result indicates success (including suboptimal / not ready).
inline bool resultOk(VkResult r) {
    return r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR || r == VK_NOT_READY ||
           r == VK_TIMEOUT || r == VK_INCOMPLETE;
}

[[noreturn]] void fatalResult(VkResult r, const char* expr, const char* file, int line);

#define VK_CHECK(expr)                                                        \
    do {                                                                      \
        VkResult r2_ = (expr);                                                \
        if (r2_ != VK_SUCCESS)                                                \
            ::room2::vk::fatalResult(r2_, #expr, __FILE__, __LINE__);         \
    } while (0)

// Same as VK_CHECK but tolerates VK_SUBOPTIMAL_KHR / VK_NOT_READY.
#define VK_CHECK_SOFT(expr)                                                   \
    do {                                                                      \
        VkResult r2_ = (expr);                                                \
        if (!::room2::vk::resultOk(r2_))                                      \
            ::room2::vk::fatalResult(r2_, #expr, __FILE__, __LINE__);         \
    } while (0)

// ---------------------------------------------------------------- debug naming
bool debugNamingEnabled();
void setDebugNaming(bool enabled);
void setObjectNameRaw(VkDevice device, uint64_t handle, VkObjectType type, const char* name);

template <typename T>
struct ObjectTypeOf;
#define R2_OBJECT_TYPE(cppType, vkEnum)                    \
    template <>                                            \
    struct ObjectTypeOf<cppType> {                         \
        static constexpr VkObjectType value = vkEnum;      \
    }
R2_OBJECT_TYPE(VkBuffer, VK_OBJECT_TYPE_BUFFER);
R2_OBJECT_TYPE(VkImage, VK_OBJECT_TYPE_IMAGE);
R2_OBJECT_TYPE(VkImageView, VK_OBJECT_TYPE_IMAGE_VIEW);
R2_OBJECT_TYPE(VkPipeline, VK_OBJECT_TYPE_PIPELINE);
R2_OBJECT_TYPE(VkPipelineLayout, VK_OBJECT_TYPE_PIPELINE_LAYOUT);
R2_OBJECT_TYPE(VkPipelineCache, VK_OBJECT_TYPE_PIPELINE_CACHE);
R2_OBJECT_TYPE(VkSampler, VK_OBJECT_TYPE_SAMPLER);
R2_OBJECT_TYPE(VkDescriptorSetLayout, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT);
R2_OBJECT_TYPE(VkDescriptorPool, VK_OBJECT_TYPE_DESCRIPTOR_POOL);
R2_OBJECT_TYPE(VkDescriptorSet, VK_OBJECT_TYPE_DESCRIPTOR_SET);
R2_OBJECT_TYPE(VkShaderModule, VK_OBJECT_TYPE_SHADER_MODULE);
R2_OBJECT_TYPE(VkCommandBuffer, VK_OBJECT_TYPE_COMMAND_BUFFER);
R2_OBJECT_TYPE(VkCommandPool, VK_OBJECT_TYPE_COMMAND_POOL);
R2_OBJECT_TYPE(VkFramebuffer, VK_OBJECT_TYPE_FRAMEBUFFER);
R2_OBJECT_TYPE(VkRenderPass, VK_OBJECT_TYPE_RENDER_PASS);
R2_OBJECT_TYPE(VkSemaphore, VK_OBJECT_TYPE_SEMAPHORE);
R2_OBJECT_TYPE(VkFence, VK_OBJECT_TYPE_FENCE);
R2_OBJECT_TYPE(VkQueryPool, VK_OBJECT_TYPE_QUERY_POOL);
R2_OBJECT_TYPE(VkSwapchainKHR, VK_OBJECT_TYPE_SWAPCHAIN_KHR);
R2_OBJECT_TYPE(VkEvent, VK_OBJECT_TYPE_EVENT);
#undef R2_OBJECT_TYPE

template <typename T>
inline uint64_t handleToU64(T handle) {
    if constexpr (std::is_pointer_v<T>) {
        return reinterpret_cast<uint64_t>(handle);
    } else {
        return static_cast<uint64_t>(handle);
    }
}

template <typename T>
inline void setObjectName(VkDevice device, T handle, const char* name) {
    setObjectNameRaw(device, handleToU64(handle), ObjectTypeOf<T>::value, name);
}

#define VK_NAME(device, handle, ...)                                          \
    do {                                                                      \
        if (::room2::vk::debugNamingEnabled()) {                              \
            char buf2_[256];                                                  \
            std::snprintf(buf2_, sizeof(buf2_), __VA_ARGS__);                 \
            ::room2::vk::setObjectName(device, handle, buf2_);                \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------- format helpers
bool hasStencil(VkFormat f);
bool isDepthFormat(VkFormat f);
bool isDepthStencilFormat(VkFormat f);
uint32_t formatTexelSize(VkFormat f);
const char* formatName(VkFormat f);
// True when the format stores values in the sRGB transfer function.
bool isSrgbFormat(VkFormat f);

// ---------------------------------------------------------------- defaults
inline VkImageSubresourceRange colorRange(uint32_t mipLevels = 1, uint32_t layerCount = 1,
                                          uint32_t baseMip = 0, uint32_t baseLayer = 0) {
    return {VK_IMAGE_ASPECT_COLOR_BIT, baseMip, mipLevels, baseLayer, layerCount};
}
inline VkImageSubresourceRange depthRange(uint32_t mipLevels = 1, uint32_t layerCount = 1) {
    return {VK_IMAGE_ASPECT_DEPTH_BIT, 0, mipLevels, 0, layerCount};
}
inline VkImageSubresourceRange depthStencilRange() {
    return {VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1};
}
// Field order follows VkImageSubresourceLayers: aspect, mip, baseArrayLayer, layerCount.
inline VkImageSubresourceLayers colorLayers(uint32_t mipLevel = 0, uint32_t layerCount = 1,
                                            uint32_t baseArrayLayer = 0) {
    return {VK_IMAGE_ASPECT_COLOR_BIT, mipLevel, baseArrayLayer, layerCount};
}
inline VkExtent3D extent3D(uint32_t w, uint32_t h, uint32_t d = 1) { return {w, h, d}; }

}  // namespace room2::vk
