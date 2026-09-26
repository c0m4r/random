// room2 - GPU resource helpers: buffers, images, samplers, barriers, uploads and
// descriptor set plumbing. Deliberately thin: one VkDeviceMemory per resource,
// which is fine at this scale (a few hundred long-lived resources) and keeps the
// lifetime rules obvious.
#pragma once

#include <cstring>
#include <string>
#include <vector>

#include "gfx/device.hpp"

namespace room2::gfx {

// ---------------------------------------------------------------- buffer
struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    VkDeviceSize size = 0;
    void* mapped = nullptr;   // non-null when the memory is persistently mapped

    explicit operator bool() const { return handle != VK_NULL_HANDLE; }
    template <typename T>
    T* as() const {
        return static_cast<T*>(mapped);
    }
    void write(const void* data, size_t bytes, size_t offset = 0) const {
        if (mapped && bytes) std::memcpy(static_cast<char*>(mapped) + offset, data, bytes);
    }
};

struct BufferDesc {
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkMemoryPropertyFlags memory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const char* name = nullptr;
};

Buffer createBuffer(Device& device, const BufferDesc& desc);
void destroyBuffer(Device& device, Buffer& buffer);

// ---------------------------------------------------------------- image
struct Image {
    VkImage handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent3D extent{0, 0, 1};
    uint32_t mipLevels = 1;
    uint32_t arrayLayers = 1;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;

    explicit operator bool() const { return handle != VK_NULL_HANDLE; }
    VkExtent2D extent2D() const { return {extent.width, extent.height}; }
    float aspectRatio() const {
        return extent.height ? static_cast<float>(extent.width) / static_cast<float>(extent.height)
                             : 1.0f;
    }
    VkDescriptorImageInfo descriptorInfo(VkSampler sampler) const {
        return {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    }
};

struct ImageDesc {
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    VkExtent3D extent{1, 1, 1};
    uint32_t mipLevels = 1;
    uint32_t arrayLayers = 1;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    VkImageType type = VK_IMAGE_TYPE_2D;
    VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D;
    VkImageCreateFlags flags = 0;
    VkComponentMapping swizzle{};
    const char* name = nullptr;
};

Image createImage(Device& device, const ImageDesc& desc);
void destroyImage(Device& device, Image& image);
// Creates only the view (used for swapchain images we do not own).
VkImageView createImageView(Device& device, VkImage image, VkFormat format,
                            VkImageAspectFlags aspect, VkImageViewType type, uint32_t mipLevels,
                            uint32_t arrayLayers, const char* name = nullptr);

// ---------------------------------------------------------------- sampler
VkSampler createSampler(Device& device, VkFilter minFilter, VkFilter magFilter,
                        VkSamplerMipmapMode mipMode, VkSamplerAddressMode address,
                        float maxAnisotropy = 1.0f, bool compareEnable = false,
                        VkCompareOp compareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
                        float minLod = 0.0f, float maxLod = VK_LOD_CLAMP_NONE,
                        bool reduction = false, VkSamplerReductionMode reductionMode =
                                                       VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE,
                        const char* name = nullptr);

// ---------------------------------------------------------------- barriers
void cmdImageBarrier(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout,
                     VkImageLayout newLayout, VkPipelineStageFlags2 srcStage,
                     VkPipelineStageFlags2 dstStage, VkAccessFlags2 srcAccess,
                     VkAccessFlags2 dstAccess, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                     uint32_t baseMip = 0, uint32_t mipCount = 1, uint32_t baseLayer = 0,
                     uint32_t layerCount = 1);
void cmdMemoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags2 srcStage,
                      VkPipelineStageFlags2 dstStage, VkAccessFlags2 srcAccess,
                      VkAccessFlags2 dstAccess);

// ---------------------------------------------------------------- uploads
// Batches staging copies into a single one-shot submission. Used at load time.
class UploadBatch {
public:
    explicit UploadBatch(Device& device);
    ~UploadBatch();

    UploadBatch(const UploadBatch&) = delete;
    UploadBatch& operator=(const UploadBatch&) = delete;

    void buffer(const Buffer& dst, const void* data, size_t bytes, size_t dstOffset = 0);
    // Uploads tightly packed texel data for one mip level / array layer.
    void image(const Image& dst, const void* data, size_t bytes, uint32_t mip = 0,
               uint32_t layer = 0);
    // Submits everything recorded so far and blocks until finished.
    void submit(const char* name = "upload");
    size_t pendingCount() const { return pendingBuffers_.size() + pendingImages_.size(); }

private:
    Device& device_;
    struct PendingBuffer {
        VkBuffer staging;
        VkDeviceMemory stagingMemory;
        VkBuffer dst;
        VkDeviceSize dstOffset;
        VkDeviceSize size;
    };
    struct PendingImage {
        VkBuffer staging;
        VkDeviceMemory stagingMemory;
        VkImage dst;
        uint32_t mip;
        uint32_t layer;
        VkDeviceSize size;
        VkExtent3D extent;
    };
    std::vector<PendingBuffer> pendingBuffers_;
    std::vector<PendingImage> pendingImages_;

    VkBuffer makeStaging(VkDeviceSize size, VkDeviceMemory& memory);
    void release(PendingBuffer& p);
    void release(PendingImage& p);
};

// ---------------------------------------------------------------- descriptors
struct DescriptorBinding {
    uint32_t binding = 0;
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uint32_t count = 1;
    VkShaderStageFlags stages = VK_SHADER_STAGE_ALL;
    const VkSampler* immutableSamplers = nullptr;
    VkDescriptorBindingFlags bindingFlags = 0;
};

VkDescriptorSetLayout createDescriptorSetLayout(Device& device,
                                                const std::vector<DescriptorBinding>& bindings,
                                                const char* name = nullptr);

class DescriptorPool {
public:
    void init(Device& device, const std::vector<VkDescriptorPoolSize>& sizes,
              uint32_t maxSets, bool updateAfterBind = false, const char* name = nullptr);
    void shutdown();
    VkDescriptorSet allocate(VkDescriptorSetLayout layout, const char* name = nullptr);
    void reset();
    VkDescriptorPool handle() const { return pool_; }

private:
    Device* device_ = nullptr;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
};

// Convenience write helpers (chainable through VkWriteDescriptorSet).
VkWriteDescriptorSet writeBuffer(VkWriteDescriptorSet& w, VkDescriptorSet set, uint32_t binding,
                                 VkDescriptorType type, const VkDescriptorBufferInfo* info,
                                 uint32_t count = 1, uint32_t arrayElement = 0);
VkWriteDescriptorSet writeImage(VkWriteDescriptorSet& w, VkDescriptorSet set, uint32_t binding,
                                VkDescriptorType type, const VkDescriptorImageInfo* info,
                                uint32_t count = 1, uint32_t arrayElement = 0);

// ---------------------------------------------------------------- misc
// Computes the number of mip levels for a full mip chain.
uint32_t mipCountFor(VkExtent3D extent);
VkDeviceSize alignedSize(VkDeviceSize value, VkDeviceSize alignment);

}  // namespace room2::gfx
