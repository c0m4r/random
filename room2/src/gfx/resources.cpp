#include "gfx/resources.hpp"

#include <algorithm>

#include "core/log.hpp"

namespace room2::gfx {
namespace {

VkImageAspectFlags aspectForFormat(VkFormat format) {
    if (vk::isDepthStencilFormat(format)) {
        if (vk::hasStencil(format)) return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        return VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    return VK_IMAGE_ASPECT_COLOR_BIT;
}

}  // namespace

// ---------------------------------------------------------------- buffer
Buffer createBuffer(Device& device, const BufferDesc& desc) {
    Buffer out;
    out.size = desc.size;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = desc.size;
    info.usage = desc.usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(device.device(), &info, nullptr, &out.handle));
    if (desc.name) VK_NAME(device.device(), out.handle, "%s", desc.name);

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device.device(), out.handle, &req);

    const bool wantHostVisible = (desc.memory & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex =
        device.findMemoryType(req.memoryTypeBits, desc.memory,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateFlagsInfo flagsInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    if (desc.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        alloc.pNext = &flagsInfo;
    }
    VK_CHECK(vkAllocateMemory(device.device(), &alloc, nullptr, &out.memory));
    VK_CHECK(vkBindBufferMemory(device.device(), out.handle, out.memory, 0));

    if (desc.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        VkBufferDeviceAddressInfo addrInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        addrInfo.buffer = out.handle;
        out.address = vkGetBufferDeviceAddress(device.device(), &addrInfo);
    }
    if (wantHostVisible) {
        VK_CHECK(vkMapMemory(device.device(), out.memory, 0, VK_WHOLE_SIZE, 0, &out.mapped));
    }
    return out;
}

void destroyBuffer(Device& device, Buffer& buffer) {
    if (buffer.mapped) {
        vkUnmapMemory(device.device(), buffer.memory);
        buffer.mapped = nullptr;
    }
    if (buffer.handle) vkDestroyBuffer(device.device(), buffer.handle, nullptr);
    if (buffer.memory) vkFreeMemory(device.device(), buffer.memory, nullptr);
    buffer = Buffer{};
}

// ---------------------------------------------------------------- image
Image createImage(Device& device, const ImageDesc& desc) {
    Image out;
    out.format = desc.format;
    out.extent = desc.extent;
    out.mipLevels = desc.mipLevels;
    out.arrayLayers = desc.arrayLayers;
    out.samples = desc.samples;
    out.aspect = aspectForFormat(desc.format);

    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.flags = desc.flags;
    info.imageType = desc.type;
    info.format = desc.format;
    info.extent = desc.extent;
    info.mipLevels = desc.mipLevels;
    info.arrayLayers = desc.arrayLayers;
    info.samples = desc.samples;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = desc.usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(device.device(), &info, nullptr, &out.handle));
    if (desc.name) VK_NAME(device.device(), out.handle, "%s", desc.name);

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device.device(), out.handle, &req);
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = device.findMemoryType(
        req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device.device(), &alloc, nullptr, &out.memory));
    VK_CHECK(vkBindImageMemory(device.device(), out.handle, out.memory, 0));

    out.view = createImageView(device, out.handle, out.format, out.aspect, desc.viewType,
                               out.mipLevels, out.arrayLayers, desc.name);
    if (desc.swizzle.r != VK_COMPONENT_SWIZZLE_IDENTITY ||
        desc.swizzle.g != VK_COMPONENT_SWIZZLE_IDENTITY ||
        desc.swizzle.b != VK_COMPONENT_SWIZZLE_IDENTITY ||
        desc.swizzle.a != VK_COMPONENT_SWIZZLE_IDENTITY) {
        vkDestroyImageView(device.device(), out.view, nullptr);
        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = out.handle;
        viewInfo.viewType = desc.viewType;
        viewInfo.format = desc.format;
        viewInfo.components = desc.swizzle;
        viewInfo.subresourceRange = {out.aspect, 0, out.mipLevels, 0, out.arrayLayers};
        VK_CHECK(vkCreateImageView(device.device(), &viewInfo, nullptr, &out.view));
    }
    return out;
}

VkImageView createImageView(Device& device, VkImage image, VkFormat format,
                            VkImageAspectFlags aspect, VkImageViewType type, uint32_t mipLevels,
                            uint32_t arrayLayers, const char* name) {
    VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    info.image = image;
    info.viewType = type;
    info.format = format;
    info.subresourceRange = {aspect, 0, mipLevels, 0, arrayLayers};
    VkImageView view = VK_NULL_HANDLE;
    VK_CHECK(vkCreateImageView(device.device(), &info, nullptr, &view));
    if (name) VK_NAME(device.device(), view, "%s", name);
    return view;
}

void destroyImage(Device& device, Image& image) {
    if (image.view) vkDestroyImageView(device.device(), image.view, nullptr);
    if (image.handle) vkDestroyImage(device.device(), image.handle, nullptr);
    if (image.memory) vkFreeMemory(device.device(), image.memory, nullptr);
    image = Image{};
}

// ---------------------------------------------------------------- sampler
VkSampler createSampler(Device& device, VkFilter minFilter, VkFilter magFilter,
                        VkSamplerMipmapMode mipMode, VkSamplerAddressMode address,
                        float maxAnisotropy, bool compareEnable, VkCompareOp compareOp,
                        float minLod, float maxLod, bool reduction,
                        VkSamplerReductionMode reductionMode, const char* name) {
    VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    info.magFilter = magFilter;
    info.minFilter = minFilter;
    info.mipmapMode = mipMode;
    info.addressModeU = address;
    info.addressModeV = address;
    info.addressModeW = address;
    info.mipLodBias = 0.0f;
    info.anisotropyEnable =
        (maxAnisotropy > 1.0f && device.features().samplerAnisotropy) ? VK_TRUE : VK_FALSE;
    info.maxAnisotropy = info.anisotropyEnable ? std::min(maxAnisotropy,
                                                          device.features().maxSamplerAnisotropy)
                                               : 1.0f;
    info.compareEnable = compareEnable ? VK_TRUE : VK_FALSE;
    info.compareOp = compareOp;
    info.minLod = minLod;
    info.maxLod = maxLod;
    info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    info.unnormalizedCoordinates = VK_FALSE;

    VkSamplerReductionModeCreateInfo reductionInfo{
        VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO};
    if (reduction) {
        reductionInfo.reductionMode = reductionMode;
        info.pNext = &reductionInfo;
    }
    VkSampler sampler = VK_NULL_HANDLE;
    VK_CHECK(vkCreateSampler(device.device(), &info, nullptr, &sampler));
    if (name) VK_NAME(device.device(), sampler, "%s", name);
    return sampler;
}

// ---------------------------------------------------------------- barriers
void cmdImageBarrier(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout,
                     VkImageLayout newLayout, VkPipelineStageFlags2 srcStage,
                     VkPipelineStageFlags2 dstStage, VkAccessFlags2 srcAccess,
                     VkAccessFlags2 dstAccess, VkImageAspectFlags aspect, uint32_t baseMip,
                     uint32_t mipCount, uint32_t baseLayer, uint32_t layerCount) {
    VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    barrier.srcStageMask = srcStage;
    barrier.dstStageMask = dstStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {aspect, baseMip, mipCount, baseLayer, layerCount};
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}

void cmdMemoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags2 srcStage,
                      VkPipelineStageFlags2 dstStage, VkAccessFlags2 srcAccess,
                      VkAccessFlags2 dstAccess) {
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = srcStage;
    barrier.dstStageMask = dstStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}

// ---------------------------------------------------------------- uploads
UploadBatch::UploadBatch(Device& device) : device_(device) {}

UploadBatch::~UploadBatch() {
    for (auto& p : pendingBuffers_) release(p);
    for (auto& p : pendingImages_) release(p);
}

VkBuffer UploadBatch::makeStaging(VkDeviceSize size, VkDeviceMemory& memory) {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VK_CHECK(vkCreateBuffer(device_.device(), &info, nullptr, &buffer));

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device_.device(), buffer, &req);
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = device_.findMemoryType(
        req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(device_.device(), &alloc, nullptr, &memory));
    VK_CHECK(vkBindBufferMemory(device_.device(), buffer, memory, 0));
    return buffer;
}

void UploadBatch::release(PendingBuffer& p) {
    vkDestroyBuffer(device_.device(), p.staging, nullptr);
    vkFreeMemory(device_.device(), p.stagingMemory, nullptr);
    p.staging = VK_NULL_HANDLE;
}

void UploadBatch::release(PendingImage& p) {
    vkDestroyBuffer(device_.device(), p.staging, nullptr);
    vkFreeMemory(device_.device(), p.stagingMemory, nullptr);
    p.staging = VK_NULL_HANDLE;
}

void UploadBatch::buffer(const Buffer& dst, const void* data, size_t bytes, size_t dstOffset) {
    if (bytes == 0) return;
    PendingBuffer p{};
    p.size = bytes;
    p.dst = dst.handle;
    p.dstOffset = dstOffset;
    p.staging = makeStaging(bytes, p.stagingMemory);
    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(device_.device(), p.stagingMemory, 0, bytes, 0, &mapped));
    std::memcpy(mapped, data, bytes);
    vkUnmapMemory(device_.device(), p.stagingMemory);
    pendingBuffers_.push_back(p);
}

void UploadBatch::image(const Image& dst, const void* data, size_t bytes, uint32_t mip,
                        uint32_t layer) {
    if (bytes == 0) return;
    PendingImage p{};
    p.size = bytes;
    p.dst = dst.handle;
    p.mip = mip;
    p.layer = layer;
    p.extent = {std::max(1u, dst.extent.width >> mip), std::max(1u, dst.extent.height >> mip),
                std::max(1u, dst.extent.depth >> mip)};
    p.staging = makeStaging(bytes, p.stagingMemory);
    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(device_.device(), p.stagingMemory, 0, bytes, 0, &mapped));
    std::memcpy(mapped, data, bytes);
    vkUnmapMemory(device_.device(), p.stagingMemory);
    pendingImages_.push_back(p);
}

void UploadBatch::submit(const char* name) {
    if (pendingBuffers_.empty() && pendingImages_.empty()) return;
    VkCommandBuffer cmd = device_.beginImmediate(name ? name : "upload");
    device_.beginLabel(cmd, name ? name : "upload", 0.9f, 0.7f, 0.3f);

    for (auto& p : pendingImages_) {
        cmdImageBarrier(cmd, p.dst, VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                        VK_ACCESS_2_NONE, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT, p.mip, 1, p.layer, 1);
    }
    if (!pendingImages_.empty()) {
        cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    }
    for (auto& p : pendingBuffers_) {
        VkBufferCopy region{};
        region.srcOffset = 0;
        region.dstOffset = p.dstOffset;
        region.size = p.size;
        vkCmdCopyBuffer(cmd, p.staging, p.dst, 1, &region);
    }
    for (auto& p : pendingImages_) {
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, p.mip, p.layer, 1};
        region.imageOffset = {0, 0, 0};
        region.imageExtent = p.extent;
        vkCmdCopyBufferToImage(cmd, p.staging, p.dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                               &region);
    }
    if (!pendingImages_.empty()) {
        cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT,
                         VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        for (auto& p : pendingImages_) {
            cmdImageBarrier(cmd, p.dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_PIPELINE_STAGE_2_COPY_BIT,
                            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                            VK_IMAGE_ASPECT_COLOR_BIT, p.mip, 1, p.layer, 1);
        }
    }
    device_.endLabel(cmd);
    device_.endImmediate(cmd, true);

    for (auto& p : pendingBuffers_) release(p);
    for (auto& p : pendingImages_) release(p);
    pendingBuffers_.clear();
    pendingImages_.clear();
}

// ---------------------------------------------------------------- descriptors
VkDescriptorSetLayout createDescriptorSetLayout(Device& device,
                                                const std::vector<DescriptorBinding>& bindings,
                                                const char* name) {
    std::vector<VkDescriptorSetLayoutBinding> vkBindings(bindings.size());
    std::vector<VkDescriptorBindingFlags> flags(bindings.size());
    for (size_t i = 0; i < bindings.size(); ++i) {
        vkBindings[i].binding = bindings[i].binding;
        vkBindings[i].descriptorType = bindings[i].type;
        vkBindings[i].descriptorCount = bindings[i].count;
        vkBindings[i].stageFlags = bindings[i].stages;
        vkBindings[i].pImmutableSamplers = bindings[i].immutableSamplers;
        flags[i] = bindings[i].bindingFlags;
    }
    VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    bool anyFlags = false;
    for (auto f : flags)
        if (f) anyFlags = true;
    if (anyFlags) {
        flagsInfo.bindingCount = static_cast<uint32_t>(flags.size());
        flagsInfo.pBindingFlags = flags.data();
    }
    VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    info.pNext = anyFlags ? &flagsInfo : nullptr;
    // The spec requires the UPDATE_AFTER_BIND_POOL flag whenever any binding is
    // update-after-bind; the set must then come from a pool created with the same flag.
    for (auto f : flags) {
        if (f & (VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                 VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT)) {
            info.flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
            break;
        }
    }
    info.bindingCount = static_cast<uint32_t>(vkBindings.size());
    info.pBindings = vkBindings.empty() ? nullptr : vkBindings.data();
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorSetLayout(device.device(), &info, nullptr, &layout));
    if (name) VK_NAME(device.device(), layout, "%s", name);
    return layout;
}

void DescriptorPool::init(Device& device, const std::vector<VkDescriptorPoolSize>& sizes,
                          uint32_t maxSets, bool updateAfterBind, const char* name) {
    shutdown();
    device_ = &device;
    VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    if (updateAfterBind) info.flags |= VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    info.maxSets = maxSets;
    info.poolSizeCount = static_cast<uint32_t>(sizes.size());
    info.pPoolSizes = sizes.empty() ? nullptr : sizes.data();
    VK_CHECK(vkCreateDescriptorPool(device.device(), &info, nullptr, &pool_));
    if (name) VK_NAME(device.device(), pool_, "%s", name);
}

void DescriptorPool::shutdown() {
    if (pool_ && device_) vkDestroyDescriptorPool(device_->device(), pool_, nullptr);
    pool_ = VK_NULL_HANDLE;
}

VkDescriptorSet DescriptorPool::allocate(VkDescriptorSetLayout layout, const char* name) {
    VkDescriptorSetAllocateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    info.descriptorPool = pool_;
    info.descriptorSetCount = 1;
    info.pSetLayouts = &layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateDescriptorSets(device_->device(), &info, &set));
    if (name) VK_NAME(device_->device(), set, "%s", name);
    return set;
}

void DescriptorPool::reset() {
    if (pool_ && device_) vkResetDescriptorPool(device_->device(), pool_, 0);
}

VkWriteDescriptorSet writeBuffer(VkWriteDescriptorSet& w, VkDescriptorSet set, uint32_t binding,
                                 VkDescriptorType type, const VkDescriptorBufferInfo* info,
                                 uint32_t count, uint32_t arrayElement) {
    w = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = set;
    w.dstBinding = binding;
    w.dstArrayElement = arrayElement;
    w.descriptorCount = count;
    w.descriptorType = type;
    w.pBufferInfo = info;
    return w;
}

VkWriteDescriptorSet writeImage(VkWriteDescriptorSet& w, VkDescriptorSet set, uint32_t binding,
                                VkDescriptorType type, const VkDescriptorImageInfo* info,
                                uint32_t count, uint32_t arrayElement) {
    w = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = set;
    w.dstBinding = binding;
    w.dstArrayElement = arrayElement;
    w.descriptorCount = count;
    w.descriptorType = type;
    w.pImageInfo = info;
    return w;
}

uint32_t mipCountFor(VkExtent3D extent) {
    uint32_t dim = std::max({extent.width, extent.height, extent.depth});
    uint32_t levels = 1;
    while (dim > 1) {
        dim >>= 1;
        ++levels;
    }
    return levels;
}

VkDeviceSize alignedSize(VkDeviceSize value, VkDeviceSize alignment) {
    if (alignment == 0) return value;
    return (value + alignment - 1) / alignment * alignment;
}

}  // namespace room2::gfx
