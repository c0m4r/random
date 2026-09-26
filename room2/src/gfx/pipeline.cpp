#include "gfx/pipeline.hpp"

#include <cstdlib>
#include <fstream>

#include "core/log.hpp"

#if defined(__linux__)
#include <unistd.h>
#endif

namespace room2::gfx {
namespace {

std::string executableDirectory() {
#if defined(__linux__)
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        std::string path(buf);
        size_t slash = path.find_last_of('/');
        return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
    }
#endif
    return ".";
}

std::vector<char> readFile(const std::string& path, bool& ok) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    ok = false;
    if (!file.is_open()) return {};
    size_t size = static_cast<size_t>(file.tellg());
    std::vector<char> data(size);
    file.seekg(0);
    file.read(data.data(), static_cast<std::streamsize>(size));
    ok = true;
    return data;
}

VkPipelineShaderStageCreateInfo stageInfo(VkShaderStageFlagBits stage, VkShaderModule module,
                                          const char* entry,
                                          const VkSpecializationInfo* spec) {
    VkPipelineShaderStageCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    info.stage = stage;
    info.module = module;
    info.pName = entry;
    info.pSpecializationInfo = spec;
    return info;
}

void fillStencil(VkStencilOpState& out, const StencilState& in, bool enabled) {
    out.failOp = enabled ? in.failOp : VK_STENCIL_OP_KEEP;
    out.passOp = enabled ? in.passOp : VK_STENCIL_OP_KEEP;
    out.depthFailOp = enabled ? in.depthFailOp : VK_STENCIL_OP_KEEP;
    out.compareOp = enabled ? in.compareOp : VK_COMPARE_OP_ALWAYS;
    out.compareMask = in.compareMask;
    out.writeMask = enabled ? in.writeMask : 0u;
    out.reference = in.reference;
}

}  // namespace

std::string resolveShaderPath(const std::string& relative) {
    if (const char* dir = std::getenv("ROOM2_SHADER_DIR")) {
        std::string candidate = std::string(dir) + "/" + relative;
        std::ifstream probe(candidate);
        if (probe.good()) return candidate;
    }
    const std::string base = executableDirectory();
    for (const std::string& prefix : {base + "/shaders/", base + "/../shaders/",
                                      base + "/../../shaders/", std::string("shaders/")}) {
        std::string candidate = prefix + relative;
        std::ifstream probe(candidate);
        if (probe.good()) return candidate;
    }
    return base + "/shaders/" + relative;
}

VkShaderModule loadShaderModule(Device& device, const std::string& spvPath) {
    bool ok = false;
    std::vector<char> code = readFile(spvPath, ok);
    if (!ok || code.empty()) {
        R2_FATAL("cannot read shader module: ", spvPath);
        return VK_NULL_HANDLE;
    }
    if (code.size() % 4 != 0) {
        R2_FATAL("shader module size is not a multiple of 4: ", spvPath);
        return VK_NULL_HANDLE;
    }
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = code.size();
    info.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule module = VK_NULL_HANDLE;
    VkResult r = vkCreateShaderModule(device.device(), &info, nullptr, &module);
    if (r != VK_SUCCESS) {
        R2_FATAL("vkCreateShaderModule failed for ", spvPath, ": ", vk::resultString(r));
        return VK_NULL_HANDLE;
    }
    return module;
}

// ---------------------------------------------------------------- pipeline cache
void PipelineCache::init(Device& device, const std::string& cacheFile) {
    device_ = &device;
    file_ = cacheFile;
    std::vector<char> initial;
    if (!cacheFile.empty()) {
        bool ok = false;
        initial = readFile(cacheFile, ok);
        if (!ok) initial.clear();
    }
    VkPipelineCacheCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    info.initialDataSize = initial.size();
    info.pInitialData = initial.empty() ? nullptr : initial.data();
    VkResult r = vkCreatePipelineCache(device.device(), &info, nullptr, &cache_);
    if (r != VK_SUCCESS) {
        R2_WARN("vkCreatePipelineCache failed (", vk::resultString(r), "); continuing without");
        cache_ = VK_NULL_HANDLE;
    }
}

void PipelineCache::shutdown() {
    if (cache_ && device_) vkDestroyPipelineCache(device_->device(), cache_, nullptr);
    cache_ = VK_NULL_HANDLE;
}

void PipelineCache::save() const {
    if (!cache_ || !device_ || file_.empty()) return;
    size_t size = 0;
    if (vkGetPipelineCacheData(device_->device(), cache_, &size, nullptr) != VK_SUCCESS || !size)
        return;
    std::vector<char> data(size);
    if (vkGetPipelineCacheData(device_->device(), cache_, &size, data.data()) != VK_SUCCESS) return;
    std::ofstream out(file_, std::ios::binary | std::ios::trunc);
    if (out.good()) out.write(data.data(), static_cast<std::streamsize>(size));
}

// ---------------------------------------------------------------- graphics pipeline
VkPipeline createGraphicsPipeline(Device& device, PipelineCache& cache,
                                  const GraphicsPipelineDesc& desc, const char* name) {
    VkSpecializationInfo specInfo{};
    const VkSpecializationInfo* specPtr = nullptr;
    if (!desc.specEntries.empty()) {
        specInfo.mapEntryCount = static_cast<uint32_t>(desc.specEntries.size());
        specInfo.pMapEntries = desc.specEntries.data();
        specInfo.dataSize = desc.specData.size() * sizeof(uint32_t);
        specInfo.pData = desc.specData.data();
        specPtr = &specInfo;
    }

    VkPipelineShaderStageCreateInfo stages[2] = {
        stageInfo(VK_SHADER_STAGE_VERTEX_BIT, desc.vertexShader, desc.vertexEntry, specPtr),
        stageInfo(VK_SHADER_STAGE_FRAGMENT_BIT, desc.fragmentShader, desc.fragmentEntry, specPtr),
    };
    uint32_t stageCount = desc.fragmentShader != VK_NULL_HANDLE ? 2u : 1u;

    VkPipelineVertexInputStateCreateInfo vertexInput{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertexInput.vertexBindingDescriptionCount = static_cast<uint32_t>(desc.vertexBindings.size());
    std::vector<VkVertexInputBindingDescription> bindings;
    bindings.reserve(desc.vertexBindings.size());
    for (const auto& b : desc.vertexBindings)
        bindings.push_back({b.binding, b.stride, b.inputRate});
    vertexInput.pVertexBindingDescriptions = bindings.empty() ? nullptr : bindings.data();
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(desc.vertexAttributes.size());
    std::vector<VkVertexInputAttributeDescription> attributes;
    attributes.reserve(desc.vertexAttributes.size());
    for (const auto& a : desc.vertexAttributes)
        attributes.push_back({a.location, a.binding, a.format, a.offset});
    vertexInput.pVertexAttributeDescriptions = attributes.empty() ? nullptr : attributes.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology = desc.topology;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewportState{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.depthClampEnable = desc.depthClampEnable ? VK_TRUE : VK_FALSE;
    raster.rasterizerDiscardEnable = desc.rasterizerDiscard ? VK_TRUE : VK_FALSE;
    raster.polygonMode = desc.polygonMode;
    raster.cullMode = desc.cullMode;
    raster.frontFace = desc.frontFace;
    raster.depthBiasEnable = desc.depthBiasEnable ? VK_TRUE : VK_FALSE;
    raster.depthBiasConstantFactor = desc.depthBiasConstant;
    raster.depthBiasSlopeFactor = desc.depthBiasSlope;
    raster.lineWidth = desc.lineWidth;

    VkPipelineMultisampleStateCreateInfo multisample{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = desc.samples;
    multisample.sampleShadingEnable = VK_FALSE;
    multisample.minSampleShading = 1.0f;
    multisample.alphaToOneEnable = VK_FALSE;

    VkPipelineDepthStencilStateCreateInfo depthStencil{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depthStencil.depthTestEnable = (desc.depthFormat != VK_FORMAT_UNDEFINED && desc.depthTestEnable)
                                       ? VK_TRUE
                                       : VK_FALSE;
    depthStencil.depthWriteEnable = (desc.depthFormat != VK_FORMAT_UNDEFINED &&
                                     desc.depthWriteEnable)
                                        ? VK_TRUE
                                        : VK_FALSE;
    depthStencil.depthCompareOp = desc.depthCompareOp;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = desc.stencilTestEnable ? VK_TRUE : VK_FALSE;
    fillStencil(depthStencil.front, desc.front, desc.stencilTestEnable);
    fillStencil(depthStencil.back, desc.back, desc.stencilTestEnable);

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.blendEnable = desc.blendEnable ? VK_TRUE : VK_FALSE;
    blendAttachment.srcColorBlendFactor = desc.srcColorBlendFactor;
    blendAttachment.dstColorBlendFactor = desc.dstColorBlendFactor;
    blendAttachment.colorBlendOp = desc.colorBlendOp;
    blendAttachment.srcAlphaBlendFactor = desc.srcAlphaBlendFactor;
    blendAttachment.dstAlphaBlendFactor = desc.dstAlphaBlendFactor;
    blendAttachment.alphaBlendOp = desc.alphaBlendOp;
    blendAttachment.colorWriteMask = desc.colorWriteMask;

    std::vector<VkPipelineColorBlendAttachmentState> blendAttachments(
        std::max<size_t>(1, desc.colorFormats.size()), blendAttachment);
    VkPipelineColorBlendStateCreateInfo colorBlend{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    colorBlend.logicOpEnable = VK_FALSE;
    colorBlend.attachmentCount = static_cast<uint32_t>(blendAttachments.size());
    colorBlend.pAttachments = blendAttachments.data();

    VkPipelineDynamicStateCreateInfo dynamic{
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = static_cast<uint32_t>(desc.dynamicStates.size());
    dynamic.pDynamicStates = desc.dynamicStates.empty() ? nullptr : desc.dynamicStates.data();

    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = static_cast<uint32_t>(desc.colorFormats.size());
    rendering.pColorAttachmentFormats = desc.colorFormats.empty() ? nullptr : desc.colorFormats.data();
    rendering.depthAttachmentFormat = desc.depthFormat;

    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.pNext = &rendering;
    info.flags = desc.flags;
    info.stageCount = stageCount;
    info.pStages = stages;
    info.pVertexInputState = &vertexInput;
    info.pInputAssemblyState = &inputAssembly;
    info.pViewportState = &viewportState;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pDepthStencilState = &depthStencil;
    info.pColorBlendState = &colorBlend;
    info.pDynamicState = &dynamic;
    info.layout = desc.layout;
    info.renderPass = VK_NULL_HANDLE;
    info.subpass = 0;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult r = vkCreateGraphicsPipelines(device.device(), cache.handle(), 1, &info, nullptr,
                                           &pipeline);
    if (r != VK_SUCCESS) {
        R2_FATAL("vkCreateGraphicsPipelines failed for '", name ? name : "<unnamed>",
                 "': ", vk::resultString(r));
        return VK_NULL_HANDLE;
    }
    if (name) VK_NAME(device.device(), pipeline, "%s", name);
    return pipeline;
}

VkPipeline createComputePipeline(Device& device, PipelineCache& cache,
                                 const ComputePipelineDesc& desc, const char* name) {
    VkSpecializationInfo specInfo{};
    const VkSpecializationInfo* specPtr = nullptr;
    if (!desc.specEntries.empty()) {
        specInfo.mapEntryCount = static_cast<uint32_t>(desc.specEntries.size());
        specInfo.pMapEntries = desc.specEntries.data();
        specInfo.dataSize = desc.specData.size() * sizeof(uint32_t);
        specInfo.pData = desc.specData.data();
        specPtr = &specInfo;
    }
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.flags = desc.flags;
    info.stage = stageInfo(VK_SHADER_STAGE_COMPUTE_BIT, desc.shader, desc.entry, specPtr);
    info.layout = desc.layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult r = vkCreateComputePipelines(device.device(), cache.handle(), 1, &info, nullptr,
                                          &pipeline);
    if (r != VK_SUCCESS) {
        R2_FATAL("vkCreateComputePipelines failed for '", name ? name : "<unnamed>",
                 "': ", vk::resultString(r));
        return VK_NULL_HANDLE;
    }
    if (name) VK_NAME(device.device(), pipeline, "%s", name);
    return pipeline;
}

}  // namespace room2::gfx
