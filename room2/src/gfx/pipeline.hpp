// room2 - shader modules, graphics/compute pipeline construction and a small
// shader hot-reload helper.
#pragma once

#include <ctime>
#include <string>
#include <vector>

#include "gfx/resources.hpp"

namespace room2::gfx {

// ---------------------------------------------------------------- shader modules
VkShaderModule loadShaderModule(Device& device, const std::string& spvPath);

// Resolves a shader path relative to the executable's directory (shaders/ next to
// the binary) and also honours the ROOM2_SHADER_DIR environment variable.
std::string resolveShaderPath(const std::string& relative);

// ---------------------------------------------------------------- pipeline cache
class PipelineCache {
public:
    void init(Device& device, const std::string& cacheFile = {});
    void shutdown();
    VkPipelineCache handle() const { return cache_; }
    // Persists the cache to disk (no-op when no cache file was given).
    void save() const;

private:
    Device* device_ = nullptr;
    VkPipelineCache cache_ = VK_NULL_HANDLE;
    std::string file_;
};

// ---------------------------------------------------------------- graphics pipeline
struct VertexBinding {
    uint32_t binding = 0;
    uint32_t stride = 0;
    VkVertexInputRate inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
};

struct VertexAttribute {
    uint32_t location = 0;
    uint32_t binding = 0;
    VkFormat format = VK_FORMAT_R32G32B32_SFLOAT;
    uint32_t offset = 0;
};

struct StencilState {
    bool enabled = false;
    VkStencilOp failOp = VK_STENCIL_OP_KEEP;
    VkStencilOp passOp = VK_STENCIL_OP_KEEP;
    VkStencilOp depthFailOp = VK_STENCIL_OP_KEEP;
    VkCompareOp compareOp = VK_COMPARE_OP_ALWAYS;
    uint32_t compareMask = 0xFF;
    uint32_t writeMask = 0xFF;
    uint32_t reference = 0;
};

struct GraphicsPipelineDesc {
    VkShaderModule vertexShader = VK_NULL_HANDLE;
    VkShaderModule fragmentShader = VK_NULL_HANDLE;
    const char* vertexEntry = "main";
    const char* fragmentEntry = "main";

    std::vector<VertexBinding> vertexBindings;
    std::vector<VertexAttribute> vertexAttributes;

    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPolygonMode polygonMode = VK_POLYGON_MODE_FILL;
    VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT;
    VkFrontFace frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    bool depthClampEnable = false;
    bool rasterizerDiscard = false;
    bool depthBiasEnable = false;
    float depthBiasConstant = 0.0f;
    float depthBiasSlope = 0.0f;
    float lineWidth = 1.0f;
    bool conservativeRaster = false;

    bool depthTestEnable = true;
    bool depthWriteEnable = true;
    VkCompareOp depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
    bool stencilTestEnable = false;
    StencilState front{};
    StencilState back{};

    bool blendEnable = false;
    VkBlendFactor srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    VkBlendFactor dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    VkBlendOp colorBlendOp = VK_BLEND_OP_ADD;
    VkBlendFactor srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    VkBlendFactor dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    VkBlendOp alphaBlendOp = VK_BLEND_OP_ADD;
    VkColorComponentFlags colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;

    // Dynamic rendering attachments. An undefined depth format disables the depth attachment.
    std::vector<VkFormat> colorFormats;
    VkFormat depthFormat = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineLayout layout = VK_NULL_HANDLE;
    std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                                 VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineCreateFlags flags = 0;
    uint32_t viewMask = 0;
    // Optional specialization constants.
    std::vector<VkSpecializationMapEntry> specEntries;
    std::vector<uint32_t> specData;
};

VkPipeline createGraphicsPipeline(Device& device, PipelineCache& cache,
                                  const GraphicsPipelineDesc& desc, const char* name = nullptr);

struct ComputePipelineDesc {
    VkShaderModule shader = VK_NULL_HANDLE;
    const char* entry = "main";
    VkPipelineLayout layout = VK_NULL_HANDLE;
    std::vector<VkSpecializationMapEntry> specEntries;
    std::vector<uint32_t> specData;
    VkPipelineCreateFlags flags = 0;
};

VkPipeline createComputePipeline(Device& device, PipelineCache& cache,
                                 const ComputePipelineDesc& desc, const char* name = nullptr);

}  // namespace room2::gfx
