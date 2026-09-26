#include "render/renderer.hpp"

#include <algorithm>
#include <chrono>
#include <deque>
#include <cstring>

#include "core/log.hpp"
#include "gfx/vk_common.hpp"

namespace room2::render {
namespace {

constexpr VkFormat kGbufferAlbedoFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat kGbufferNormalFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat kGbufferMetalFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat kGbufferEmissiveFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
constexpr VkFormat kHdrFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat kAoFormat = VK_FORMAT_R16_SFLOAT;
constexpr VkFormat kShadowFormat = VK_FORMAT_D32_SFLOAT;

constexpr uint32_t kFrameCountMax = 8;

struct ShadowPush {
    uint32_t faceIndex;
    uint32_t lightIndex;
};

struct DrawPush {
    uint32_t materialIndex;
    uint32_t flags;
};

struct BloomPush {
    uint32_t sourceMip;   // unused by the shaders but handy for debugging
    uint32_t pad[3];
};

struct ProbePush {
    Vec4 roomMin;
    Vec4 roomMax;
    Vec4 lightPosIntensity;
    Vec4 lightColorRoughness;
    Vec4 bounce;
    uint32_t mipLevel;
    uint32_t faceSize;
    uint32_t sampleCount;
    uint32_t faceIndex;
};

struct UiPush {
    Vec2 screenSize;
    float useTexture;
    float opacity;
};

// Cube face directions and up vectors, matching the Vulkan cube-map face convention
// (so the shadow map can be addressed with the same maths as a samplerCube).
struct CubeFace {
    Vec3 direction;
    Vec3 up;
};
const CubeFace kCubeFaces[6] = {
    {{1, 0, 0}, {0, -1, 0}},   // +X
    {{-1, 0, 0}, {0, -1, 0}},  // -X
    {{0, 1, 0}, {0, 0, 1}},    // +Y
    {{0, -1, 0}, {0, 0, -1}},  // -Y
    {{0, 0, 1}, {0, -1, 0}},   // +Z
    {{0, 0, -1}, {0, -1, 0}},  // -Z
};

VkRenderingAttachmentInfo colorAttachment(VkImageView view, VkImageLayout layout,
                                          VkAttachmentLoadOp loadOp,
                                          VkAttachmentStoreOp storeOp, VkClearColorValue clear) {
    VkRenderingAttachmentInfo info{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    info.imageView = view;
    info.imageLayout = layout;
    info.loadOp = loadOp;
    info.storeOp = storeOp;
    info.clearValue.color = clear;
    return info;
}

VkRenderingAttachmentInfo depthAttachment(VkImageView view, VkImageLayout layout,
                                          VkAttachmentLoadOp loadOp,
                                          VkAttachmentStoreOp storeOp, float clear = 0.0f) {
    VkRenderingAttachmentInfo info{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    info.imageView = view;
    info.imageLayout = layout;
    info.loadOp = loadOp;
    info.storeOp = storeOp;
    info.clearValue.depthStencil = {clear, 0};
    return info;
}

uint32_t mipSize(uint32_t base, uint32_t level) {
    return std::max(1u, base >> level);
}

// IEEE 754 binary16 -> binary32.
float halfToFloat(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1Fu;
    uint32_t mantissa = h & 0x03FFu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            // Subnormal: normalise.
            exponent = 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x03FFu;
            bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

}  // namespace

Vec2 haltonJitter(uint32_t index, uint32_t width, uint32_t height) {
    auto halton = [](uint32_t i, uint32_t base) {
        float f = 1.0f, r = 0.0f;
        while (i > 0) {
            f /= static_cast<float>(base);
            r += f * static_cast<float>(i % base);
            i /= base;
        }
        return r;
    };
    // 8-tap Halton(2,3) sequence; centred on the pixel.
    uint32_t i = (index % 8) + 1;
    float x = halton(i, 2) - 0.5f;
    float y = halton(i, 3) - 0.5f;
    (void)width;
    (void)height;
    return {x, y};
}

Renderer::~Renderer() { shutdown(); }

// ---------------------------------------------------------------- init
bool Renderer::init(gfx::Device& device, gfx::Swapchain* swapchain, const RendererConfig& config) {
    device_ = &device;
    swapchain_ = swapchain;
    config_ = config;
    headless_ = config.headless || swapchain == nullptr;
    config_.maxFramesInFlight = std::clamp(config_.maxFramesInFlight, 1, static_cast<int>(kFrameCountMax));

    if (!headless_) {
        outputWidth_ = swapchain_->extent().width;
        outputHeight_ = swapchain_->extent().height;
    } else {
        outputWidth_ = config_.width;
        outputHeight_ = config_.height;
    }
    renderWidth_ = std::max(1u, static_cast<uint32_t>(outputWidth_ * config_.renderScale));
    renderHeight_ = std::max(1u, static_cast<uint32_t>(outputHeight_ * config_.renderScale));

    pipelineCache_.init(device, "room2_pipeline_cache.bin");

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = device.graphicsQueue().family;
    VK_CHECK(vkCreateCommandPool(device.device(), &poolInfo, nullptr, &commandPool_));
    VK_NAME(device.device(), commandPool_, "renderer command pool");

    commandBuffers_.resize(config_.maxFramesInFlight);
    VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = commandPool_;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = static_cast<uint32_t>(commandBuffers_.size());
    VK_CHECK(vkAllocateCommandBuffers(device.device(), &alloc, commandBuffers_.data()));

    frameFences_.resize(config_.maxFramesInFlight);
    imageAvailable_.resize(config_.maxFramesInFlight);
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VkSemaphoreCreateInfo semInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (int i = 0; i < config_.maxFramesInFlight; ++i) {
        VK_CHECK(vkCreateFence(device.device(), &fenceInfo, nullptr, &frameFences_[i]));
        VK_CHECK(vkCreateSemaphore(device.device(), &semInfo, nullptr, &imageAvailable_[i]));
    }

    samplerLinearRepeat_ = gfx::createSampler(device, VK_FILTER_LINEAR, VK_FILTER_LINEAR,
                                              VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                              VK_SAMPLER_ADDRESS_MODE_REPEAT, 8.0f, false,
                                              VK_COMPARE_OP_LESS_OR_EQUAL, 0.0f,
                                              VK_LOD_CLAMP_NONE, false,
                                              VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE,
                                              "linear repeat");
    samplerLinearClamp_ = gfx::createSampler(device, VK_FILTER_LINEAR, VK_FILTER_LINEAR,
                                             VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                             VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 4.0f, false,
                                             VK_COMPARE_OP_LESS_OR_EQUAL, 0.0f, VK_LOD_CLAMP_NONE,
                                             false, VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE,
                                             "linear clamp");
    samplerNearestClamp_ = gfx::createSampler(device, VK_FILTER_NEAREST, VK_FILTER_NEAREST,
                                              VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                              VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 1.0f, false,
                                              VK_COMPARE_OP_LESS_OR_EQUAL, 0.0f, VK_LOD_CLAMP_NONE,
                                              false, VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE,
                                              "nearest clamp");
    // A comparison sampler: the shader passes the receiver depth as the coordinate's
    // 4th component and the hardware returns 1.0 when the receiver is lit. This also
    // selects the cube face per tap, which removes the face-boundary artefacts that a
    // hand-rolled layer lookup produced at cube edges.
    //
    // The shadow map stores the NEAREST caster depth (0 = at the light, 1 = far plane),
    // so a receiver is lit when `stored >= reference`. Vulkan evaluates the comparison
    // in the opposite operand order to OpenGL: the predicate is compareOp(texel, ref),
    // which makes VK_COMPARE_OP_GREATER_OR_EQUAL the one that means "lit".
    // (Verified empirically: with LESS_OR_EQUAL the shadow term averaged 0.22 and the
    // room was almost entirely self-shadowed; with GREATER_OR_EQUAL it averages 0.78
    // with lit surfaces at 1.0.)
    samplerShadow_ = gfx::createSampler(device, VK_FILTER_LINEAR, VK_FILTER_LINEAR,
                                        VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 1.0f, true,
                                        VK_COMPARE_OP_GREATER_OR_EQUAL, 0.0f, 0.0f, false,
                                        VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE, "shadow cube");
    samplerEnvCube_ = gfx::createSampler(device, VK_FILTER_LINEAR, VK_FILTER_LINEAR,
                                         VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                         VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 1.0f, false,
                                         VK_COMPARE_OP_LESS_OR_EQUAL, 0.0f, VK_LOD_CLAMP_NONE,
                                         false, VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE,
                                         "env cube");

    bindlessSamplers_.assign(kMaxBindlessTextures, samplerLinearRepeat_);
    if (!createDescriptorLayouts()) return false;
    if (!createPipelines()) return false;
    // Shadow atlas and environment probe are referenced by the lighting/glass
    // descriptor sets, so they must exist before the render targets are bound.
    if (!createShadowResources()) return false;
    if (!createRenderTargets()) return false;
    initializeImageLayouts();

    // Reserved bindless slots 0..2 (white, flat normal, black). Material textures are
    // handed out from kReservedTextures onwards so the two ranges never overlap.
    textureSlotCount_ = scene::kReservedTextures;
    stats_.textureCount = textureSlotCount_;
    // Placeholder bindless slots 0..2 (white, flat normal, black).
    {
        procgen::TextureData white;
        white.resize(1, 1, Vec4(1, 1, 1, 1));
        procgen::TextureData flat;
        flat.resize(1, 1, Vec4(0.5f, 0.5f, 1.0f, 1.0f));
        procgen::TextureData black;
        black.resize(1, 1, Vec4(0, 0, 0, 1));
        pendingTextures_.insert(pendingTextures_.begin(),
                                {PendingTexture{white, {}, false, false, false, "white"},
                                 PendingTexture{flat, {}, false, false, false, "flat normal"},
                                 PendingTexture{black, {}, false, false, false, "black"}});
    }

    R2_INFO("renderer initialised: render ", renderWidth_, "x", renderHeight_, ", output ",
            outputWidth_, "x", outputHeight_, headless_ ? " (headless)" : "");
    return true;
}

bool Renderer::createDescriptorLayouts() {
    gfx::Device& d = *device_;
    using B = gfx::DescriptorBinding;

    setLayoutGlobals_ = gfx::createDescriptorSetLayout(
        d,
        {
            B{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL},
            B{1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL},
            B{2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL},
            B{3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL},
        },
        "set:globals");

    setLayoutTextures_ = gfx::createDescriptorSetLayout(
        d,
        {B{0,
           VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
           kMaxBindlessTextures,
           VK_SHADER_STAGE_ALL,
           bindlessSamplers_.data(),
           VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
               VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT}},
        "set:bindless");

    setLayoutEmpty_ = gfx::createDescriptorSetLayout(d, {}, "set:empty");

    auto sampled = [](uint32_t binding, VkShaderStageFlags stages) {
        return B{binding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, stages};
    };
    auto storageImage = [](uint32_t binding, VkShaderStageFlags stages) {
        return B{binding, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, stages};
    };
    const VkShaderStageFlags cs = VK_SHADER_STAGE_COMPUTE_BIT;
    const VkShaderStageFlags fs = VK_SHADER_STAGE_FRAGMENT_BIT;

    setLayoutShadow_ = setLayoutEmpty_;
    setLayoutLighting_ = gfx::createDescriptorSetLayout(
        d,
        {sampled(0, cs), sampled(1, cs), sampled(2, cs), sampled(3, cs), sampled(4, cs),
         sampled(5, cs), sampled(6, cs), sampled(7, cs), storageImage(8, cs), sampled(9, cs)},
        "set:lighting");
    setLayoutSsao_ = gfx::createDescriptorSetLayout(
        d, {sampled(0, cs), sampled(1, cs), storageImage(3, cs), sampled(4, cs)}, "set:ssao");
    setLayoutSsaoBlur_ = gfx::createDescriptorSetLayout(
        d, {sampled(0, cs), sampled(1, cs), storageImage(2, cs), sampled(3, cs)},
        "set:ssao_blur");
    setLayoutBloomPrefilter_ = gfx::createDescriptorSetLayout(
        d, {sampled(0, cs), storageImage(1, cs)}, "set:bloom_prefilter");
    setLayoutBloomDown_ =
        gfx::createDescriptorSetLayout(d, {sampled(0, cs), storageImage(1, cs)}, "set:bloom_down");
    setLayoutBloomUp_ = gfx::createDescriptorSetLayout(
        d, {sampled(0, cs), sampled(1, cs), storageImage(2, cs)}, "set:bloom_up");
    setLayoutGlass_ = gfx::createDescriptorSetLayout(
        d, {sampled(0, fs), sampled(1, fs), sampled(2, fs), sampled(3, fs)}, "set:glass");
    setLayoutTonemap_ =
        gfx::createDescriptorSetLayout(d, {sampled(0, fs), sampled(1, fs)}, "set:tonemap");
    setLayoutUi_ = gfx::createDescriptorSetLayout(d, {sampled(0, fs)}, "set:ui");
    setLayoutProbe_ = gfx::createDescriptorSetLayout(d, {storageImage(0, cs)}, "set:probe");
    setLayoutTaa_ = gfx::createDescriptorSetLayout(
        d, {sampled(0, cs), sampled(1, cs), sampled(2, cs), storageImage(3, cs), sampled(4, cs)},
        "set:taa");

    std::vector<VkDescriptorPoolSize> sizes = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 64},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 128},
    };
    mainPool_.init(d, sizes, 256, false, "main descriptor pool");
    bindlessPool_.init(d, {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxBindlessTextures}}, 1,
                       true, "bindless descriptor pool");
    setTextures_ = bindlessPool_.allocate(setLayoutTextures_, "bindless set");

    for (int i = 0; i < config_.maxFramesInFlight; ++i)
        setGlobals_[i] = mainPool_.allocate(setLayoutGlobals_, "globals set");
    setShadow_ = mainPool_.allocate(setLayoutShadow_, "shadow set");
    setLighting_ = mainPool_.allocate(setLayoutLighting_, "lighting set");
    setSsao_ = mainPool_.allocate(setLayoutSsao_, "ssao set");
    setSsaoBlur_ = mainPool_.allocate(setLayoutSsaoBlur_, "ssao blur set");
    for (uint32_t i = 0; i < kFrameCountMax; ++i) {
        setBloomDown_[i] = mainPool_.allocate(setLayoutBloomDown_, "bloom down set");
        setBloomUp_[i] = mainPool_.allocate(setLayoutBloomUp_, "bloom up set");
        setBloomPrefilter_[i] = mainPool_.allocate(setLayoutBloomPrefilter_, "bloom prefilter set");
        setTonemap_[i] = mainPool_.allocate(setLayoutTonemap_, "tonemap set");
    }
    setGlass_ = mainPool_.allocate(setLayoutGlass_, "glass set");
    setUi_ = mainPool_.allocate(setLayoutUi_, "ui set");
    setProbe_ = mainPool_.allocate(setLayoutProbe_, "probe set");
    for (int i = 0; i < 2; ++i) {
        setTaa_[i] = mainPool_.allocate(setLayoutTaa_, "taa set");
        setTaaPrev_[i] = mainPool_.allocate(setLayoutTaa_, "taa prev set");
    }
    return true;
}

// ---------------------------------------------------------------- shaders / pipelines
bool Renderer::createPipelines() {
    gfx::Device& d = *device_;
    auto load = [&](const char* name) { return gfx::loadShaderModule(d, gfx::resolveShaderPath(name)); };

    vsFullscreen_ = load("fullscreen.vert.spv");
    vsGbuffer_ = load("gbuffer.vert.spv");
    fsGbuffer_ = load("gbuffer.frag.spv");
    vsShadow_ = load("shadow.vert.spv");
    fsShadow_ = load("shadow.frag.spv");
    vsGlass_ = load("glass.vert.spv");
    fsGlass_ = load("glass.frag.spv");
    csLighting_ = load("lighting.comp.spv");
    csSsao_ = load("ssao.comp.spv");
    csSsaoBlur_ = load("ssao_blur.comp.spv");
    csBloomPrefilter_ = load("bloom_prefilter.comp.spv");
    csBloomDown_ = load("bloom_down.comp.spv");
    csBloomUp_ = load("bloom_up.comp.spv");
    fsTonemap_ = load("tonemap.frag.spv");
    vsUi_ = load("ui.vert.spv");
    fsUi_ = load("ui.frag.spv");
    csProbe_ = load("env_probe.comp.spv");
    csTaa_ = load("taa.comp.spv");

    // --- pipeline layouts -------------------------------------------------
    auto makeLayout = [&](std::initializer_list<VkDescriptorSetLayout> sets, uint32_t pushSize,
                          const char* name) {
        VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        std::vector<VkDescriptorSetLayout> v(sets);
        info.setLayoutCount = static_cast<uint32_t>(v.size());
        info.pSetLayouts = v.empty() ? nullptr : v.data();
        VkPushConstantRange range{};
        if (pushSize > 0) {
            range.stageFlags = VK_SHADER_STAGE_ALL;
            range.offset = 0;
            range.size = pushSize;
            info.pushConstantRangeCount = 1;
            info.pPushConstantRanges = &range;
        }
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VK_CHECK(vkCreatePipelineLayout(d.device(), &info, nullptr, &layout));
        if (name) VK_NAME(d.device(), layout, "%s", name);
        return layout;
    };

    layoutGbuffer_ = makeLayout({setLayoutGlobals_, setLayoutTextures_, setLayoutEmpty_},
                                sizeof(DrawPush), "layout:gbuffer");
    layoutShadow_ = makeLayout({setLayoutGlobals_, setLayoutTextures_, setLayoutEmpty_},
                               sizeof(ShadowPush), "layout:shadow");
    layoutGlass_ = makeLayout({setLayoutGlobals_, setLayoutTextures_, setLayoutGlass_},
                              sizeof(DrawPush), "layout:glass");
    layoutLighting_ = makeLayout({setLayoutGlobals_, setLayoutTextures_, setLayoutLighting_}, 0,
                                 "layout:lighting");
    layoutSsao_ = makeLayout({setLayoutGlobals_, setLayoutTextures_, setLayoutSsao_}, 0,
                             "layout:ssao");
    layoutSsaoBlur_ = makeLayout({setLayoutGlobals_, setLayoutTextures_, setLayoutSsaoBlur_}, 0,
                                 "layout:ssao_blur");
    layoutBloomPrefilter_ =
        makeLayout({setLayoutGlobals_, setLayoutTextures_, setLayoutBloomPrefilter_},
                   sizeof(BloomPush), "layout:bloom_prefilter");
    layoutBloomDown_ = makeLayout({setLayoutGlobals_, setLayoutTextures_, setLayoutBloomDown_},
                                  sizeof(BloomPush), "layout:bloom_down");
    layoutBloomUp_ = makeLayout({setLayoutGlobals_, setLayoutTextures_, setLayoutBloomUp_},
                                sizeof(BloomPush), "layout:bloom_up");
    layoutTonemap_ = makeLayout({setLayoutGlobals_, setLayoutTextures_, setLayoutTonemap_}, 0,
                                "layout:tonemap");
    layoutUi_ = makeLayout({setLayoutGlobals_, setLayoutTextures_, setLayoutUi_}, sizeof(UiPush),
                           "layout:ui");
    layoutProbe_ = makeLayout({setLayoutGlobals_, setLayoutTextures_, setLayoutProbe_},
                              sizeof(ProbePush), "layout:probe");
    layoutTaa_ = makeLayout({setLayoutGlobals_, setLayoutTextures_, setLayoutTaa_}, 0, "layout:taa");

    // --- graphics pipelines ----------------------------------------------
    {
        gfx::GraphicsPipelineDesc desc;
        desc.vertexShader = vsGbuffer_;
        desc.fragmentShader = fsGbuffer_;
        desc.vertexBindings = {{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX}};
        desc.vertexAttributes = {
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
            {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
            {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, tangent)},
            {3, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)},
            {4, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv1)},
        };
        desc.colorFormats = {kGbufferAlbedoFormat, kGbufferNormalFormat, kGbufferMetalFormat,
                             kGbufferEmissiveFormat};
        desc.depthFormat = kDepthFormat;
        desc.cullMode = VK_CULL_MODE_BACK_BIT;
        desc.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        desc.layout = layoutGbuffer_;
        pipelineGbuffer_ = gfx::createGraphicsPipeline(d, pipelineCache_, desc, "pipeline:gbuffer");
    }
    {
        gfx::GraphicsPipelineDesc desc;
        desc.vertexShader = vsShadow_;
        desc.fragmentShader = VK_NULL_HANDLE;
        desc.vertexBindings = {{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX}};
        desc.vertexAttributes = {
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
        };
        desc.colorFormats = {};
        desc.depthFormat = kShadowFormat;
        desc.cullMode = VK_CULL_MODE_NONE;   // room shell is single sided
        desc.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        desc.depthBiasEnable = true;
        desc.depthBiasConstant = 1.5f;
        desc.depthBiasSlope = 2.5f;
        desc.layout = layoutShadow_;
        pipelineShadow_ = gfx::createGraphicsPipeline(d, pipelineCache_, desc, "pipeline:shadow");
    }
    {
        gfx::GraphicsPipelineDesc desc;
        desc.vertexShader = vsGlass_;
        desc.fragmentShader = fsGlass_;
        desc.vertexBindings = {{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX}};
        desc.vertexAttributes = {
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
            {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
            {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, tangent)},
            {3, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)},
            {4, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv1)},
        };
        desc.colorFormats = {kHdrFormat};
        desc.depthFormat = kDepthFormat;
        desc.cullMode = VK_CULL_MODE_NONE;   // render both shells of the glass
        desc.depthTestEnable = true;
        desc.depthWriteEnable = false;
        desc.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        desc.blendEnable = true;
        desc.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        desc.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        desc.colorBlendOp = VK_BLEND_OP_ADD;
        desc.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        desc.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        desc.layout = layoutGlass_;
        pipelineGlass_ = gfx::createGraphicsPipeline(d, pipelineCache_, desc, "pipeline:glass");
    }
    {
        gfx::GraphicsPipelineDesc desc;
        desc.vertexShader = vsFullscreen_;
        desc.fragmentShader = fsTonemap_;
        desc.colorFormats = {headless_ ? VK_FORMAT_R8G8B8A8_UNORM : swapchain_->format()};
        desc.depthFormat = VK_FORMAT_UNDEFINED;
        desc.depthTestEnable = false;
        desc.depthWriteEnable = false;
        desc.cullMode = VK_CULL_MODE_NONE;
        desc.layout = layoutTonemap_;
        pipelineTonemap_ = gfx::createGraphicsPipeline(d, pipelineCache_, desc, "pipeline:tonemap");
    }
    {
        gfx::GraphicsPipelineDesc desc;
        desc.vertexShader = vsUi_;
        desc.fragmentShader = fsUi_;
        desc.vertexBindings = {{0, sizeof(UiVertex), VK_VERTEX_INPUT_RATE_VERTEX}};
        desc.vertexAttributes = {
            {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(UiVertex, position)},
            {1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(UiVertex, uv)},
            {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(UiVertex, color)},
        };
        desc.colorFormats = {headless_ ? VK_FORMAT_R8G8B8A8_UNORM : swapchain_->format()};
        desc.depthFormat = VK_FORMAT_UNDEFINED;
        desc.depthTestEnable = false;
        desc.depthWriteEnable = false;
        desc.cullMode = VK_CULL_MODE_NONE;
        desc.blendEnable = true;
        desc.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        desc.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        desc.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        desc.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        desc.layout = layoutUi_;
        pipelineUi_ = gfx::createGraphicsPipeline(d, pipelineCache_, desc, "pipeline:ui");
    }

    // --- compute pipelines ------------------------------------------------
    auto compute = [&](VkShaderModule shader, VkPipelineLayout layout, const char* name) {
        gfx::ComputePipelineDesc desc;
        desc.shader = shader;
        desc.layout = layout;
        return gfx::createComputePipeline(d, pipelineCache_, desc, name);
    };
    pipelineLighting_ = compute(csLighting_, layoutLighting_, "pipeline:lighting");
    pipelineSsao_ = compute(csSsao_, layoutSsao_, "pipeline:ssao");
    pipelineSsaoBlur_ = compute(csSsaoBlur_, layoutSsaoBlur_, "pipeline:ssao_blur");
    pipelineBloomPrefilter_ =
        compute(csBloomPrefilter_, layoutBloomPrefilter_, "pipeline:bloom_prefilter");
    pipelineBloomDown_ = compute(csBloomDown_, layoutBloomDown_, "pipeline:bloom_down");
    pipelineBloomUp_ = compute(csBloomUp_, layoutBloomUp_, "pipeline:bloom_up");
    pipelineProbe_ = compute(csProbe_, layoutProbe_, "pipeline:probe");
    pipelineTaa_ = compute(csTaa_, layoutTaa_, "pipeline:taa");
    return true;
}

// ---------------------------------------------------------------- render targets
void Renderer::destroyRenderTargets() {
    gfx::destroyImage(*device_, gbufferAlbedo_);
    gfx::destroyImage(*device_, gbufferNormal_);
    gfx::destroyImage(*device_, gbufferMetal_);
    gfx::destroyImage(*device_, gbufferEmissive_);
    gfx::destroyImage(*device_, depth_);
    gfx::destroyImage(*device_, hdrColor_);
    gfx::destroyImage(*device_, hdrOpaqueCopy_);
    for (auto& h : hdrHistory_) gfx::destroyImage(*device_, h);
    gfx::destroyImage(*device_, aoRaw_);
    gfx::destroyImage(*device_, aoBlur_);
    for (auto& b : bloomMips_) gfx::destroyImage(*device_, b);
    gfx::destroyImage(*device_, ldrColor_);
}

bool Renderer::createRenderTargets() {
    gfx::Device& d = *device_;
    destroyRenderTargets();

    const uint32_t w = renderWidth_, h = renderHeight_;
    const uint32_t aoW = std::max(1u, w / 2), aoH = std::max(1u, h / 2);
    const uint32_t outW = headless_ ? outputWidth_ : swapchain_->extent().width;
    const uint32_t outH = headless_ ? outputHeight_ : swapchain_->extent().height;

    auto make2D = [&](VkFormat format, uint32_t width, uint32_t height, VkImageUsageFlags usage,
                      const char* name) {
        gfx::ImageDesc desc;
        desc.format = format;
        desc.extent = {width, height, 1};
        desc.usage = usage;
        desc.name = name;
        return gfx::createImage(d, desc);
    };

    const VkImageUsageFlags rt = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    gbufferAlbedo_ = make2D(kGbufferAlbedoFormat, w, h, rt, "gbuffer albedo");
    gbufferNormal_ = make2D(kGbufferNormalFormat, w, h, rt, "gbuffer normal");
    gbufferMetal_ = make2D(kGbufferMetalFormat, w, h, rt, "gbuffer metal");
    gbufferEmissive_ = make2D(kGbufferEmissiveFormat, w, h, rt, "gbuffer emissive");
    depth_ = make2D(kDepthFormat, w, h,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                    "depth");
    hdrColor_ = make2D(kHdrFormat, w, h,
                       rt | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                       "hdr color");
    hdrOpaqueCopy_ = make2D(kHdrFormat, w, h,
                            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            "hdr opaque copy");
    for (int i = 0; i < 2; ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "hdr history %d", i);
        hdrHistory_[i] = make2D(kHdrFormat, w, h,
                                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                name);
    }
    aoRaw_ = make2D(kAoFormat, aoW, aoH, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    "ao raw");
    aoBlur_ = make2D(kAoFormat, aoW, aoH,
                     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                     "ao blurred");

    bloomMipCount_ = 0;
    uint32_t bw = std::max(1u, w / 2), bh = std::max(1u, h / 2);
    for (uint32_t i = 0; i < kFrameCountMax; ++i) {
        if (bw < 4 || bh < 4) break;
        bloomMips_[i] = make2D(kHdrFormat, bw, bh,
                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                               "bloom mip");
        ++bloomMipCount_;
        bw = std::max(1u, bw / 2);
        bh = std::max(1u, bh / 2);
    }

    if (headless_) {
        ldrColor_ = make2D(VK_FORMAT_R8G8B8A8_UNORM, outW, outH,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                               VK_IMAGE_USAGE_SAMPLED_BIT,
                           "ldr color");
    }
    (void)outW;
    (void)outH;

    // --- descriptor updates ----------------------------------------------
    std::vector<VkWriteDescriptorSet> writes;
    // std::deque: push_back must not invalidate the pointers stored in `writes`.
    std::deque<VkDescriptorImageInfo> infos;

    auto addImage = [&](VkDescriptorSet set, uint32_t binding, const gfx::Image& image,
                        VkSampler sampler) {
        infos.push_back({sampler, image.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
        VkWriteDescriptorSet w{};
        gfx::writeImage(w, set, binding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &infos.back());
        writes.push_back(w);
    };
    auto addStorage = [&](VkDescriptorSet set, uint32_t binding, const gfx::Image& image) {
        infos.push_back({VK_NULL_HANDLE, image.view, VK_IMAGE_LAYOUT_GENERAL});
        VkWriteDescriptorSet w{};
        gfx::writeImage(w, set, binding, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &infos.back());
        writes.push_back(w);
    };
    // Compute-written images stay in VK_IMAGE_LAYOUT_GENERAL for their whole life.
    // That removes a whole class of layout-transition bugs between the storage-image
    // write and the sampled read of the same image within one pass chain.
    auto addGeneral = [&](VkDescriptorSet set, uint32_t binding, const gfx::Image& image,
                          VkSampler sampler) {
        infos.push_back({sampler, image.view, VK_IMAGE_LAYOUT_GENERAL});
        VkWriteDescriptorSet w{};
        gfx::writeImage(w, set, binding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &infos.back());
        writes.push_back(w);
    };

    // Lighting
    {
        VkDescriptorSet s = setLighting_;
        addGeneral(s, 0, gbufferAlbedo_, samplerNearestClamp_);
        addGeneral(s, 1, gbufferNormal_, samplerNearestClamp_);
        addGeneral(s, 2, gbufferMetal_, samplerNearestClamp_);
        addGeneral(s, 3, gbufferEmissive_, samplerNearestClamp_);
        addGeneral(s, 4, depth_, samplerNearestClamp_);
        addGeneral(s, 5, aoBlur_, samplerLinearClamp_);
        addGeneral(s, 6, shadowAtlas_, samplerShadow_);
        addGeneral(s, 7, envProbe_, samplerEnvCube_);
        addStorage(s, 8, hdrColor_);
        // Diagnostic-only: the same cube bound with a plain sampler so a debug mode can
        // read the raw stored depth and compare it against the shader's reference depth.
        addGeneral(s, 9, shadowAtlas_, samplerLinearClamp_);
    }
    {
        VkDescriptorSet s = setSsao_;
        addGeneral(s, 0, depth_, samplerNearestClamp_);
        addGeneral(s, 1, gbufferNormal_, samplerNearestClamp_);
        addStorage(s, 3, aoRaw_);
        addGeneral(s, 4, gbufferEmissive_, samplerNearestClamp_);
    }
    {
        VkDescriptorSet s = setSsaoBlur_;
        addGeneral(s, 0, depth_, samplerNearestClamp_);
        addGeneral(s, 1, aoRaw_, samplerLinearClamp_);
        addStorage(s, 2, aoBlur_);
        addGeneral(s, 3, gbufferEmissive_, samplerNearestClamp_);
    }
    for (int i = 0; i < config_.maxFramesInFlight; ++i) {
        VkDescriptorSet s = setBloomPrefilter_[i];
        addGeneral(s, 0, hdrHistory_[0], samplerLinearClamp_);
        addStorage(s, 1, bloomMips_[0]);
    }
    for (uint32_t i = 0; i + 1 < bloomMipCount_; ++i) {
        VkDescriptorSet s = setBloomDown_[i];
        addGeneral(s, 0, bloomMips_[i], samplerLinearClamp_);
        addStorage(s, 1, bloomMips_[i + 1]);
    }
    for (uint32_t i = 1; i < bloomMipCount_; ++i) {
        VkDescriptorSet s = setBloomUp_[i - 1];
        addGeneral(s, 0, bloomMips_[i], samplerLinearClamp_);
        addGeneral(s, 1, bloomMips_[i - 1], samplerLinearClamp_);
        addStorage(s, 2, bloomMips_[i - 1]);
    }
    {
        VkDescriptorSet s = setGlass_;
        addGeneral(s, 0, hdrOpaqueCopy_, samplerLinearClamp_);
        addGeneral(s, 1, depth_, samplerNearestClamp_);
        addGeneral(s, 2, envProbe_, samplerEnvCube_);
        addGeneral(s, 3, shadowAtlas_, samplerShadow_);
    }
    for (int i = 0; i < config_.maxFramesInFlight; ++i) {
        VkDescriptorSet s = setTonemap_[i];
        addGeneral(s, 0, hdrHistory_[0], samplerLinearClamp_);
        addGeneral(s, 1, bloomMips_[0], samplerLinearClamp_);
    }
    for (int i = 0; i < 2; ++i) {
        VkDescriptorSet s = setTaa_[i];
        addGeneral(s, 0, hdrColor_, samplerNearestClamp_);
        addGeneral(s, 1, depth_, samplerNearestClamp_);
        addGeneral(s, 2, hdrHistory_[i], samplerLinearClamp_);
        addStorage(s, 3, hdrHistory_[1 - i]);
        addGeneral(s, 4, gbufferEmissive_, samplerNearestClamp_);
    }
    {
        VkDescriptorSet s = setProbe_;
        addStorage(s, 0, envProbe_);
    }
    if (!writes.empty())
        vkUpdateDescriptorSets(d.device(), static_cast<uint32_t>(writes.size()), writes.data(), 0,
                               nullptr);

    historyValid_ = false;
    return true;
}

// Every offscreen target permanently lives in VK_IMAGE_LAYOUT_GENERAL. Doing the
// UNDEFINED -> GENERAL transition once at creation removes all per-frame layout
// bookkeeping and makes sampling an image that is simultaneously bound as an
// attachment (depth, in the glass pass) legal.
void Renderer::initializeImageLayouts() {
    std::vector<VkImage> images;
    auto add = [&](const gfx::Image& image) {
        if (image.handle) images.push_back(image.handle);
    };
    add(gbufferAlbedo_); add(gbufferNormal_); add(gbufferMetal_); add(gbufferEmissive_);
    add(hdrColor_); add(hdrOpaqueCopy_); add(hdrHistory_[0]); add(hdrHistory_[1]);
    add(aoRaw_); add(aoBlur_); add(ldrColor_);
    for (uint32_t i = 0; i < bloomMipCount_; ++i) add(bloomMips_[i]);

    VkCommandBuffer cmd = device_->beginImmediate("target layouts");
    for (VkImage image : images) {
        gfx::cmdImageBarrier(cmd, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                             VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_NONE,
                             VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                             VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0,
                             VK_REMAINING_ARRAY_LAYERS);
    }
    // Depth-aspect images need a depth-aspect transition, not a colour one.
    for (VkImage image : {depth_.handle, shadowAtlas_.handle}) {
        gfx::cmdImageBarrier(cmd, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                             VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_NONE,
                             VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                             VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0,
                             VK_REMAINING_ARRAY_LAYERS);
    }
    device_->endImmediate(cmd, true);
}

bool Renderer::createShadowResources() {
    gfx::Device& d = *device_;
    const uint32_t size = static_cast<uint32_t>(config_.shadowResolution);
    const uint32_t layers = kMaxShadowLights;
    gfx::ImageDesc desc;
    desc.format = kShadowFormat;
    desc.extent = {size, size, 1};
    desc.arrayLayers = layers;
    desc.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    desc.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    desc.name = "shadow maps";
    shadowAtlas_ = gfx::createImage(d, desc);

    shadowFaceViews_.assign(layers, VK_NULL_HANDLE);
    for (uint32_t layer = 0; layer < layers; ++layer) {
        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = shadowAtlas_.handle;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = shadowAtlas_.format;
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, layer, 1};
        VK_CHECK(vkCreateImageView(d.device(), &viewInfo, nullptr, &shadowFaceViews_[layer]));
        VK_NAME(d.device(), shadowFaceViews_[layer], "shadow face %u", layer);
    }

    // Environment probe cubemap, filled analytically at load time.
    const uint32_t probeSize = static_cast<uint32_t>(config_.envProbeSize);
    gfx::ImageDesc probeDesc;
    probeDesc.format = kHdrFormat;
    probeDesc.extent = {probeSize, probeSize, 1};
    probeDesc.arrayLayers = 6;
    probeDesc.mipLevels = gfx::mipCountFor(probeDesc.extent);
    probeDesc.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    probeDesc.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    probeDesc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    probeDesc.name = "env probe";
    envProbe_ = gfx::createImage(d, probeDesc);
    return true;
}

// ---------------------------------------------------------------- textures
uint32_t Renderer::addTexture(const procgen::TextureData& image, bool srgb, bool generateMips,
                              const std::string& name) {
    if (textureSlotCount_ >= kMaxBindlessTextures) {
        R2_ERROR("bindless texture slots exhausted (", kMaxBindlessTextures, ")");
        return scene::kTextureWhite;
    }
    PendingTexture pending;
    pending.image = image;
    pending.srgb = srgb;
    pending.generateMips = generateMips;
    pending.cube = false;
    pending.name = name;
    pendingTextures_.push_back(std::move(pending));
    texturesDirty_ = true;
    return textureSlotCount_++;
}

uint32_t Renderer::addCubeTexture(const std::vector<procgen::TextureData>& faces,
                                  const std::string& name, bool srgb) {
    if (textureSlotCount_ >= kMaxBindlessTextures) return scene::kTextureWhite;
    PendingTexture pending;
    pending.faces = faces;
    pending.cube = true;
    pending.srgb = srgb;
    pending.generateMips = true;
    pending.name = name;
    pendingTextures_.push_back(std::move(pending));
    texturesDirty_ = true;
    return textureSlotCount_++;
}

void Renderer::flushTextureUploads() {
    if (!texturesDirty_ || pendingTextures_.empty()) return;
    gfx::Device& d = *device_;
    gfx::UploadBatch batch(d);

    std::vector<uint8_t> scratch;
    std::vector<gfx::Image> newImages;
    newImages.reserve(pendingTextures_.size());
    size_t newBytes = 0;

    for (auto& pending : pendingTextures_) {
        const procgen::TextureData& src = pending.cube ? pending.faces[0] : pending.image;
        if (!src.valid()) {
            R2_WARN("skipping invalid texture '", pending.name, "'");
            newImages.push_back(gfx::Image{});
            continue;
        }
        gfx::ImageDesc desc;
        desc.format = pending.srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
        desc.extent = {src.width, src.height, 1};
        desc.arrayLayers = pending.cube ? 6u : 1u;
        desc.mipLevels = pending.generateMips ? gfx::mipCountFor(desc.extent) : 1u;
        desc.viewType = pending.cube ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
        desc.flags = pending.cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0u;
        desc.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        desc.name = pending.name.c_str();
        gfx::Image image = gfx::createImage(d, desc);

        const uint32_t layers = pending.cube ? 6u : 1u;
        for (uint32_t layer = 0; layer < layers; ++layer) {
            const procgen::TextureData& face = pending.cube ? pending.faces[layer] : pending.image;
            procgen::TextureData current = face;
            uint32_t w = desc.extent.width, h = desc.extent.height;
            for (uint32_t mip = 0; mip < desc.mipLevels; ++mip) {
                scratch = pending.srgb ? procgen::toSrgb8(current) : procgen::toLinear8(current);
                newBytes += scratch.size();
                batch.image(image, scratch.data(), scratch.size(), mip, layer);
                if (mip + 1 < desc.mipLevels) current = procgen::downsample(current);
            }
            (void)w;
            (void)h;
        }
        newImages.push_back(image);
    }
    batch.submit("texture upload");

    for (auto& image : newImages) textures_.push_back(image);
    pendingTextures_.clear();

    // Rebuild the bindless descriptor array. Slots that were skipped keep the
    // previous image (partially bound array), which is why slot 0 is the white texel.
    textureInfos_.clear();
    textureInfos_.reserve(textures_.size());
    for (auto& image : textures_) {
        if (!image) {
            textureInfos_.push_back({samplerLinearRepeat_, VK_NULL_HANDLE,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
        } else {
            textureInfos_.push_back({samplerLinearRepeat_, image.view,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
        }
    }
    // Keep the environment probe in its own slot at the end of the used range.
    envProbeSlot_ = static_cast<uint32_t>(textures_.size());
    textureInfos_.push_back({samplerEnvCube_, envProbe_.view,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});

    if (!textureInfos_.empty()) {
        VkWriteDescriptorSet write{};
        gfx::writeImage(write, setTextures_, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        textureInfos_.data(), static_cast<uint32_t>(textureInfos_.size()));
        vkUpdateDescriptorSets(d.device(), 1, &write, 0, nullptr);
    }
    stats_.textureBytes = static_cast<uint32_t>(newBytes);
    stats_.textureCount = static_cast<uint32_t>(textures_.size());
    texturesDirty_ = false;
    R2_INFO("uploaded ", textures_.size(), " textures (", newBytes / (1024 * 1024), " MiB)");
}

void Renderer::setUiAtlas(const gfx::Image& atlas) {
    VkDescriptorImageInfo info{samplerLinearClamp_, atlas.view,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet write{};
    gfx::writeImage(write, setUi_, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &info);
    vkUpdateDescriptorSets(device_->device(), 1, &write, 0, nullptr);
}

// ---------------------------------------------------------------- scene upload
void Renderer::uploadScene(scene::Scene& scene) {
    gfx::Device& d = *device_;
    flushTextureUploads();

    const auto& vertices = scene.vertices();
    const auto& indices = scene.indices();

    const size_t reserveV = std::max<size_t>(vertices.size() + 65536, 262144);
    const size_t reserveI = std::max<size_t>(indices.size() + 262144, 1048576);

    if (!vertexBuffer_ || reserveV > vertexCapacity_) {
        gfx::destroyBuffer(d, vertexBuffer_);
        gfx::BufferDesc desc;
        desc.size = reserveV * sizeof(Vertex);
        desc.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        desc.memory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        desc.name = "scene vertices";
        vertexBuffer_ = gfx::createBuffer(d, desc);
        vertexCapacity_ = reserveV;
    }
    if (!indexBuffer_ || reserveI > indexCapacity_) {
        gfx::destroyBuffer(d, indexBuffer_);
        gfx::BufferDesc desc;
        desc.size = reserveI * sizeof(uint32_t);
        desc.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        desc.memory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        desc.name = "scene indices";
        indexBuffer_ = gfx::createBuffer(d, desc);
        indexCapacity_ = reserveI;
    }

    if (!vertices.empty()) {
        gfx::UploadBatch batch(d);
        batch.buffer(vertexBuffer_, vertices.data(), vertices.size() * sizeof(Vertex));
        if (!indices.empty())
            batch.buffer(indexBuffer_, indices.data(), indices.size() * sizeof(uint32_t));
        batch.submit("scene geometry");
    }

    // --- per-frame host-visible buffers -----------------------------------
    const size_t instanceCount = std::max<size_t>(1024, scene.instanceCount() + 512);
    if (instanceBuffers_.empty() || instanceCount > instanceCapacity_) {
        for (auto& b : instanceBuffers_) gfx::destroyBuffer(d, b);
        instanceBuffers_.clear();
        gfx::BufferDesc desc;
        desc.size = instanceCount * sizeof(GpuInstance);
        desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        desc.memory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        desc.name = "instances";
        for (int i = 0; i < config_.maxFramesInFlight; ++i)
            instanceBuffers_.push_back(gfx::createBuffer(d, desc));
        instanceCapacity_ = instanceCount;
    }
    if (globalsBuffers_.empty()) {
        gfx::BufferDesc desc;
        desc.size = sizeof(GpuGlobals);
        desc.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        desc.memory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        desc.name = "globals";
        for (int i = 0; i < config_.maxFramesInFlight; ++i)
            globalsBuffers_.push_back(gfx::createBuffer(d, desc));
    }
    const size_t materialCount = std::max<size_t>(256, scene.materialCount() + 64);
    if (!materialBuffer_ || materialCount > materialCapacity_) {
        gfx::destroyBuffer(d, materialBuffer_);
        gfx::BufferDesc desc;
        desc.size = materialCount * sizeof(GpuMaterial);
        desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        desc.memory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        desc.name = "materials";
        materialBuffer_ = gfx::createBuffer(d, desc);
        materialCapacity_ = materialCount;
    }
    const size_t lightCount = std::max<size_t>(16, scene.lights().size() + 8);
    if (!lightBuffer_ || lightCount > lightCapacity_) {
        gfx::destroyBuffer(d, lightBuffer_);
        gfx::BufferDesc desc;
        desc.size = lightCount * sizeof(GpuLight);
        desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        desc.memory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        desc.name = "lights";
        lightBuffer_ = gfx::createBuffer(d, desc);
        lightCapacity_ = lightCount;
    }

    // --- bind the global descriptor sets ----------------------------------
    std::vector<VkWriteDescriptorSet> writes;
    // std::deque again: the writes below hold pointers into this container.
    std::deque<VkDescriptorBufferInfo> bufferInfos;
    for (int i = 0; i < config_.maxFramesInFlight; ++i) {
        VkDescriptorBufferInfo globalsInfo{globalsBuffers_[i].handle, 0, sizeof(GpuGlobals)};
        VkDescriptorBufferInfo instancesInfo{instanceBuffers_[i].handle, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo materialsInfo{materialBuffer_.handle, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo lightsInfo{lightBuffer_.handle, 0, VK_WHOLE_SIZE};
        bufferInfos.push_back(globalsInfo);
        bufferInfos.push_back(instancesInfo);
        bufferInfos.push_back(materialsInfo);
        bufferInfos.push_back(lightsInfo);
        VkWriteDescriptorSet w[4];
        gfx::writeBuffer(w[0], setGlobals_[i], 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                         &bufferInfos[bufferInfos.size() - 4]);
        gfx::writeBuffer(w[1], setGlobals_[i], 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         &bufferInfos[bufferInfos.size() - 3]);
        gfx::writeBuffer(w[2], setGlobals_[i], 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         &bufferInfos[bufferInfos.size() - 2]);
        gfx::writeBuffer(w[3], setGlobals_[i], 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         &bufferInfos[bufferInfos.size() - 1]);
        for (auto& x : w) writes.push_back(x);
    }
    if (!writes.empty())
        vkUpdateDescriptorSets(d.device(), static_cast<uint32_t>(writes.size()), writes.data(), 0,
                               nullptr);

    updateMaterials(scene);
    updateLights(scene);
    updateInstances(scene);
    buildEnvironmentProbe(scene);

    stats_.gpuBufferBytes = static_cast<uint32_t>(
        (vertexBuffer_.size + indexBuffer_.size + materialBuffer_.size + lightBuffer_.size) /
        (1024 * 1024));
    needsFullUpload_ = false;
}

void Renderer::updateMaterials(const scene::Scene& scene) {
    if (!materialBuffer_ || !materialBuffer_.mapped) return;
    std::vector<GpuMaterial> gpu(scene.materialCount());
    for (uint32_t i = 0; i < scene.materialCount(); ++i) {
        const scene::Material& m = scene.material(i);
        GpuMaterial& g = gpu[i];
        g.baseColorFactor = m.baseColorFactor;
        g.emissiveFactor = Vec4(m.emissiveFactor, 0.0f);
        g.attenuationColor = Vec4(m.attenuationColor, m.thickness);
        g.texBaseColor = m.baseColorTex;
        g.texNormal = m.normalTex;
        g.texOrm = m.ormTex;
        g.texEmissive = m.emissiveTex;
        g.params0 = Vec4(m.metallic, m.roughness, m.normalScale, m.occlusionStrength);
        const float kind = m.isTransmissive() ? 1.0f : 0.0f;
        g.params1 = Vec4(m.transmission, m.ior, m.alpha, kind);
    }
    const size_t bytes = gpu.size() * sizeof(GpuMaterial);
    if (bytes) materialBuffer_.write(gpu.data(), bytes);
}

void Renderer::updateLights(const scene::Scene& scene) {
    if (!lightBuffer_ || !lightBuffer_.mapped) return;
    std::vector<GpuLight> gpu(std::max<size_t>(1, scene.lights().size()));
    uint32_t shadowIndex = 0;
    for (size_t i = 0; i < scene.lights().size(); ++i) {
        const scene::Light& l = scene.lights()[i];
        GpuLight& g = gpu[i];
        g.positionRange = Vec4(l.position, l.range);
        g.colorIntensity = Vec4(l.color, l.intensity);
        g.directionRadius = Vec4(normalize(l.direction), l.radius);
        const float type = static_cast<float>(static_cast<uint32_t>(l.type));
        float shadowFlag = 0.0f;
        if (l.castsShadow && config_.enableShadows && shadowIndex < kMaxShadowLights) {
            shadowFlag = static_cast<float>(shadowIndex + 1);
        }
        g.shadowParams = Vec4(l.innerConeCos, l.outerConeCos, type, shadowFlag);
        g.shadowNearFar = Vec4(l.shadowNear, l.shadowFar, 0.0f, 0.0f);
        for (int f = 0; f < 6; ++f) g.shadowViewProj[f] = l.shadowViewProj[f];
        if (shadowFlag > 0.0f) ++shadowIndex;
    }
    const size_t bytes = gpu.size() * sizeof(GpuLight);
    if (bytes) lightBuffer_.write(gpu.data(), bytes);
}

void Renderer::updateInstances(const scene::Scene& scene) {
    if (instanceBuffers_.empty()) return;
    gfx::Buffer& buffer = instanceBuffers_[frameIndex_ % instanceBuffers_.size()];
    if (!buffer.mapped) return;
    GpuInstance* dst = buffer.as<GpuInstance>();
    const uint32_t count = std::min<uint32_t>(scene.instanceCount(),
                                              static_cast<uint32_t>(instanceCapacity_));
    for (uint32_t i = 0; i < count; ++i) {
        const scene::Instance& inst = scene.instance(i);
        GpuInstance& g = dst[i];
        g.model = inst.transform;
        g.normalMatrix = mat4FromMat3(inst.normalMatrix);
        uint32_t flags = 0;
        if (inst.castsShadow) flags |= INSTANCE_CASTS_SHADOW;
        if (inst.dynamic) flags |= INSTANCE_DYNAMIC;
        g.materialIndex = inst.materialOverride;
        g.userData = inst.userData;
        g.flags = flags;
        g.pad = 0;
    }
    uploadedInstanceCount_ = count;
}

// ---------------------------------------------------------------- environment probe
void Renderer::buildEnvironmentProbe(const scene::Scene& scene) {
    // Derive the room box and lamp from the scene bounds and the first point light,
    // then evaluate the analytic radiance field into the cube mip chain. Rough mips
    // average many GGX-distributed rays, giving a usable prefiltered reflection probe.
    Aabb bounds = scene.bounds();
    if (!bounds.valid()) {
        bounds.mn = Vec3(-3, 0, -4);
        bounds.mx = Vec3(3, 3, 4);
    }
    Vec3 lampPos(0.0f, bounds.mx.y - 0.2f, 0.0f);
    Vec3 lampColor(1.0f, 0.96f, 0.90f);
    float lampIntensity = 6.0f;
    for (const auto& l : scene.lights()) {
        if (l.type == scene::LightType::Point) {
            lampPos = l.position;
            lampColor = l.color;
            lampIntensity = l.intensity;
            break;
        }
    }

    const uint32_t mips = envProbe_.mipLevels;
    const uint32_t baseSize = envProbe_.extent.width;
    const uint32_t samples = static_cast<uint32_t>(std::max(1, config_.envProbeSamples));

    // One 2D-array view per mip: a storage image view may only cover a single level.
    for (VkImageView v : envProbeMipViews_)
        if (v) vkDestroyImageView(device_->device(), v, nullptr);
    envProbeMipViews_.clear();
    for (uint32_t mip = 0; mip < mips; ++mip) {
        VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        info.image = envProbe_.handle;
        info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        info.format = envProbe_.format;
        info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 6};
        VkImageView view = VK_NULL_HANDLE;
        VK_CHECK(vkCreateImageView(device_->device(), &info, nullptr, &view));
        envProbeMipViews_.push_back(view);
    }

    probePool_.init(*device_,
                    {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, mips * 6}},
                    mips * 6, false, "probe pool");

    VkCommandBuffer cmd = device_->beginImmediate("env probe");
    device_->beginLabel(cmd, "environment probe", 0.4f, 0.8f, 0.4f);
    gfx::cmdImageBarrier(cmd, envProbe_.handle, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_NONE,
                         VK_ACCESS_2_SHADER_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 6);

    const Vec3 bounce =
        scene.environment().wallBounce * 2.2f + scene.environment().ceilingBounce * 1.4f;
    for (uint32_t mip = 0; mip < mips; ++mip) {
        const uint32_t size = mipSize(baseSize, mip);
        const float roughness =
            mips > 1 ? static_cast<float>(mip) / static_cast<float>(mips - 1) : 0.0f;
        for (uint32_t face = 0; face < 6; ++face) {
            VkDescriptorSet set = probePool_.allocate(setLayoutProbe_, "probe set");
            VkDescriptorImageInfo imageInfo{VK_NULL_HANDLE, envProbeMipViews_[mip],
                                            VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSet write{};
            gfx::writeImage(write, set, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageInfo);
            vkUpdateDescriptorSets(device_->device(), 1, &write, 0, nullptr);

            ProbePush push;
            push.roomMin = Vec4(bounds.mn, 0.0f);
            push.roomMax = Vec4(bounds.mx, 0.0f);
            push.lightPosIntensity = Vec4(lampPos, lampIntensity);
            push.lightColorRoughness = Vec4(lampColor, roughness);
            push.bounce = Vec4(bounce, 1.0f);
            push.mipLevel = mip;
            push.faceSize = size;
            push.sampleCount = mip == 0 ? 1u : samples;
            push.faceIndex = face;

            VkDescriptorSet sets[3] = {setGlobals_[0], setTextures_, set};
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineProbe_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layoutProbe_, 0, 3, sets,
                                    0, nullptr);
            vkCmdPushConstants(cmd, layoutProbe_, VK_SHADER_STAGE_ALL, 0, sizeof(push), &push);
            const uint32_t groups = (size + 7) / 8;
            vkCmdDispatch(cmd, groups, groups, 1);
            gfx::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
        }
    }
    gfx::cmdImageBarrier(cmd, envProbe_.handle, VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                         VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 6);
    device_->endLabel(cmd);
    device_->endImmediate(cmd, true);

    // Keep the probe in the bindless array so the shaders can sample it by slot.
    if (envProbeSlot_ < textureInfos_.size()) {
        textureInfos_[envProbeSlot_] = {samplerEnvCube_, envProbe_.view,
                                        VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet write{};
        gfx::writeImage(write, setTextures_, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        &textureInfos_[envProbeSlot_], 1,
                        static_cast<uint32_t>(envProbeSlot_));
        vkUpdateDescriptorSets(device_->device(), 1, &write, 0, nullptr);
    }
    R2_INFO("environment probe built (", baseSize, "px cube, ", mips, " mips)");
}

// ---------------------------------------------------------------- frame plumbing
void Renderer::resize(uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) return;
    if (width == outputWidth_ && height == outputHeight_ && renderWidth_ > 0) return;
    outputWidth_ = width;
    outputHeight_ = height;
    renderWidth_ = std::max(1u, static_cast<uint32_t>(width * config_.renderScale));
    renderHeight_ = std::max(1u, static_cast<uint32_t>(height * config_.renderScale));
    device_->waitIdle();
    createRenderTargets();
    initializeImageLayouts();
    R2_INFO("renderer resized to ", renderWidth_, "x", renderHeight_);
}

bool Renderer::beginFrame() {
    const uint32_t slot = frameIndex_ % config_.maxFramesInFlight;
    // A bounded wait: a GPU stall should surface as a logged error rather than a frozen
    // window that the compositor reports as "not responding".
    const VkResult fenceResult =
        vkWaitForFences(device_->device(), 1, &frameFences_[slot], VK_TRUE, 5000000000ull);
    if (fenceResult == VK_TIMEOUT) {
        R2_ERROR("timed out waiting for frame ", frameIndex_, " to retire; the GPU or the "
                 "presentation queue appears to be stalled");
        return false;
    }
    VK_CHECK(fenceResult);

    if (!headless_) {
        if (swapchain_->needsRecreate()) return false;
        if (swapchain_->extent().width != outputWidth_ ||
            swapchain_->extent().height != outputHeight_) {
            resize(swapchain_->extent().width, swapchain_->extent().height);
        }
        if (!swapchain_->acquire(slot, imageAvailable_[slot], swapchainImageIndex_)) return false;
    }

    VK_CHECK(vkResetFences(device_->device(), 1, &frameFences_[slot]));
    VkCommandBuffer cmd = commandBuffers_[slot];
    VK_CHECK(vkResetCommandBuffer(cmd, 0));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));
    frameOpen_ = true;
    return true;
}

void Renderer::writeGlobals(const scene::Scene& scene, const Camera& camera, float dt,
                            float exposure) {
    const uint32_t slot = frameIndex_ % config_.maxFramesInFlight;
    GpuGlobals globals;
    const Mat4 view = camera.view();
    const Mat4 viewProj = camera.jitteredViewProjection();
    globals.view = view;
    globals.projection = camera.projectionUnjittered();
    globals.viewProjection = viewProj;
    globals.invViewProjection = inverse(viewProj);
    // prevViewProjection from the previous frame is stored in the globals buffer; the
    // CPU keeps the authoritative copy in prevViewProj_.
    globals.prevViewProjection = historyValid_ ? prevViewProjection_ : viewProj;
    prevViewProjection_ = viewProj;
    cameraPosition_ = camera.position();

    globals.cameraPosition = Vec4(camera.position(), 1.0f / camera.zFar());
    globals.viewportSize = Vec4(static_cast<float>(renderWidth_), static_cast<float>(renderHeight_),
                                1.0f / static_cast<float>(renderWidth_),
                                1.0f / static_cast<float>(renderHeight_));
    const scene::Environment& env = scene.environment();
    globals.envCeiling = Vec4(env.ceilingBounce, env.ambientIntensity);
    globals.envWall = Vec4(env.wallBounce, 0.0f);
    globals.envFloor = Vec4(env.floorBounce, 0.0f);
    globals.ambientTint = Vec4(env.ambientTint, 0.0f);
    globals.misc = Vec4(timeAccum_, static_cast<float>(frameIndex_), static_cast<float>(
                                                                           scene.lights().size()),
                        exposure);
    const Vec2 jitter = camera.jitter();
    globals.jitter = Vec4(jitter.x, jitter.y, 0.0f, 0.0f);
    globals.shadowParams = Vec4(camera.zNear(), camera.zFar(),
                                1.0f / static_cast<float>(config_.shadowResolution), 0.0028f);
    const float taaBlend = (config_.enableTAA && historyValid_) ? 0.12f : 0.0f;
    globals.postParams = Vec4(config_.enableSSAO ? 1.0f : 0.0f, config_.enableBloom ? 0.055f : 0.0f,
                              taaBlend, camera.fovY());
    globals.renderSize = Vec4(static_cast<float>(renderWidth_), static_cast<float>(renderHeight_),
                              1.0f / static_cast<float>(renderWidth_),
                              1.0f / static_cast<float>(renderHeight_));
    (void)dt;

    gfx::Buffer& buffer = globalsBuffers_[slot];
    if (buffer.mapped) buffer.write(&globals, sizeof(globals));
}

void Renderer::barrierForSampling(VkCommandBuffer cmd, const gfx::Image& image) {
    gfx::cmdImageBarrier(cmd, image.handle, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_NONE,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

void Renderer::cmdBeginRendering(VkCommandBuffer cmd,
                                 const std::vector<VkRenderingAttachmentInfo>& color,
                                 const VkRenderingAttachmentInfo* depth, VkExtent2D extent,
                                 const char* label) {
    VkRenderingInfo info{VK_STRUCTURE_TYPE_RENDERING_INFO};
    info.renderArea = {{0, 0}, extent};
    info.layerCount = 1;
    info.colorAttachmentCount = static_cast<uint32_t>(color.size());
    info.pColorAttachments = color.empty() ? nullptr : color.data();
    info.pDepthAttachment = depth;
    vkCmdBeginRendering(cmd, &info);
    if (label) device_->beginLabel(cmd, label);
}

void Renderer::cmdDispatch(VkCommandBuffer cmd, VkPipeline pipeline, VkPipelineLayout layout,
                           VkDescriptorSet set, uint32_t gx, uint32_t gy, uint32_t gz,
                           const void* push, uint32_t pushSize, const char* label) {
    if (label) device_->beginLabel(cmd, label, 0.3f, 0.5f, 0.9f);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    VkDescriptorSet sets[3] = {setGlobals_[frameIndex_ % config_.maxFramesInFlight], setTextures_,
                               set};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 3, sets, 0, nullptr);
    if (push && pushSize)
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_ALL, 0, pushSize, push);
    vkCmdDispatch(cmd, gx, gy, gz);
    if (label) device_->endLabel(cmd);
}

void Renderer::drawSceneGeometry(VkCommandBuffer cmd, const scene::Scene& scene, bool shadowPass) {
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer_.handle, &offset);
    vkCmdBindIndexBuffer(cmd, indexBuffer_.handle, 0, VK_INDEX_TYPE_UINT32);

    const Frustum* frustum = shadowPass ? nullptr : cullFrustum_;
    uint32_t draws = 0;
    uint32_t tris = 0;
    uint32_t skippedInvisible = 0;
    uint32_t skippedFrustum = 0;
    uint32_t lastVisible = 0;
    for (uint32_t i = 0; i < uploadedInstanceCount_; ++i) {
        const scene::Instance& inst = scene.instance(i);
        if (!inst.visible) { ++skippedInvisible; continue; }
        if (inst.mesh >= scene.meshCount()) continue;
        if (shadowPass && !inst.castsShadow) continue;
        if (frustum && !inst.alwaysDraw && !frustum->intersectsAabb(inst.worldBounds)) {
            ++skippedFrustum;
            if (config_.verboseStats && stats_.frameIndex == 10 && i < 24) {
                R2_INFO("  culled instance ", i, " mesh=", inst.mesh, " bounds=(",
                        inst.worldBounds.mn.x, ",", inst.worldBounds.mn.y, ",",
                        inst.worldBounds.mn.z, ")-(", inst.worldBounds.mx.x, ",",
                        inst.worldBounds.mx.y, ",", inst.worldBounds.mx.z, ")");
            }
            continue;
        }
        lastVisible = i;
        const scene::Mesh& mesh = scene.mesh(inst.mesh);
        for (const scene::MeshRange& range : mesh.ranges) {
            if (range.indexCount == 0) continue;
            DrawPush push;
            push.materialIndex = inst.materialOverride != scene::kInvalidIndex
                                     ? inst.materialOverride
                                     : range.material;
            push.flags = 0;
            if (!shadowPass) {
                vkCmdPushConstants(cmd, layoutGbuffer_, VK_SHADER_STAGE_ALL, 0, sizeof(DrawPush),
                                   &push);
            }
            vkCmdDrawIndexed(cmd, range.indexCount, 1, range.firstIndex, range.vertexOffset, i);
            ++draws;
            tris += range.indexCount / 3;
        }
    }
    if (!shadowPass) {
        stats_.drawCalls = draws;
        stats_.triangles = tris;
        if (config_.verboseStats && stats_.frameIndex < 400) {
            R2_INFO("draws=", draws, " tris=", tris, " invisible=", skippedInvisible,
                    " frustumCulled=", skippedFrustum, " lastVisibleInstance=", lastVisible);
        }
    } else {
        stats_.shadowDrawCalls = draws;
    }
}

void Renderer::drawGlassGeometry(VkCommandBuffer cmd, const scene::Scene& scene) {
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer_.handle, &offset);
    vkCmdBindIndexBuffer(cmd, indexBuffer_.handle, 0, VK_INDEX_TYPE_UINT32);

    // Collect transmissive instances and sort back to front.
    struct Item {
        uint32_t instance;
        float distance;
    };
    std::vector<Item> items;
    for (uint32_t i = 0; i < uploadedInstanceCount_; ++i) {
        const scene::Instance& inst = scene.instance(i);
        if (!inst.visible || inst.mesh >= scene.meshCount()) continue;
        const scene::Mesh& mesh = scene.mesh(inst.mesh);
        bool transmissive = false;
        for (const auto& r : mesh.ranges) {
            const uint32_t mat = inst.materialOverride != scene::kInvalidIndex
                                     ? inst.materialOverride
                                     : r.material;
            if (mat < scene.materialCount() && scene.material(mat).isTransmissive()) {
                transmissive = true;
                break;
            }
        }
        if (!transmissive) continue;
        items.push_back({i, distance(cameraPosition_, inst.worldBounds.center())});
    }
    std::sort(items.begin(), items.end(),
              [](const Item& a, const Item& b) { return a.distance > b.distance; });

    for (const Item& item : items) {
        const scene::Instance& inst = scene.instance(item.instance);
        const scene::Mesh& mesh = scene.mesh(inst.mesh);
        for (const auto& range : mesh.ranges) {
            if (range.indexCount == 0) continue;
            DrawPush push;
            push.materialIndex = inst.materialOverride != scene::kInvalidIndex
                                     ? inst.materialOverride
                                     : range.material;
            push.flags = 1;
            vkCmdPushConstants(cmd, layoutGlass_, VK_SHADER_STAGE_ALL, 0, sizeof(DrawPush), &push);
            vkCmdDrawIndexed(cmd, range.indexCount, 1, range.firstIndex, range.vertexOffset,
                             item.instance);
        }
    }
}


// ---------------------------------------------------------------- render
namespace {
// Barrier helper for images that permanently live in VK_IMAGE_LAYOUT_GENERAL.
void generalSync(VkCommandBuffer cmd, VkImage image, VkPipelineStageFlags2 srcStage,
                 VkPipelineStageFlags2 dstStage, VkAccessFlags2 srcAccess,
                 VkAccessFlags2 dstAccess) {
    gfx::cmdImageBarrier(cmd, image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, srcStage,
                         dstStage, srcAccess, dstAccess);
}
}  // namespace

void Renderer::render(const scene::Scene& scene, const Camera& camera, float dt,
                      const UiDrawList* ui, float exposure) {
    if (!frameOpen_) return;
    const auto cpuStart = std::chrono::high_resolution_clock::now();
    const uint32_t slot = frameIndex_ % config_.maxFramesInFlight;
    VkCommandBuffer cmd = commandBuffers_[slot];
    cullFrustum_ = &camera.frustum();
    timeAccum_ += dt;

    updateMaterials(scene);
    updateLights(scene);
    updateInstances(scene);
    writeGlobals(scene, camera, dt, exposure);

    const VkExtent2D renderExtent{renderWidth_, renderHeight_};
    const VkPipelineStageFlags2 kCompute = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    const VkPipelineStageFlags2 kFragment = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    const VkAccessFlags2 kShaderRead = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    const VkAccessFlags2 kShaderWrite = VK_ACCESS_2_SHADER_WRITE_BIT;

    // ---- 1. shadow atlas -------------------------------------------------
    if (config_.enableShadows) {
        device_->beginLabel(cmd, "shadow pass", 0.2f, 0.2f, 0.35f);
        const uint32_t layerCount = kShadowCubeFaces * kMaxShadowLights;
        gfx::cmdImageBarrier(cmd, shadowAtlas_.handle, VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                             VK_ACCESS_2_NONE, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                             VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, layerCount);

        uint32_t shadowIndex = 0;
        const uint32_t shadowRes = static_cast<uint32_t>(config_.shadowResolution);
        for (uint32_t li = 0; li < scene.lights().size() && shadowIndex < kMaxShadowLights; ++li) {
            const scene::Light& light = scene.lights()[li];
            if (!light.castsShadow) continue;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineShadow_);
            VkDescriptorSet sets[3] = {setGlobals_[slot], setTextures_, setShadow_};
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layoutShadow_, 0, 3, sets,
                                    0, nullptr);
            VkRenderingAttachmentInfo depthAtt = depthAttachment(
                shadowFaceViews_[shadowIndex], VK_IMAGE_LAYOUT_GENERAL,
                VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, 1.0f);
            cmdBeginRendering(cmd, {}, &depthAtt, {shadowRes, shadowRes}, "shadow pass");
            VkViewport viewport{0.0f, 0.0f, static_cast<float>(shadowRes),
                                static_cast<float>(shadowRes), 0.0f, 1.0f};
            VkRect2D scissor{{0, 0}, {shadowRes, shadowRes}};
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            vkCmdSetScissor(cmd, 0, 1, &scissor);
            ShadowPush push{0, li};
            vkCmdPushConstants(cmd, layoutShadow_, VK_SHADER_STAGE_ALL, 0, sizeof(push), &push);
            drawSceneGeometry(cmd, scene, true);
            vkCmdEndRendering(cmd);
            ++shadowIndex;
        }
        gfx::cmdImageBarrier(cmd, shadowAtlas_.handle, VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                             kCompute | kFragment,
                             VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, kShaderRead,
                             VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, kMaxShadowLights);
        device_->endLabel(cmd);
    }

    // ---- 2. G-buffer -----------------------------------------------------
    {
        device_->beginLabel(cmd, "gbuffer pass", 0.2f, 0.35f, 0.2f);
        const VkClearColorValue zero{{0.0f, 0.0f, 0.0f, 1.0f}};
        const VkClearColorValue normalClear{{0.5f, 0.5f, 0.5f, 1.0f}};
        std::vector<VkRenderingAttachmentInfo> colors = {
            colorAttachment(gbufferAlbedo_.view, VK_IMAGE_LAYOUT_GENERAL,
                            VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, zero),
            colorAttachment(gbufferNormal_.view, VK_IMAGE_LAYOUT_GENERAL,
                            VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, normalClear),
            colorAttachment(gbufferMetal_.view, VK_IMAGE_LAYOUT_GENERAL,
                            VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, zero),
            colorAttachment(gbufferEmissive_.view, VK_IMAGE_LAYOUT_GENERAL,
                            VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, zero),
        };
        VkRenderingAttachmentInfo depthAtt = depthAttachment(
            depth_.view, VK_IMAGE_LAYOUT_GENERAL, VK_ATTACHMENT_LOAD_OP_CLEAR,
            VK_ATTACHMENT_STORE_OP_STORE, 1.0f);
        cmdBeginRendering(cmd, colors, &depthAtt, renderExtent, nullptr);
        VkViewport viewport{0.0f, 0.0f, static_cast<float>(renderWidth_),
                            static_cast<float>(renderHeight_), 0.0f, 1.0f};
        VkRect2D scissor{{0, 0}, {renderWidth_, renderHeight_}};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineGbuffer_);
        VkDescriptorSet sets[3] = {setGlobals_[slot], setTextures_, setShadow_};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layoutGbuffer_, 0, 3, sets, 0,
                                nullptr);
        drawSceneGeometry(cmd, scene, false);
        vkCmdEndRendering(cmd);
        device_->endLabel(cmd);
    }
    for (const gfx::Image* image : {&gbufferAlbedo_, &gbufferNormal_, &gbufferMetal_,
                                    &gbufferEmissive_}) {
        gfx::cmdImageBarrier(cmd, image->handle, VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, kCompute,
                             VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, kShaderRead);
    }
    gfx::cmdImageBarrier(cmd, depth_.handle, VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, kCompute | kFragment,
                         VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, kShaderRead,
                         VK_IMAGE_ASPECT_DEPTH_BIT);

    // ---- 3. SSAO ---------------------------------------------------------
    const uint32_t aoW = std::max(1u, renderWidth_ / 2), aoH = std::max(1u, renderHeight_ / 2);
    if (config_.enableSSAO) {
        cmdDispatch(cmd, pipelineSsao_, layoutSsao_, setSsao_, (aoW + 7) / 8, (aoH + 7) / 8, 1,
                    nullptr, 0, "ssao");
        generalSync(cmd, aoRaw_.handle, kCompute, kCompute, kShaderWrite, kShaderRead);
        cmdDispatch(cmd, pipelineSsaoBlur_, layoutSsaoBlur_, setSsaoBlur_, (aoW + 7) / 8,
                    (aoH + 7) / 8, 1, nullptr, 0, "ssao blur");
        generalSync(cmd, aoBlur_.handle, kCompute, kCompute, kShaderWrite, kShaderRead);
    } else {
        gfx::cmdImageBarrier(cmd, aoBlur_.handle, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_NONE,
                             VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkClearColorValue white{{1.0f, 1.0f, 1.0f, 1.0f}};
        VkImageSubresourceRange range = vk::colorRange();
        vkCmdClearColorImage(cmd, aoBlur_.handle, VK_IMAGE_LAYOUT_GENERAL, &white, 1, &range);
        generalSync(cmd, aoBlur_.handle, VK_PIPELINE_STAGE_2_CLEAR_BIT, kCompute,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT, kShaderRead);
    }

    // ---- 4. deferred lighting -------------------------------------------
    gfx::cmdImageBarrier(cmd, hdrColor_.handle, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, kCompute,
                         VK_ACCESS_2_NONE, kShaderWrite);
    cmdDispatch(cmd, pipelineLighting_, layoutLighting_, setLighting_, (renderWidth_ + 7) / 8,
                (renderHeight_ + 7) / 8, 1, nullptr, 0, "lighting");
    generalSync(cmd, hdrColor_.handle, kCompute, VK_PIPELINE_STAGE_2_COPY_BIT, kShaderWrite,
                VK_ACCESS_2_TRANSFER_READ_BIT);

    // ---- 5. copy the opaque result for screen-space refraction -----------
    gfx::cmdImageBarrier(cmd, hdrOpaqueCopy_.handle, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                         VK_ACCESS_2_NONE, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    {
        VkImageCopy region{};
        region.srcSubresource = vk::colorLayers();
        region.dstSubresource = vk::colorLayers();
        region.extent = {renderWidth_, renderHeight_, 1};
        vkCmdCopyImage(cmd, hdrColor_.handle, VK_IMAGE_LAYOUT_GENERAL, hdrOpaqueCopy_.handle,
                       VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    }
    gfx::cmdImageBarrier(cmd, hdrOpaqueCopy_.handle, VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COPY_BIT,
                         kFragment, VK_ACCESS_2_TRANSFER_WRITE_BIT, kShaderRead);

    // ---- 6. glass (forward, blended) -------------------------------------
    if (config_.enableGlass) {
        gfx::cmdImageBarrier(cmd, hdrColor_.handle, VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_GENERAL, kCompute,
                             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, kShaderWrite,
                             VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT);
        gfx::cmdImageBarrier(cmd, depth_.handle, VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_GENERAL, kCompute,
                             VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, kShaderRead,
                             VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                             VK_IMAGE_ASPECT_DEPTH_BIT);
        const VkClearColorValue zero{{0, 0, 0, 1}};
        std::vector<VkRenderingAttachmentInfo> colors = {colorAttachment(
            hdrColor_.view, VK_IMAGE_LAYOUT_GENERAL, VK_ATTACHMENT_LOAD_OP_LOAD,
            VK_ATTACHMENT_STORE_OP_STORE, zero)};
        VkRenderingAttachmentInfo depthAtt =
            depthAttachment(depth_.view, VK_IMAGE_LAYOUT_GENERAL,
                            VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
        cmdBeginRendering(cmd, colors, &depthAtt, renderExtent, "glass pass");
        VkViewport viewport{0.0f, 0.0f, static_cast<float>(renderWidth_),
                            static_cast<float>(renderHeight_), 0.0f, 1.0f};
        VkRect2D scissor{{0, 0}, {renderWidth_, renderHeight_}};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineGlass_);
        VkDescriptorSet sets[3] = {setGlobals_[slot], setTextures_, setGlass_};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layoutGlass_, 0, 3, sets, 0,
                                nullptr);
        drawGlassGeometry(cmd, scene);
        vkCmdEndRendering(cmd);
        device_->endLabel(cmd);
        gfx::cmdImageBarrier(cmd, depth_.handle, VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, kCompute | kFragment,
                             VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, kShaderRead,
                             VK_IMAGE_ASPECT_DEPTH_BIT);
        gfx::cmdImageBarrier(cmd, hdrColor_.handle, VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, kCompute,
                             VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, kShaderWrite | kShaderRead);
    }

    // ---- 7. TAA resolve --------------------------------------------------
    const gfx::Image* resolved = &hdrColor_;
    if (config_.enableTAA) {
        const uint32_t write = historyIndex_;
        if (historyValid_) {
            generalSync(cmd, hdrHistory_[write].handle, kCompute, kCompute, kShaderWrite,
                        kShaderRead);
        }
        cmdDispatch(cmd, pipelineTaa_, layoutTaa_, setTaa_[historyIndex_], (renderWidth_ + 7) / 8,
                    (renderHeight_ + 7) / 8, 1, nullptr, 0, "taa");
        // setTaa_[i] reads history[i] and writes history[1-i]: the resolved frame is
        // the image we just wrote, and next frame reads it back as history.
        const gfx::Image& taaOutput = hdrHistory_[1 - write];
        generalSync(cmd, taaOutput.handle, kCompute, kCompute | kFragment, kShaderWrite,
                    kShaderRead);
        resolved = &taaOutput;
        historyIndex_ = 1 - historyIndex_;
        historyValid_ = true;
    } else {
        historyValid_ = false;
    }

    // ---- 8. bloom --------------------------------------------------------
    if (config_.enableBloom && bloomMipCount_ > 0) {
        VkDescriptorImageInfo info0{samplerLinearClamp_, resolved->view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet writes[2];
        gfx::writeImage(writes[0], setBloomPrefilter_[slot], 0,
                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &info0);
        gfx::writeImage(writes[1], setTonemap_[slot], 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        &info0);
        vkUpdateDescriptorSets(device_->device(), 2, writes, 0, nullptr);

        const uint32_t w0 = bloomMips_[0].extent.width, h0 = bloomMips_[0].extent.height;
        cmdDispatch(cmd, pipelineBloomPrefilter_, layoutBloomPrefilter_, setBloomPrefilter_[slot],
                    (w0 + 7) / 8, (h0 + 7) / 8, 1, nullptr, 0, "bloom prefilter");
        for (uint32_t i = 0; i + 1 < bloomMipCount_; ++i) {
            generalSync(cmd, bloomMips_[i].handle, kCompute, kCompute, kShaderWrite, kShaderRead);
            const uint32_t w = bloomMips_[i + 1].extent.width;
            const uint32_t h = bloomMips_[i + 1].extent.height;
            cmdDispatch(cmd, pipelineBloomDown_, layoutBloomDown_, setBloomDown_[i], (w + 7) / 8,
                        (h + 7) / 8, 1, nullptr, 0, "bloom down");
            generalSync(cmd, bloomMips_[i + 1].handle, kCompute, kCompute, kShaderWrite,
                        kShaderRead);
        }
        for (uint32_t i = bloomMipCount_ - 1; i > 0; --i) {
            const uint32_t w = bloomMips_[i - 1].extent.width;
            const uint32_t h = bloomMips_[i - 1].extent.height;
            cmdDispatch(cmd, pipelineBloomUp_, layoutBloomUp_, setBloomUp_[i - 1], (w + 7) / 8,
                        (h + 7) / 8, 1, nullptr, 0, "bloom up");
            generalSync(cmd, bloomMips_[i - 1].handle, kCompute, kCompute, kShaderWrite,
                        kShaderRead);
        }
    }

    // ---- 9. tonemap into the presentation target -------------------------
    VkImage targetImage = VK_NULL_HANDLE;
    VkImageView targetView = VK_NULL_HANDLE;
    bool headlessTarget = headless_;
    const VkExtent2D outExtent{headlessTarget ? outputWidth_ : swapchain_->extent().width,
                               headlessTarget ? outputHeight_ : swapchain_->extent().height};
    if (headlessTarget) {
        targetImage = ldrColor_.handle;
        targetView = ldrColor_.view;
    } else {
        targetImage = swapchain_->image(swapchainImageIndex_);
        targetView = swapchain_->view(swapchainImageIndex_);
    }
    const VkImageLayout targetLayout =
        headlessTarget ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    gfx::cmdImageBarrier(cmd, targetImage, VK_IMAGE_LAYOUT_UNDEFINED, targetLayout,
                         VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_NONE,
                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    {
        const VkClearColorValue black{{0, 0, 0, 1}};
        std::vector<VkRenderingAttachmentInfo> colors = {colorAttachment(
            targetView, headlessTarget ? VK_IMAGE_LAYOUT_GENERAL
                                       : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, black)};
        cmdBeginRendering(cmd, colors, nullptr, outExtent, "tonemap pass");
        VkViewport viewport{0.0f, 0.0f, static_cast<float>(outExtent.width),
                            static_cast<float>(outExtent.height), 0.0f, 1.0f};
        VkRect2D scissor{{0, 0}, {outExtent.width, outExtent.height}};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineTonemap_);
        VkDescriptorSet sets[3] = {setGlobals_[slot], setTextures_, setTonemap_[slot]};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layoutTonemap_, 0, 3, sets,
                                0, nullptr);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);
        device_->endLabel(cmd);
    }

    // ---- 10. UI overlay --------------------------------------------------
    if (config_.enableUi && ui && !ui->empty()) {
        const size_t vertexBytes = ui->vertices.size() * sizeof(UiVertex);
        const size_t indexBytes = ui->indices.size() * sizeof(uint32_t);
        if (!uiVertexBuffer_ || vertexBytes > uiVertexCapacity_) {
            gfx::destroyBuffer(*device_, uiVertexBuffer_);
            gfx::BufferDesc desc;
            desc.size = std::max<size_t>(vertexBytes * 2, 64 * 1024);
            desc.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            desc.memory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            desc.name = "ui vertices";
            uiVertexBuffer_ = gfx::createBuffer(*device_, desc);
            uiVertexCapacity_ = desc.size;
        }
        if (!uiIndexBuffer_ || indexBytes > uiIndexCapacity_) {
            gfx::destroyBuffer(*device_, uiIndexBuffer_);
            gfx::BufferDesc desc;
            desc.size = std::max<size_t>(indexBytes * 2, 64 * 1024);
            desc.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
            desc.memory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            desc.name = "ui indices";
            uiIndexBuffer_ = gfx::createBuffer(*device_, desc);
            uiIndexCapacity_ = desc.size;
        }
        uiVertexBuffer_.write(ui->vertices.data(), vertexBytes);
        uiIndexBuffer_.write(ui->indices.data(), indexBytes);
        stats_.drawCalls += static_cast<uint32_t>(ui->batches.size());

        gfx::cmdImageBarrier(cmd, targetImage, targetLayout, targetLayout,
                             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                             VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        const VkClearColorValue black{{0, 0, 0, 1}};
        std::vector<VkRenderingAttachmentInfo> colors = {colorAttachment(
            targetView, headlessTarget ? VK_IMAGE_LAYOUT_GENERAL
                                       : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, black)};
        cmdBeginRendering(cmd, colors, nullptr, outExtent, "ui pass");
        VkViewport viewport{0.0f, 0.0f, static_cast<float>(outExtent.width),
                            static_cast<float>(outExtent.height), 0.0f, 1.0f};
        VkRect2D scissor{{0, 0}, {outExtent.width, outExtent.height}};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineUi_);
        VkDescriptorSet sets[3] = {setGlobals_[slot], setTextures_, setUi_};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layoutUi_, 0, 3, sets, 0,
                                nullptr);
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &uiVertexBuffer_.handle, &offset);
        vkCmdBindIndexBuffer(cmd, uiIndexBuffer_.handle, 0, VK_INDEX_TYPE_UINT32);
        for (const auto& batch : ui->batches) {
            UiPush push;
            push.screenSize = Vec2(static_cast<float>(outExtent.width),
                                   static_cast<float>(outExtent.height));
            push.useTexture = batch.useTexture ? 1.0f : 0.0f;
            push.opacity = 1.0f;
            vkCmdPushConstants(cmd, layoutUi_, VK_SHADER_STAGE_ALL, 0, sizeof(push), &push);
            vkCmdDrawIndexed(cmd, batch.indexCount, 1, batch.firstIndex, 0, 0);
        }
        vkCmdEndRendering(cmd);
        device_->endLabel(cmd);
    }

    if (headlessTarget) {
        gfx::cmdImageBarrier(cmd, ldrColor_.handle, VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_2_COPY_BIT,
                             VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                             VK_ACCESS_2_TRANSFER_READ_BIT);
        capturePending_ = true;
    } else {
        // Optional readback of the presented image (used by --windowed-shot so the
        // windowed/swapchain path can be verified without a human looking at a window).
        if (capturePending_) {
            const size_t needed = static_cast<size_t>(outputWidth_) * outputHeight_ * 4;
            if (!readbackBuffer_ || needed > readbackCapacity_) {
                gfx::destroyBuffer(*device_, readbackBuffer_);
                gfx::BufferDesc desc;
                desc.size = needed;
                desc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                desc.memory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                desc.name = "readback";
                readbackBuffer_ = gfx::createBuffer(*device_, desc);
                readbackCapacity_ = needed;
            }
            gfx::cmdImageBarrier(cmd, targetImage, targetLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 VK_PIPELINE_STAGE_2_COPY_BIT,
                                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                 VK_ACCESS_2_TRANSFER_READ_BIT);
            VkBufferImageCopy region{};
            region.imageSubresource = vk::colorLayers();
            region.imageExtent = {outExtent.width, outExtent.height, 1};
            vkCmdCopyImageToBuffer(cmd, targetImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   readbackBuffer_.handle, 1, &region);
            gfx::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT,
                                  VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                  VK_ACCESS_2_HOST_READ_BIT);
            gfx::cmdImageBarrier(cmd, targetImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                 VK_PIPELINE_STAGE_2_COPY_BIT,
                                 VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                                 VK_ACCESS_2_TRANSFER_READ_BIT, VK_ACCESS_2_NONE);
            capturePending_ = false;
        } else {
            gfx::cmdImageBarrier(cmd, targetImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_2_NONE);
        }
    }

    if (config_.verboseStats) {
        R2_INFO("frame ", frameIndex_, ": gbuffer draws=", stats_.drawCalls,
                " tris=", stats_.triangles, " shadow draws=", stats_.shadowDrawCalls,
                " instances=", stats_.instances);
    }
    const auto cpuEnd = std::chrono::high_resolution_clock::now();
    stats_.cpuFrameMs =
        std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
    stats_.instances = uploadedInstanceCount_;
    stats_.frameIndex = frameIndex_;
}

void Renderer::endFrame() {
    if (!frameOpen_) return;
    const uint32_t slot = frameIndex_ % config_.maxFramesInFlight;
    VkCommandBuffer cmd = commandBuffers_[slot];
    if (capturePending_) {
        const size_t needed = static_cast<size_t>(outputWidth_) * outputHeight_ * 4;
        if (!readbackBuffer_ || needed > readbackCapacity_) {
            gfx::destroyBuffer(*device_, readbackBuffer_);
            gfx::BufferDesc desc;
            desc.size = needed;
            desc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            desc.memory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            desc.name = "readback";
            readbackBuffer_ = gfx::createBuffer(*device_, desc);
            readbackCapacity_ = needed;
        }
        VkBufferImageCopy region{};
        region.imageSubresource = vk::colorLayers();
        region.imageExtent = {outputWidth_, outputHeight_, 1};
        vkCmdCopyImageToBuffer(cmd, ldrColor_.handle, VK_IMAGE_LAYOUT_GENERAL,
                               readbackBuffer_.handle, 1, &region);
        gfx::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT,
                              VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                              VK_ACCESS_2_HOST_READ_BIT);
    }
    VK_CHECK(vkEndCommandBuffer(cmd));
    frameOpen_ = false;

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    if (!headless_) {
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &imageAvailable_[slot];
        submit.pWaitDstStageMask = &waitStage;
    }
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    // The semaphore waited on by vkQueuePresentKHR belongs to the swapchain and is
    // indexed by the acquired image; signalling our own semaphore here would leave the
    // present waiting forever on an unsignalled one.
    VkSemaphore presentWait = VK_NULL_HANDLE;
    if (!headless_) {
        presentWait = swapchain_->renderFinishedSemaphore(swapchainImageIndex_);
        if (presentWait == VK_NULL_HANDLE) {
            R2_ERROR("no present semaphore for swapchain image ", swapchainImageIndex_);
            return;
        }
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &presentWait;
    }
    VK_CHECK(vkQueueSubmit(device_->graphicsQueue().handle, 1, &submit, frameFences_[slot]));

    if (!headless_) {
        if (!swapchain_->present(slot, swapchainImageIndex_)) {
            R2_DEBUG("swapchain present failed; recreation scheduled");
        }
        // Incremented after a successful present so the next frame waits on fresh state.
        ++frameIndex_;
        return;
    }
    VK_CHECK(vkWaitForFences(device_->device(), 1, &frameFences_[slot], VK_TRUE, ~0ull));
    ++frameIndex_;
}

bool Renderer::renderOffscreen(const scene::Scene& scene, const Camera& camera, float dt,
                               const UiDrawList* ui, float exposure, int iterations,
                               std::vector<uint8_t>& rgbaOut, uint32_t& widthOut,
                               uint32_t& heightOut) {
    if (!headless_) {
        R2_ERROR("renderOffscreen requires the renderer to be initialised in headless mode");
        return false;
    }
    for (int i = 0; i < std::max(1, iterations); ++i) {
        if (!beginFrame()) return false;
        Camera jittered = camera;
        const Vec2 jitter = haltonJitter(jitterIndex_++, renderWidth_, renderHeight_);
        jittered.setJitter(jitter);
        jittered.setViewportSize(renderWidth_, renderHeight_);
        render(scene, jittered, dt, ui, exposure);
        // The last iteration is the one we keep.
        pendingCapture_ = (i == std::max(1, iterations) - 1);
        capturePending_ = pendingCapture_;
        endFrame();
    }
    if (!readbackBuffer_.mapped) return false;
    rgbaOut.assign(static_cast<size_t>(outputWidth_) * outputHeight_ * 4, 0);
    std::memcpy(rgbaOut.data(), readbackBuffer_.mapped, rgbaOut.size());
    widthOut = outputWidth_;
    heightOut = outputHeight_;
    return true;
}

bool Renderer::capture(std::vector<uint8_t>& rgbaOut, uint32_t& widthOut, uint32_t& heightOut) {
    if (!readbackBuffer_ || !readbackBuffer_.mapped) return false;
    rgbaOut.assign(static_cast<size_t>(outputWidth_) * outputHeight_ * 4, 0);
    std::memcpy(rgbaOut.data(), readbackBuffer_.mapped, rgbaOut.size());
    // A B8G8R8A8 swapchain stores blue first; the image is written in logical RGBA order
    // but read back as raw bytes, so swap R and B to produce a true RGBA image.
    if (swapchain_ != nullptr) {
        const VkFormat f = swapchain_->format();
        if (f == VK_FORMAT_B8G8R8A8_UNORM || f == VK_FORMAT_B8G8R8A8_SRGB) {
            for (size_t i = 0; i + 3 < rgbaOut.size(); i += 4) std::swap(rgbaOut[i], rgbaOut[i + 2]);
        }
    }
    widthOut = outputWidth_;
    heightOut = outputHeight_;
    return true;
}

void Renderer::setConfigFlag(const std::string& name, bool value) {
    if (name == "ssao") config_.enableSSAO = value;
    else if (name == "bloom") config_.enableBloom = value;
    else if (name == "taa") config_.enableTAA = value;
    else if (name == "glass") config_.enableGlass = value;
    else if (name == "shadows") config_.enableShadows = value;
    else if (name == "ui") config_.enableUi = value;
    else R2_WARN("unknown renderer flag '", name, "'");
    if (name == "ssao" || name == "bloom" || name == "taa") historyValid_ = false;
}

bool Renderer::configFlag(const std::string& name) const {
    if (name == "ssao") return config_.enableSSAO;
    if (name == "bloom") return config_.enableBloom;
    if (name == "taa") return config_.enableTAA;
    if (name == "glass") return config_.enableGlass;
    if (name == "shadows") return config_.enableShadows;
    if (name == "ui") return config_.enableUi;
    return false;
}

std::vector<Renderer::DebugTarget> Renderer::debugReadTargets() {
    struct Entry {
        const char* name;
        const gfx::Image* image;
        uint32_t layer = 0;
    };
    std::vector<Entry> entries = {
        {"gbuffer_albedo", &gbufferAlbedo_}, {"gbuffer_normal", &gbufferNormal_},
        {"gbuffer_metal", &gbufferMetal_},   {"gbuffer_emissive", &gbufferEmissive_},
        {"depth", &depth_},                  {"hdr_color", &hdrColor_},
        {"hdr_history0", &hdrHistory_[0]},   {"hdr_history1", &hdrHistory_[1]},
        {"ao_blur", &aoBlur_},               {"env_probe", &envProbe_},
        {"shadow_f0", &shadowAtlas_, 0}, {"shadow_f1", &shadowAtlas_, 1},
        {"shadow_f2", &shadowAtlas_, 2}, {"shadow_f3", &shadowAtlas_, 3},
        {"shadow_f4", &shadowAtlas_, 4}, {"shadow_f5", &shadowAtlas_, 5},
        {"ldr_color", &ldrColor_},
    };
    for (uint32_t i = 0; i < bloomMipCount_ && i < 2; ++i) {
        static const char* names[2] = {"bloom0", "bloom1"};
        entries.push_back({names[i], &bloomMips_[i]});
    }

    std::vector<DebugTarget> out;
    for (const auto& entry : entries) {
        const gfx::Image& image = *entry.image;
        if (!image.handle) continue;
        const uint32_t layers = 1;
        const uint32_t baseLayer = std::min(entry.layer, image.arrayLayers - 1);
        const uint32_t texelSize = vk::formatTexelSize(image.format);
        const size_t byteSize = static_cast<size_t>(image.extent.width) * image.extent.height *
                                texelSize * layers;
        if (byteSize == 0) continue;

        gfx::Buffer readback;
        {
            gfx::BufferDesc desc;
            desc.size = byteSize;
            desc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            desc.memory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            desc.name = "debug readback";
            readback = gfx::createBuffer(*device_, desc);
        }
        VkCommandBuffer cmd = device_->beginImmediate("debug readback");
        VkBufferImageCopy region{};
        region.imageSubresource = {vk::isDepthFormat(image.format) ? VK_IMAGE_ASPECT_DEPTH_BIT
                                                                   : VK_IMAGE_ASPECT_COLOR_BIT,
                                   0, baseLayer, layers};
        region.imageExtent = {image.extent.width, image.extent.height, 1};
        vkCmdCopyImageToBuffer(cmd, image.handle, VK_IMAGE_LAYOUT_GENERAL, readback.handle, 1,
                               &region);
        gfx::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                              VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_HOST_READ_BIT);
        device_->endImmediate(cmd, true);

        DebugTarget target;
        target.name = entry.name;
        target.width = image.extent.width;
        target.height = image.extent.height;
        target.rgba.assign(static_cast<size_t>(target.width) * target.height * 4, 0.0f);
        const uint8_t* src = static_cast<const uint8_t*>(readback.mapped);
        float minV = 1e30f, maxV = -1e30f;
        double sum = 0.0;
        const size_t texels = static_cast<size_t>(target.width) * target.height;
        for (size_t i = 0; i < texels; ++i) {
            float r = 0, g = 0, b = 0, a = 1;
            switch (image.format) {
                case VK_FORMAT_R8G8B8A8_UNORM: {
                    const uint8_t* p = src + i * 4;
                    r = p[0] / 255.0f; g = p[1] / 255.0f; b = p[2] / 255.0f; a = p[3] / 255.0f;
                    break;
                }
                case VK_FORMAT_R16G16B16A16_SFLOAT: {
                    const uint16_t* p = reinterpret_cast<const uint16_t*>(src + i * 8);
                    r = halfToFloat(p[0]); g = halfToFloat(p[1]);
                    b = halfToFloat(p[2]); a = halfToFloat(p[3]);
                    break;
                }
                case VK_FORMAT_R16_SFLOAT: {
                    r = g = b = halfToFloat(reinterpret_cast<const uint16_t*>(src)[i]);
                    break;
                }
                case VK_FORMAT_D32_SFLOAT: {
                    r = g = b = reinterpret_cast<const float*>(src)[i];
                    break;
                }
                default: {
                    const uint8_t* p = src + i * texelSize;
                    r = g = b = p[0] / 255.0f;
                    break;
                }
            }
            target.rgba[i * 4 + 0] = r;
            target.rgba[i * 4 + 1] = g;
            target.rgba[i * 4 + 2] = b;
            target.rgba[i * 4 + 3] = a;
            const float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            minV = std::min(minV, luma);
            maxV = std::max(maxV, luma);
            sum += luma;
        }
        target.minValue = minV;
        target.maxValue = maxV;
        target.meanLuma = static_cast<float>(sum / static_cast<double>(texels));
        out.push_back(std::move(target));
        gfx::destroyBuffer(*device_, readback);
    }
    return out;
}

void Renderer::shutdown() {
    if (!device_) return;
    device_->waitIdle();
    gfx::Device& d = *device_;

    for (VkImageView v : envProbeMipViews_)
        if (v) vkDestroyImageView(d.device(), v, nullptr);
    envProbeMipViews_.clear();
    for (VkImageView v : shadowFaceViews_)
        if (v) vkDestroyImageView(d.device(), v, nullptr);
    shadowFaceViews_.clear();

    pipelineCache_.save();
    pipelineCache_.shutdown();

    auto destroyPipeline = [&](VkPipeline& p) {
        if (p) vkDestroyPipeline(d.device(), p, nullptr);
        p = VK_NULL_HANDLE;
    };
    auto destroyLayout = [&](VkPipelineLayout& l) {
        if (l) vkDestroyPipelineLayout(d.device(), l, nullptr);
        l = VK_NULL_HANDLE;
    };
    auto destroySetLayout = [&](VkDescriptorSetLayout& l) {
        if (l) vkDestroyDescriptorSetLayout(d.device(), l, nullptr);
        l = VK_NULL_HANDLE;
    };
    auto destroyShader = [&](VkShaderModule& m) {
        if (m) vkDestroyShaderModule(d.device(), m, nullptr);
        m = VK_NULL_HANDLE;
    };

    destroyPipeline(pipelineShadow_);
    destroyPipeline(pipelineGbuffer_);
    destroyPipeline(pipelineGlass_);
    destroyPipeline(pipelineLighting_);
    destroyPipeline(pipelineSsao_);
    destroyPipeline(pipelineSsaoBlur_);
    destroyPipeline(pipelineBloomPrefilter_);
    destroyPipeline(pipelineBloomDown_);
    destroyPipeline(pipelineBloomUp_);
    destroyPipeline(pipelineTonemap_);
    destroyPipeline(pipelineUi_);
    destroyPipeline(pipelineProbe_);
    destroyPipeline(pipelineTaa_);

    destroyLayout(layoutShadow_);
    destroyLayout(layoutGbuffer_);
    destroyLayout(layoutGlass_);
    destroyLayout(layoutLighting_);
    destroyLayout(layoutSsao_);
    destroyLayout(layoutSsaoBlur_);
    destroyLayout(layoutBloomPrefilter_);
    destroyLayout(layoutBloomDown_);
    destroyLayout(layoutBloomUp_);
    destroyLayout(layoutTonemap_);
    destroyLayout(layoutUi_);
    destroyLayout(layoutProbe_);
    destroyLayout(layoutTaa_);

    for (VkShaderModule* m : {&vsFullscreen_, &vsGbuffer_, &fsGbuffer_, &vsShadow_, &fsShadow_,
                              &vsGlass_, &fsGlass_, &csLighting_, &csSsao_, &csSsaoBlur_,
                              &csBloomPrefilter_, &csBloomDown_, &csBloomUp_, &fsTonemap_, &vsUi_,
                              &fsUi_, &csProbe_, &csTaa_})
        destroyShader(*m);

    mainPool_.shutdown();
    bindlessPool_.shutdown();
    probePool_.shutdown();
    framePool_.shutdown();

    destroySetLayout(setLayoutGlobals_);
    destroySetLayout(setLayoutTextures_);
    destroySetLayout(setLayoutEmpty_);
    destroySetLayout(setLayoutLighting_);
    destroySetLayout(setLayoutSsao_);
    destroySetLayout(setLayoutSsaoBlur_);
    destroySetLayout(setLayoutBloomPrefilter_);
    destroySetLayout(setLayoutBloomDown_);
    destroySetLayout(setLayoutBloomUp_);
    destroySetLayout(setLayoutGlass_);
    destroySetLayout(setLayoutTonemap_);
    destroySetLayout(setLayoutUi_);
    destroySetLayout(setLayoutProbe_);
    destroySetLayout(setLayoutTaa_);

    destroyRenderTargets();
    gfx::destroyImage(d, envProbe_);
    gfx::destroyImage(d, shadowAtlas_);
    gfx::destroyImage(d, fontAtlas_);
    gfx::destroyImage(d, offscreenColor_);

    gfx::destroyBuffer(d, vertexBuffer_);
    gfx::destroyBuffer(d, indexBuffer_);
    gfx::destroyBuffer(d, materialBuffer_);
    gfx::destroyBuffer(d, lightBuffer_);
    gfx::destroyBuffer(d, uiVertexBuffer_);
    gfx::destroyBuffer(d, uiIndexBuffer_);
    gfx::destroyBuffer(d, readbackBuffer_);
    for (auto& b : instanceBuffers_) gfx::destroyBuffer(d, b);
    for (auto& b : globalsBuffers_) gfx::destroyBuffer(d, b);
    instanceBuffers_.clear();
    globalsBuffers_.clear();
    for (auto& image : textures_) gfx::destroyImage(d, image);
    textures_.clear();

    if (samplerShadow_) vkDestroySampler(d.device(), samplerShadow_, nullptr);
    if (samplerLinearRepeat_) vkDestroySampler(d.device(), samplerLinearRepeat_, nullptr);
    if (samplerLinearClamp_) vkDestroySampler(d.device(), samplerLinearClamp_, nullptr);
    if (samplerNearestClamp_) vkDestroySampler(d.device(), samplerNearestClamp_, nullptr);
    if (samplerEnvCube_) vkDestroySampler(d.device(), samplerEnvCube_, nullptr);

    for (VkFence f : frameFences_) vkDestroyFence(d.device(), f, nullptr);
    for (VkSemaphore s : imageAvailable_) vkDestroySemaphore(d.device(), s, nullptr);
    frameFences_.clear();
    imageAvailable_.clear();
    if (commandPool_) vkDestroyCommandPool(d.device(), commandPool_, nullptr);
    commandPool_ = VK_NULL_HANDLE;
    device_ = nullptr;
}

}  // namespace room2::render
