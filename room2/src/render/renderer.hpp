// room2 - the deferred physically based renderer.
//
// Frame graph (all passes are recorded into one primary command buffer):
//   1. environment probe   (compute, once at load: analytic room radiance -> cubemap)
//   2. shadow atlas        (depth-only, 6 cube faces per shadow-casting light)
//   3. G-buffer            (4 MRT + depth)
//   4. SSAO + bilateral blur (compute, half resolution)
//   5. deferred lighting   (compute: direct + shadows + ambient + probe specular)
//   6. opaque copy         (for screen-space refraction)
//   7. glass               (forward, blended)
//   8. TAA resolve         (compute, HDR history)
//   9. bloom               (prefilter + downsample/upsample chain)
//  10. tonemap + grade     (fullscreen triangle into the swapchain image)
//  11. UI overlay          (alpha blended quads + font atlas)
#pragma once

#include <string>
#include <vector>

#include "gfx/device.hpp"
#include "gfx/pipeline.hpp"
#include "gfx/resources.hpp"
#include "gfx/swapchain.hpp"
#include "procgen/texture.hpp"
#include "render/camera.hpp"
#include "render/gpu_types.hpp"
#include "scene/scene.hpp"

namespace room2::render {

struct RendererConfig {
    uint32_t width = 1280;
    uint32_t height = 720;
    float renderScale = 1.0f;
    bool headless = false;
    int shadowResolution = 1024;
    int envProbeSize = 64;
    int envProbeSamples = 64;
    int maxFramesInFlight = 2;
    bool enableSSAO = true;
    bool enableBloom = true;
    bool enableTAA = true;
    bool enableGlass = true;
    bool enableUi = true;
    bool enableShadows = true;
    bool enableValidation = true;
    bool forceMsaa1 = true;
    bool verboseStats = false;
};

struct FrameStats {
    double cpuFrameMs = 0.0;
    double gpuFrameMs = 0.0;
    double fps = 0.0;
    uint32_t drawCalls = 0;
    uint32_t shadowDrawCalls = 0;
    uint32_t triangles = 0;
    uint32_t instances = 0;
    uint32_t gpuBufferBytes = 0;
    uint32_t textureBytes = 0;
    uint32_t textureCount = 0;
    uint32_t frameIndex = 0;
};

// A minimal immediate-mode UI draw list (filled by the game each frame).
struct UiVertex {
    Vec2 position;
    Vec2 uv;
    Vec4 color;
};

struct UiDrawList {
    std::vector<UiVertex> vertices;
    std::vector<uint32_t> indices;
    // Texture switches are expressed as separate "batches" over the index buffer.
    struct Batch {
        uint32_t firstIndex = 0;
        uint32_t indexCount = 0;
        bool useTexture = false;
    };
    std::vector<Batch> batches;

    void clear() {
        vertices.clear();
        indices.clear();
        batches.clear();
    }
    void addQuad(Vec2 min, Vec2 max, Vec4 color, Vec2 uvMin = Vec2(0, 0),
                 Vec2 uvMax = Vec2(1, 1), bool textured = false);
    void addTexturedQuad(Vec2 min, Vec2 max, Vec2 uvMin, Vec2 uvMax, Vec4 color);
    bool empty() const { return indices.empty(); }
};

class Renderer {
public:
    Renderer() = default;
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    bool init(gfx::Device& device, gfx::Swapchain* swapchain, const RendererConfig& config);
    void shutdown();

    // Recreates the internal render targets. Safe to call every frame; it is a no-op
    // when the size has not changed.
    void resize(uint32_t width, uint32_t height);

    // --- texture / material upload ---------------------------------------
    // Uploads a 2D texture and returns its bindless slot. `srgb` selects the sRGB
    // format for base-colour data. Mips are generated when `generateMips` is true.
    uint32_t addTexture(const procgen::TextureData& image, bool srgb, bool generateMips,
                        const std::string& name);
    // Uploads a cubemap (6 faces, square) and returns its bindless slot; cube slots
    // live in a separate range starting at kCubeSlotBase.
    uint32_t addCubeTexture(const std::vector<procgen::TextureData>& faces,
                            const std::string& name, bool srgb);
    void flushTextureUploads();
    // Binds the UI font atlas to the overlay pass (call after creating it).
    void setUiAtlas(const gfx::Image& atlas);

    // --- scene upload -----------------------------------------------------
    // Creates or grows the GPU-side geometry, material and texture data. Must be
    // called after the scene has been built and again whenever geometry is appended.
    void uploadScene(scene::Scene& scene);

    // --- frame ------------------------------------------------------------
    bool beginFrame();
    void render(const scene::Scene& scene, const Camera& camera, float dt, const UiDrawList* ui,
                float exposure);
    void endFrame();

    // Renders one frame without presenting (used by --screenshot / --selftest).
    // `iterations` frames are simulated so temporal effects settle.
    bool renderOffscreen(const scene::Scene& scene, const Camera& camera, float dt,
                         const UiDrawList* ui, float exposure, int iterations,
                         std::vector<uint8_t>& rgbaOut, uint32_t& widthOut, uint32_t& heightOut);

    // --- info -------------------------------------------------------------
    // Captures the next presented (windowed) frame instead of only the offscreen one.
    void requestCapture() { capturePending_ = true; }
    // Copies the most recent offscreen/swapchain frame back to host memory as RGBA8.
    bool capture(std::vector<uint8_t>& rgbaOut, uint32_t& widthOut, uint32_t& heightOut);
    // Diagnostic: reads back an internal target into linear RGBA floats.
    struct DebugTarget {
        std::string name;
        uint32_t width = 0;
        uint32_t height = 0;
        std::vector<float> rgba;   // 4 floats per texel
        float minValue = 0.0f;
        float maxValue = 0.0f;
        float meanLuma = 0.0f;
    };
    std::vector<DebugTarget> debugReadTargets();
    const FrameStats& stats() const { return stats_; }
    const RendererConfig& config() const { return config_; }
    uint32_t renderWidth() const { return renderWidth_; }
    uint32_t renderHeight() const { return renderHeight_; }
    float aspectRatio() const {
        return static_cast<float>(renderWidth_) / static_cast<float>(renderHeight_);
    }
    void setConfigFlag(const std::string& name, bool value);
    bool configFlag(const std::string& name) const;
    // Exposed so the game can query the bindless slot of the environment probe.
    uint32_t envProbeSlot() const { return envProbeSlot_; }

private:
    // --- setup helpers ----------------------------------------------------
    bool createDescriptorLayouts();
    bool createPipelines();
    bool createRenderTargets();
    void destroyRenderTargets();
    bool createShadowResources();
    void initializeImageLayouts();
    void buildEnvironmentProbe(const scene::Scene& scene);

    void writeGlobals(const scene::Scene& scene, const Camera& camera, float dt, float exposure);
    void updateLights(const scene::Scene& scene);
    void updateInstances(const scene::Scene& scene);
    void updateMaterials(const scene::Scene& scene);

    void cmdBeginRendering(VkCommandBuffer cmd, const std::vector<VkRenderingAttachmentInfo>& color,
                           const VkRenderingAttachmentInfo* depth, VkExtent2D extent,
                           const char* label);
    void cmdDispatch(VkCommandBuffer cmd, VkPipeline pipeline, VkPipelineLayout layout,
                     VkDescriptorSet set, uint32_t gx, uint32_t gy, uint32_t gz,
                     const void* push = nullptr, uint32_t pushSize = 0,
                     const char* label = nullptr);
    void drawSceneGeometry(VkCommandBuffer cmd, const scene::Scene& scene, bool shadowPass);
    void drawGlassGeometry(VkCommandBuffer cmd, const scene::Scene& scene);
    void barrierForSampling(VkCommandBuffer cmd, const gfx::Image& image);

    gfx::Device* device_ = nullptr;
    gfx::Swapchain* swapchain_ = nullptr;
    RendererConfig config_{};
    gfx::PipelineCache pipelineCache_{};

    // --- descriptor plumbing ---------------------------------------------
    VkDescriptorSetLayout setLayoutGlobals_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayoutTextures_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayoutEmpty_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayoutShadow_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayoutLighting_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayoutSsao_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayoutSsaoBlur_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayoutBloomPrefilter_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayoutBloomDown_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayoutBloomUp_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayoutGlass_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayoutTonemap_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayoutUi_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayoutProbe_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayoutTaa_ = VK_NULL_HANDLE;

    gfx::DescriptorPool mainPool_{};
    gfx::DescriptorPool bindlessPool_{};
    gfx::DescriptorPool framePool_{};
    VkDescriptorSet setGlobals_[8]{};
    VkDescriptorSet setTextures_ = VK_NULL_HANDLE;
    VkDescriptorSet setShadow_ = VK_NULL_HANDLE;
    VkDescriptorSet setLighting_ = VK_NULL_HANDLE;
    VkDescriptorSet setSsao_ = VK_NULL_HANDLE;
    VkDescriptorSet setSsaoBlur_ = VK_NULL_HANDLE;
    // Per frame-in-flight: these are rewritten every frame to point at the resolved
    // HDR image, so each slot needs its own set to avoid updating one in flight.
    VkDescriptorSet setBloomPrefilter_[8]{};
    VkDescriptorSet setBloomDown_[8]{};
    VkDescriptorSet setBloomUp_[8]{};
    VkDescriptorSet setGlass_ = VK_NULL_HANDLE;
    VkDescriptorSet setTonemap_[8]{};
    VkDescriptorSet setUi_ = VK_NULL_HANDLE;
    VkDescriptorSet setProbe_ = VK_NULL_HANDLE;
    VkDescriptorSet setTaa_[2]{};
    VkDescriptorSet setTaaPrev_[2]{};

    VkPipelineLayout layoutShadow_ = VK_NULL_HANDLE;
    VkPipelineLayout layoutGbuffer_ = VK_NULL_HANDLE;
    VkPipelineLayout layoutGlass_ = VK_NULL_HANDLE;
    VkPipelineLayout layoutLighting_ = VK_NULL_HANDLE;
    VkPipelineLayout layoutSsao_ = VK_NULL_HANDLE;
    VkPipelineLayout layoutSsaoBlur_ = VK_NULL_HANDLE;
    VkPipelineLayout layoutBloomPrefilter_ = VK_NULL_HANDLE;
    VkPipelineLayout layoutBloomDown_ = VK_NULL_HANDLE;
    VkPipelineLayout layoutBloomUp_ = VK_NULL_HANDLE;
    VkPipelineLayout layoutTonemap_ = VK_NULL_HANDLE;
    VkPipelineLayout layoutUi_ = VK_NULL_HANDLE;
    VkPipelineLayout layoutProbe_ = VK_NULL_HANDLE;
    VkPipelineLayout layoutTaa_ = VK_NULL_HANDLE;

    VkPipeline pipelineShadow_ = VK_NULL_HANDLE;
    VkPipeline pipelineGbuffer_ = VK_NULL_HANDLE;
    VkPipeline pipelineGlass_ = VK_NULL_HANDLE;
    VkPipeline pipelineLighting_ = VK_NULL_HANDLE;
    VkPipeline pipelineSsao_ = VK_NULL_HANDLE;
    VkPipeline pipelineSsaoBlur_ = VK_NULL_HANDLE;
    VkPipeline pipelineBloomPrefilter_ = VK_NULL_HANDLE;
    VkPipeline pipelineBloomDown_ = VK_NULL_HANDLE;
    VkPipeline pipelineBloomUp_ = VK_NULL_HANDLE;
    VkPipeline pipelineTonemap_ = VK_NULL_HANDLE;
    VkPipeline pipelineUi_ = VK_NULL_HANDLE;
    VkPipeline pipelineProbe_ = VK_NULL_HANDLE;
    VkPipeline pipelineTaa_ = VK_NULL_HANDLE;

    VkShaderModule vsFullscreen_ = VK_NULL_HANDLE;
    VkShaderModule vsGbuffer_ = VK_NULL_HANDLE;
    VkShaderModule fsGbuffer_ = VK_NULL_HANDLE;
    VkShaderModule vsShadow_ = VK_NULL_HANDLE;
    VkShaderModule fsShadow_ = VK_NULL_HANDLE;
    VkShaderModule vsGlass_ = VK_NULL_HANDLE;
    VkShaderModule fsGlass_ = VK_NULL_HANDLE;
    VkShaderModule csLighting_ = VK_NULL_HANDLE;
    VkShaderModule csSsao_ = VK_NULL_HANDLE;
    VkShaderModule csSsaoBlur_ = VK_NULL_HANDLE;
    VkShaderModule csBloomPrefilter_ = VK_NULL_HANDLE;
    VkShaderModule csBloomDown_ = VK_NULL_HANDLE;
    VkShaderModule csBloomUp_ = VK_NULL_HANDLE;
    VkShaderModule fsTonemap_ = VK_NULL_HANDLE;
    VkShaderModule vsUi_ = VK_NULL_HANDLE;
    VkShaderModule fsUi_ = VK_NULL_HANDLE;
    VkShaderModule csProbe_ = VK_NULL_HANDLE;
    VkShaderModule csTaa_ = VK_NULL_HANDLE;

    // --- render targets ---------------------------------------------------
    uint32_t renderWidth_ = 0;
    uint32_t renderHeight_ = 0;
    uint32_t outputWidth_ = 0;
    uint32_t outputHeight_ = 0;
    uint32_t bloomMipCount_ = 0;

    gfx::Image gbufferAlbedo_;    // R8G8B8A8_UNORM: albedo.rgb, ao
    gfx::Image gbufferNormal_;    // R16G16B16A16_SFLOAT: normal.rgb, roughness
    gfx::Image gbufferMetal_;     // R8G8B8A8_UNORM: metallic, transmission, kind, 1
    gfx::Image gbufferEmissive_;  // R16G16B16A16_SFLOAT: emissive
    gfx::Image depth_;            // D32_SFLOAT
    gfx::Image hdrColor_;         // RGBA16F lighting result / glass target
    gfx::Image hdrOpaqueCopy_;    // RGBA16F before glass
    gfx::Image hdrHistory_[2];    // TAA history ping-pong
    gfx::Image aoRaw_;            // R16F half res
    gfx::Image aoBlur_;           // R16F half res
    gfx::Image bloomMips_[8];     // RGBA16F chain
    gfx::Image ldrColor_;         // RGBA8 tonemapped (headless path / FXAA input)
    gfx::Image uiColor_;          // swapchain-sized UI target (headless only)

    // --- shadow / environment --------------------------------------------
    gfx::Image shadowAtlas_;      // D32_SFLOAT 2D array, 6 layers per light
    // Per-layer views, created once. A view must outlive the command buffer that
    // references it, so these cannot be made and destroyed inside the render loop.
    std::vector<VkImageView> shadowFaceViews_;
    VkSampler samplerShadow_ = VK_NULL_HANDLE;
    VkSampler samplerLinearRepeat_ = VK_NULL_HANDLE;
    VkSampler samplerLinearClamp_ = VK_NULL_HANDLE;
    VkSampler samplerNearestClamp_ = VK_NULL_HANDLE;
    gfx::Image envProbe_;         // RGBA16F cubemap with a rough mip chain
    std::vector<VkImageView> envProbeMipViews_;   // one 2D-array view per mip
    VkSampler samplerEnvCube_ = VK_NULL_HANDLE;
    // Vulkan requires the immutable-sampler array to hold descriptorCount entries.
    std::vector<VkSampler> bindlessSamplers_;
    gfx::DescriptorPool probePool_{};

    // --- scene GPU data ---------------------------------------------------
    gfx::Buffer vertexBuffer_;
    gfx::Buffer indexBuffer_;
    size_t vertexCapacity_ = 0;
    size_t indexCapacity_ = 0;
    std::vector<gfx::Buffer> instanceBuffers_;   // one per frame in flight
    std::vector<gfx::Buffer> globalsBuffers_;    // one per frame in flight
    gfx::Buffer materialBuffer_;
    gfx::Buffer lightBuffer_;
    size_t materialCapacity_ = 0;
    size_t lightCapacity_ = 0;
    size_t instanceCapacity_ = 0;
    uint32_t uploadedInstanceCount_ = 0;

    // --- textures ---------------------------------------------------------
    struct PendingTexture {
        procgen::TextureData image;
        std::vector<procgen::TextureData> faces;
        bool srgb = false;
        bool generateMips = true;
        bool cube = false;
        std::string name;
    };
    std::vector<PendingTexture> pendingTextures_;
    std::vector<gfx::Image> textures_;
    std::vector<VkDescriptorImageInfo> textureInfos_;
    uint32_t textureSlotCount_ = 0;
    uint32_t envProbeSlot_ = 0;
    bool texturesDirty_ = false;

    // --- UI ---------------------------------------------------------------
    gfx::Buffer uiVertexBuffer_;
    gfx::Buffer uiIndexBuffer_;
    size_t uiVertexCapacity_ = 0;
    size_t uiIndexCapacity_ = 0;
    gfx::Image fontAtlas_;
    uint32_t uiQuadCount_ = 0;

    // --- frame state ------------------------------------------------------
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers_;
    std::vector<VkFence> frameFences_;
    std::vector<VkSemaphore> imageAvailable_;
    uint32_t frameIndex_ = 0;
    uint32_t swapchainImageIndex_ = 0;
    bool frameOpen_ = false;
    bool historyValid_ = false;
    uint32_t jitterIndex_ = 0;
    uint32_t historyIndex_ = 0;
    FrameStats stats_{};
    float timeAccum_ = 0.0f;
    Mat4 prevViewProjection_{};
    Vec3 cameraPosition_{0, 0, 0};
    const Frustum* cullFrustum_ = nullptr;
    double lastCpuFrameMs_ = 0.0;

    // --- offscreen target -------------------------------------------------
    gfx::Image offscreenColor_;
    gfx::Buffer readbackBuffer_;
    size_t readbackCapacity_ = 0;
    bool headless_ = false;
    bool needsFullUpload_ = true;
    bool capturePending_ = false;
    bool pendingCapture_ = false;
};

// Halton sequence (base 2, 3) used for the TAA sample jitter.
Vec2 haltonJitter(uint32_t index, uint32_t width, uint32_t height);

}  // namespace room2::render
