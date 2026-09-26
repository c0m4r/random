// room2 - Vulkan instance / physical device / logical device management.
#pragma once

#include <string>
#include <vector>

#include "gfx/vk_common.hpp"

struct SDL_Window;

namespace room2::gfx {

struct DeviceConfig {
    std::string appName = "room2";
    bool enableValidation = true;
    bool headless = false;          // no window, no swapchain (offscreen rendering)
    bool preferDiscreteGpu = true;
    int requestedDeviceIndex = -1;  // -1 = automatic
    bool enableDebugNaming = true;
    bool softwareFallback = true;   // accept a CPU implementation if nothing else exists
    bool requestRayTracing = false;
    int maxFramesInFlight = 2;
};

// A queue family together with the single queue we use from it.
struct Queue {
    uint32_t family = VK_QUEUE_FAMILY_IGNORED;
    VkQueue handle = VK_NULL_HANDLE;
    bool valid() const { return handle != VK_NULL_HANDLE; }
};

struct DeviceFeatures {
    bool descriptorIndexing = false;
    bool timelineSemaphore = false;
    bool hostQueryReset = false;
    bool samplerAnisotropy = false;
    bool fillModeNonSolid = false;
    bool wideLines = false;
    bool independentBlend = false;
    bool depthClamp = false;
    bool shaderInt64 = false;
    bool scalarBlockLayout = false;
    bool bufferDeviceAddress = false;
    bool multiview = false;
    float maxSamplerAnisotropy = 1.0f;
    uint32_t maxMsaaSamples = 1;
    VkSampleCountFlagBits bestMsaa = VK_SAMPLE_COUNT_1_BIT;
    VkPhysicalDeviceLimits limits{};
};

class Device {
public:
    Device() = default;
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    // `window` may be null when cfg.headless is true.
    bool init(SDL_Window* window, const DeviceConfig& cfg);
    void shutdown();
    bool valid() const { return device_ != VK_NULL_HANDLE; }

    // --- handles ---------------------------------------------------------
    VkInstance instance() const { return instance_; }
    VkPhysicalDevice physicalDevice() const { return physical_; }
    VkDevice device() const { return device_; }
    VkSurfaceKHR surface() const { return surface_; }
    VkAllocationCallbacks* allocator() const { return nullptr; }

    const Queue& graphicsQueue() const { return graphics_; }
    const Queue& computeQueue() const { return compute_; }
    const Queue& transferQueue() const { return transfer_; }

    const VkPhysicalDeviceProperties& properties() const { return props_; }
    const VkPhysicalDeviceMemoryProperties& memoryProperties() const { return memProps_; }
    const DeviceFeatures& features() const { return features_; }
    const DeviceConfig& config() const { return cfg_; }

    bool hasSurface() const { return surface_ != VK_NULL_HANDLE; }
    bool supportsPresent() const { return presentSupported_; }

    // --- helpers ---------------------------------------------------------
    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required,
                            VkMemoryPropertyFlags preferred = 0) const;

    // One-shot command buffer; must be paired with endImmediate().
    VkCommandBuffer beginImmediate(const char* name = "immediate");
    void endImmediate(VkCommandBuffer cmd, bool wait = true);

    // Convenience: run `fn` inside a one-shot command buffer and wait for it.
    template <typename Fn>
    void immediate(Fn&& fn, const char* name = "immediate") {
        VkCommandBuffer cmd = beginImmediate(name);
        fn(cmd);
        endImmediate(cmd, true);
    }

    void waitIdle() const;

    // Debug messenger label helpers (no-ops when validation is off).
    void beginLabel(VkCommandBuffer cmd, const char* name, float r = 0.6f, float g = 0.6f,
                    float b = 0.6f) const;
    void endLabel(VkCommandBuffer cmd) const;
    void insertLabel(VkCommandBuffer cmd, const char* name, float r = 0.6f, float g = 0.6f,
                     float b = 0.6f) const;

    // Enables/disables GPU-assisted validation and sync validation at runtime.
    void setValidationSync(bool enabled);
    bool validationSyncEnabled() const { return syncValidation_; }

    // Human-readable summary of the selected device.
    const std::string& summary() const { return summary_; }

private:
    bool createInstance(SDL_Window* window);
    bool pickPhysicalDevice();
    bool createLogicalDevice(SDL_Window* window);
    void setupDebugMessenger();
    bool isDeviceSuitable(VkPhysicalDevice pd, uint32_t& outGraphics, uint32_t& outCompute,
                          uint32_t& outTransfer, bool& outPresent) const;
    int scoreDevice(VkPhysicalDevice pd) const;
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT types,
        const VkDebugUtilsMessengerCallbackDataEXT* data, void* userData);
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallback(
        VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType, uint64_t object,
        size_t location, int32_t code, const char* layerPrefix, const char* message,
        void* userData);

    DeviceConfig cfg_{};
    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;
    VkDebugReportCallbackEXT debugReport_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessengerNoSync_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props_{};
    VkPhysicalDeviceMemoryProperties memProps_{};
    DeviceFeatures features_{};
    Queue graphics_{}, compute_{}, transfer_{};
    bool presentSupported_ = false;
    bool syncValidation_ = false;
    uint32_t validationSeverity_ = 0;
    VkCommandPool immediatePool_ = VK_NULL_HANDLE;
    VkFence immediateFence_ = VK_NULL_HANDLE;
    std::string summary_;
};

}  // namespace room2::gfx
