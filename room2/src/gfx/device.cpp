#include "gfx/device.hpp"

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#include <algorithm>
#include <cstring>
#include <set>

#include "core/log.hpp"

namespace room2::gfx {
namespace {

const char* kValidationLayers[] = {"VK_LAYER_KHRONOS_validation"};

bool layerAvailable(const char* name) {
    uint32_t count = 0;
    if (vkEnumerateInstanceLayerProperties(&count, nullptr) != VK_SUCCESS) return false;
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    for (const auto& l : layers)
        if (std::strcmp(l.layerName, name) == 0) return true;
    return false;
}

bool extensionAvailable(const char* name, const std::vector<VkExtensionProperties>& list) {
    for (const auto& e : list)
        if (std::strcmp(e.extensionName, name) == 0) return true;
    return false;
}

std::vector<VkExtensionProperties> instanceExtensions() {
    uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> list(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, list.data());
    return list;
}

std::vector<VkExtensionProperties> deviceExtensions(VkPhysicalDevice pd) {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> list(count);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &count, list.data());
    return list;
}

}  // namespace

Device::~Device() { shutdown(); }

// ---------------------------------------------------------------- debug callbacks
VKAPI_ATTR VkBool32 VKAPI_CALL Device::debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT types,
    const VkDebugUtilsMessengerCallbackDataEXT* data, void* userData) {
    (void)userData;
    const bool isPerf = (types & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT) != 0;
    (void)isPerf;
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        R2_ERROR("[vk", isPerf ? "/perf" : "", "] ", data->pMessage);
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        R2_WARN("[vk", isPerf ? "/perf" : "", "] ", data->pMessage);
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
        R2_DEBUG("[vk] ", data->pMessage);
    } else {
        R2_TRACE("[vk] ", data->pMessage);
    }
    return VK_FALSE;
}

VKAPI_ATTR VkBool32 VKAPI_CALL Device::debugReportCallback(
    VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType, uint64_t object,
    size_t location, int32_t code, const char* layerPrefix, const char* message, void* userData) {
    (void)objectType; (void)object; (void)location; (void)code; (void)userData;
    if (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT)
        R2_ERROR("[vk/", layerPrefix, "] ", message);
    else if (flags & VK_DEBUG_REPORT_WARNING_BIT_EXT)
        R2_WARN("[vk/", layerPrefix, "] ", message);
    else
        R2_DEBUG("[vk/", layerPrefix, "] ", message);
    return VK_FALSE;
}

// ---------------------------------------------------------------- init
bool Device::init(SDL_Window* window, const DeviceConfig& cfg) {
    cfg_ = cfg;
    vk::setDebugNaming(cfg_.enableDebugNaming);
    if (!createInstance(window)) return false;
    if (!pickPhysicalDevice()) return false;
    if (!createLogicalDevice(window)) return false;

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                     VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = graphics_.family;
    VK_CHECK(vkCreateCommandPool(device_, &poolInfo, nullptr, &immediatePool_));
    VK_NAME(device_, immediatePool_, "immediate command pool");

    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(device_, &fenceInfo, nullptr, &immediateFence_));
    VK_NAME(device_, immediateFence_, "immediate fence");
    return true;
}

bool Device::createInstance(SDL_Window* window) {
    std::vector<const char*> extensions;
    uint32_t sdlExtCount = 0;
    if (window && !cfg_.headless) {
        if (!SDL_Vulkan_GetInstanceExtensions(window, &sdlExtCount, nullptr)) {
            R2_ERROR("SDL_Vulkan_GetInstanceExtensions failed: ", SDL_GetError());
            return false;
        }
        extensions.resize(sdlExtCount);
        if (!SDL_Vulkan_GetInstanceExtensions(window, &sdlExtCount, extensions.data())) {
            R2_ERROR("SDL_Vulkan_GetInstanceExtensions failed: ", SDL_GetError());
            return false;
        }
    }

    const auto available = instanceExtensions();
    const bool haveDebugUtils = extensionAvailable(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, available);
    const bool haveDebugReport = extensionAvailable(VK_EXT_DEBUG_REPORT_EXTENSION_NAME, available);

    std::vector<const char*> layers;
    bool useValidation = cfg_.enableValidation;
    if (useValidation) {
        bool ok = true;
        for (const char* l : kValidationLayers) {
            if (!layerAvailable(l)) {
                ok = false;
                R2_WARN("validation layer not available: ", l);
            }
        }
        if (ok) {
            for (const char* l : kValidationLayers) layers.push_back(l);
            if (haveDebugUtils) extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            if (haveDebugReport) extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
            extensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
            if (!extensionAvailable(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME, available))
                extensions.pop_back();
        } else {
            useValidation = false;
            layers.clear();
        }
    }
    if (!useValidation) cfg_.enableValidation = false;

    // Surface extensions that SDL did not request but we may still need.
    for (const char* e : {VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME}) {
        if (extensionAvailable(e, available) &&
            std::find(extensions.begin(), extensions.end(), e) == extensions.end())
            extensions.push_back(e);
    }

    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = cfg_.appName.c_str();
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "room2";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkValidationFeaturesEXT validationFeatures{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
    VkValidationFeatureEnableEXT enabledFeatures[] = {
        VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT,
        VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT,
    };
    validationFeatures.enabledValidationFeatureCount = 2;
    validationFeatures.pEnabledValidationFeatures = enabledFeatures;

    VkInstanceCreateInfo createInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();
    createInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
    createInfo.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();
    if (useValidation && extensionAvailable(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME, available))
        createInfo.pNext = &validationFeatures;

    VkResult r = vkCreateInstance(&createInfo, nullptr, &instance_);
    if (r != VK_SUCCESS) {
        R2_ERROR("vkCreateInstance failed: ", vk::resultString(r));
        return false;
    }

    setupDebugMessenger();
    R2_INFO("Vulkan instance created (", extensions.size(), " extensions, validation ",
            useValidation ? "on" : "off", ")");
    return true;
}

void Device::setupDebugMessenger() {
    if (!cfg_.enableValidation) return;
    auto createUtils = [&](bool sync) {
        auto fn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
        if (!fn) return (VkDebugUtilsMessengerEXT)VK_NULL_HANDLE;
        VkDebugUtilsMessengerCreateInfoEXT info{
            VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        if (sync) info.messageSeverity |= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
        else info.messageSeverity |= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT;
        if (!sync) info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        info.pfnUserCallback = &Device::debugCallback;
        VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
        if (fn(instance_, &info, nullptr, &messenger) != VK_SUCCESS)
            return (VkDebugUtilsMessengerEXT)VK_NULL_HANDLE;
        return messenger;
    };
    debugMessenger_ = createUtils(false);
}

void Device::setValidationSync(bool enabled) {
    syncValidation_ = enabled;
    if (enabled && debugMessenger_ == VK_NULL_HANDLE) {
        debugMessenger_ = VK_NULL_HANDLE;
    }
}

// ---------------------------------------------------------------- device selection
bool Device::isDeviceSuitable(VkPhysicalDevice pd, uint32_t& outGraphics, uint32_t& outCompute,
                              uint32_t& outTransfer, bool& outPresent) const {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, nullptr);
    if (count == 0) return false;
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, families.data());

    outGraphics = VK_QUEUE_FAMILY_IGNORED;
    outCompute = VK_QUEUE_FAMILY_IGNORED;
    outTransfer = VK_QUEUE_FAMILY_IGNORED;
    for (uint32_t i = 0; i < count; ++i) {
        const auto flags = families[i].queueFlags;
        if (outGraphics == VK_QUEUE_FAMILY_IGNORED && (flags & VK_QUEUE_GRAPHICS_BIT))
            outGraphics = i;
        if (outCompute == VK_QUEUE_FAMILY_IGNORED && (flags & VK_QUEUE_COMPUTE_BIT) &&
            !(flags & VK_QUEUE_GRAPHICS_BIT))
            outCompute = i;
        if (outTransfer == VK_QUEUE_FAMILY_IGNORED && (flags & VK_QUEUE_TRANSFER_BIT) &&
            !(flags & VK_QUEUE_GRAPHICS_BIT) && !(flags & VK_QUEUE_COMPUTE_BIT))
            outTransfer = i;
    }
    if (outGraphics == VK_QUEUE_FAMILY_IGNORED) return false;
    if (outCompute == VK_QUEUE_FAMILY_IGNORED) outCompute = outGraphics;
    if (outTransfer == VK_QUEUE_FAMILY_IGNORED) outTransfer = outCompute;

    // Required Vulkan 1.3 core features.
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.pNext = &f13;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f12;
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    if (!f13.dynamicRendering || !f13.synchronization2) return false;

    outPresent = false;
    if (surface_ != VK_NULL_HANDLE) {
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(pd, outGraphics, surface_, &present);
        outPresent = present == VK_TRUE;
        if (!outPresent) return false;
        if (!extensionAvailable(VK_KHR_SWAPCHAIN_EXTENSION_NAME, deviceExtensions(pd)))
            return false;
        uint32_t formatCount = 0, presentModeCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface_, &formatCount, nullptr);
        vkGetPhysicalDeviceSurfacePresentModesKHR(pd, surface_, &presentModeCount, nullptr);
        if (formatCount == 0 || presentModeCount == 0) return false;
    }
    return true;
}

int Device::scoreDevice(VkPhysicalDevice pd) const {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(pd, &props);
    int score = 0;
    switch (props.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: score += 10000; break;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score += 5000; break;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: score += 2000; break;
        case VK_PHYSICAL_DEVICE_TYPE_CPU: score += cfg_.softwareFallback ? 100 : -100000; break;
        default: score += 500; break;
    }
    if (!cfg_.preferDiscreteGpu && props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
        score += 3000;
    score += static_cast<int>(props.limits.maxImageDimension2D / 1024);
    return score;
}

bool Device::pickPhysicalDevice() {
    uint32_t count = 0;
    VkResult r = vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (r != VK_SUCCESS || count == 0) {
        R2_ERROR("no Vulkan physical devices found");
        return false;
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());
    R2_INFO("found ", count, " Vulkan physical device(s)");

    int best = -1;
    int bestScore = -1 << 30;
    uint32_t bestG = 0, bestC = 0, bestT = 0;
    bool bestPresent = false;
    for (uint32_t i = 0; i < count; ++i) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(devices[i], &p);
        uint32_t g, c, t;
        bool present;
        bool suitable = isDeviceSuitable(devices[i], g, c, t, present);
        int score = suitable ? scoreDevice(devices[i]) : -1 << 29;
        if (cfg_.requestedDeviceIndex >= 0 && static_cast<uint32_t>(cfg_.requestedDeviceIndex) == i)
            score += 1000000;
        R2_INFO("  [", i, "] ", p.deviceName, "  api ", VK_VERSION_MAJOR(p.apiVersion), ".",
                VK_VERSION_MINOR(p.apiVersion), ".", VK_VERSION_PATCH(p.apiVersion),
                "  type=", static_cast<int>(p.deviceType),
                suitable ? "  (suitable)" : "  (missing required features)");
        if (score > bestScore) {
            bestScore = score;
            best = static_cast<int>(i);
            bestG = g; bestC = c; bestT = t; bestPresent = present;
        }
    }
    if (best < 0) {
        R2_ERROR("no suitable Vulkan device (Vulkan 1.3 with dynamicRendering + synchronization2 "
                 "is required)");
        return false;
    }
    physical_ = devices[best];
    vkGetPhysicalDeviceProperties(physical_, &props_);
    vkGetPhysicalDeviceMemoryProperties(physical_, &memProps_);
    {
        uint32_t g = bestG, c = bestC, t = bestT;
        bool present = bestPresent;
        (void)g; (void)c; (void)t; (void)present;
        graphics_.family = bestG;
        compute_.family = bestC;
        transfer_.family = bestT;
        presentSupported_ = bestPresent;
    }
    return true;
}

bool Device::createLogicalDevice(SDL_Window* window) {
    if (window && !cfg_.headless) {
        if (!SDL_Vulkan_CreateSurface(window, instance_, &surface_)) {
            R2_ERROR("SDL_Vulkan_CreateSurface failed: ", SDL_GetError());
            return false;
        }
        // Re-validate present support now that the surface exists.
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(physical_, graphics_.family, surface_, &present);
        presentSupported_ = present == VK_TRUE;
        if (!presentSupported_) {
            R2_WARN("graphics queue family does not support present on this surface");
        }
    }

    const auto devExts = deviceExtensions(physical_);
    std::vector<const char*> enabledExts;
    if (surface_ != VK_NULL_HANDLE) enabledExts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    // Optional niceties.
    if (extensionAvailable(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME, devExts))
        enabledExts.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);

    std::set<uint32_t> uniqueFamilies{graphics_.family, compute_.family, transfer_.family};
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    const float priority = 1.0f;
    for (uint32_t fam : uniqueFamilies) {
        VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qi.queueFamilyIndex = fam;
        qi.queueCount = 1;
        qi.pQueuePriorities = &priority;
        queueInfos.push_back(qi);
    }

    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    f13.shaderDemoteToHelperInvocation = VK_TRUE;   // 'discard' in the UI shader
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.pNext = &f13;
    f12.descriptorIndexing = VK_TRUE;
    f12.runtimeDescriptorArray = VK_TRUE;
    f12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    f12.descriptorBindingPartiallyBound = VK_TRUE;
    f12.descriptorBindingVariableDescriptorCount = VK_TRUE;
    f12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    f12.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
    f12.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
    f12.descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE;
    f12.hostQueryReset = VK_TRUE;
    f12.timelineSemaphore = VK_TRUE;
    f12.bufferDeviceAddress = VK_TRUE;
    f12.scalarBlockLayout = VK_TRUE;
    f12.uniformBufferStandardLayout = VK_TRUE;

    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f12;
    f2.features.samplerAnisotropy = VK_TRUE;
    f2.features.fillModeNonSolid = VK_TRUE;
    f2.features.wideLines = VK_TRUE;
    f2.features.independentBlend = VK_TRUE;
    f2.features.depthClamp = VK_TRUE;
    f2.features.depthBiasClamp = VK_TRUE;
    f2.features.fragmentStoresAndAtomics = VK_TRUE;
    f2.features.vertexPipelineStoresAndAtomics = VK_TRUE;
    f2.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    f2.features.sampleRateShading = VK_TRUE;

    VkDeviceCreateInfo createInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    createInfo.pNext = &f2;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
    createInfo.pQueueCreateInfos = queueInfos.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExts.size());
    createInfo.ppEnabledExtensionNames = enabledExts.empty() ? nullptr : enabledExts.data();
    createInfo.pEnabledFeatures = nullptr;

    VkResult r = vkCreateDevice(physical_, &createInfo, nullptr, &device_);
    if (r != VK_SUCCESS) {
        R2_ERROR("vkCreateDevice failed: ", vk::resultString(r));
        return false;
    }

    vkGetDeviceQueue(device_, graphics_.family, 0, &graphics_.handle);
    vkGetDeviceQueue(device_, compute_.family, 0, &compute_.handle);
    vkGetDeviceQueue(device_, transfer_.family, 0, &transfer_.handle);
    if (!compute_.valid()) compute_ = graphics_;
    if (!transfer_.valid()) transfer_ = graphics_;

    features_.descriptorIndexing = true;
    features_.timelineSemaphore = true;
    features_.hostQueryReset = true;
    features_.samplerAnisotropy = true;
    features_.fillModeNonSolid = true;
    features_.wideLines = true;
    features_.independentBlend = true;
    features_.depthClamp = true;
    features_.bufferDeviceAddress = true;
    features_.scalarBlockLayout = true;
    features_.maxSamplerAnisotropy = props_.limits.maxSamplerAnisotropy;
    features_.limits = props_.limits;

    VkSampleCountFlags counts = props_.limits.framebufferColorSampleCounts &
                                props_.limits.framebufferDepthSampleCounts;
    VkSampleCountFlagBits bestMsaa = VK_SAMPLE_COUNT_1_BIT;
    for (VkSampleCountFlagBits s : {VK_SAMPLE_COUNT_8_BIT, VK_SAMPLE_COUNT_4_BIT,
                                    VK_SAMPLE_COUNT_2_BIT}) {
        if (counts & s) {
            bestMsaa = s;
            break;
        }
    }
    features_.bestMsaa = bestMsaa;
    features_.maxMsaaSamples = static_cast<uint32_t>(bestMsaa);

    char buf[512];
    const uint32_t mb = static_cast<uint32_t>(props_.limits.maxComputeSharedMemorySize / 1024);
    std::snprintf(buf, sizeof(buf), "%s | %s | driver %u.%u.%u | %ux%u | %u MB shared",
                  props_.deviceName,
                  props_.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? "discrete GPU"
                  : props_.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? "integrated GPU"
                  : props_.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ? "CPU (software)" : "other",
                  VK_VERSION_MAJOR(props_.driverVersion), VK_VERSION_MINOR(props_.driverVersion),
                  VK_VERSION_PATCH(props_.driverVersion), props_.limits.maxImageDimension2D,
                  props_.limits.maxImageDimension2D, mb);
    summary_ = buf;
    R2_INFO("using device: ", summary_);
    return true;
}

// ---------------------------------------------------------------- helpers
uint32_t Device::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required,
                                VkMemoryPropertyFlags preferred) const {
    // Prefer a type that satisfies `preferred` as well, if one exists.
    for (uint32_t i = 0; i < memProps_.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) &&
            (memProps_.memoryTypes[i].propertyFlags & required) == required &&
            (memProps_.memoryTypes[i].propertyFlags & preferred) == preferred)
            return i;
    }
    for (uint32_t i = 0; i < memProps_.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) &&
            (memProps_.memoryTypes[i].propertyFlags & required) == required)
            return i;
    }
    R2_FATAL("no memory type satisfies required flags 0x", required, " (bits 0x", typeBits, ")");
    return 0;
}

VkCommandBuffer Device::beginImmediate(const char* name) {
    VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = immediatePool_;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(device_, &alloc, &cmd));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));
    if (name) VK_NAME(device_, cmd, "%s", name);
    return cmd;
}

void Device::endImmediate(VkCommandBuffer cmd, bool wait) {
    VK_CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(graphics_.handle, 1, &submit, wait ? immediateFence_ : VK_NULL_HANDLE));
    if (wait) {
        VK_CHECK(vkWaitForFences(device_, 1, &immediateFence_, VK_TRUE, ~0ull));
        VK_CHECK(vkResetFences(device_, 1, &immediateFence_));
    }
    vkFreeCommandBuffers(device_, immediatePool_, 1, &cmd);
}

void Device::waitIdle() const {
    if (device_) vkDeviceWaitIdle(device_);
}

void Device::beginLabel(VkCommandBuffer cmd, const char* name, float r, float g, float b) const {
    if (!cfg_.enableValidation) return;
    auto fn = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
        vkGetDeviceProcAddr(device_, "vkCmdBeginDebugUtilsLabelEXT"));
    if (!fn) return;
    VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    label.pLabelName = name;
    label.color[0] = r; label.color[1] = g; label.color[2] = b; label.color[3] = 1.0f;
    fn(cmd, &label);
}

void Device::endLabel(VkCommandBuffer cmd) const {
    if (!cfg_.enableValidation) return;
    auto fn = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
        vkGetDeviceProcAddr(device_, "vkCmdEndDebugUtilsLabelEXT"));
    if (fn) fn(cmd);
}

void Device::insertLabel(VkCommandBuffer cmd, const char* name, float r, float g, float b) const {
    if (!cfg_.enableValidation) return;
    auto fn = reinterpret_cast<PFN_vkCmdInsertDebugUtilsLabelEXT>(
        vkGetDeviceProcAddr(device_, "vkCmdInsertDebugUtilsLabelEXT"));
    if (!fn) return;
    VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    label.pLabelName = name;
    label.color[0] = r; label.color[1] = g; label.color[2] = b; label.color[3] = 1.0f;
    fn(cmd, &label);
}

void Device::shutdown() {
    if (!device_ && !instance_) return;
    if (device_) {
        vkDeviceWaitIdle(device_);
        if (immediateFence_) vkDestroyFence(device_, immediateFence_, nullptr);
        if (immediatePool_) vkDestroyCommandPool(device_, immediatePool_, nullptr);
        immediateFence_ = VK_NULL_HANDLE;
        immediatePool_ = VK_NULL_HANDLE;
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    if (surface_ && instance_) {
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
    }
    if (instance_) {
        if (debugMessenger_) {
            auto fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
            if (fn) fn(instance_, debugMessenger_, nullptr);
            debugMessenger_ = VK_NULL_HANDLE;
        }
        if (debugReport_) {
            auto fn = reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(
                vkGetInstanceProcAddr(instance_, "vkDestroyDebugReportCallbackEXT"));
            if (fn) fn(instance_, debugReport_, nullptr);
            debugReport_ = VK_NULL_HANDLE;
        }
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

}  // namespace room2::gfx
