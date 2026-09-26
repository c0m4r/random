#include "gfx/swapchain.hpp"

#include <algorithm>

#include "core/log.hpp"

namespace room2::gfx {

Swapchain::~Swapchain() { shutdown(); }

bool Swapchain::init(Device& device, uint32_t width, uint32_t height, bool vsync,
                     VkFormat preferredFormat) {
    device_ = &device;
    if (!device.hasSurface()) {
        R2_ERROR("Swapchain::init called without a surface");
        return false;
    }
    vsync_ = vsync;
    preferredFormat_ = preferredFormat;
    return recreate(width, height, vsync) && format_ != VK_FORMAT_UNDEFINED;
}

void Swapchain::destroyResources() {
    if (!device_) return;
    for (VkSemaphore s : renderFinished_)
        if (s) vkDestroySemaphore(device_->device(), s, nullptr);
    renderFinished_.clear();
    for (VkImageView v : views_)
        if (v) vkDestroyImageView(device_->device(), v, nullptr);
    views_.clear();
    images_.clear();
    if (swapchain_) {
        vkDestroySwapchainKHR(device_->device(), swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

void Swapchain::shutdown() {
    destroyResources();
    device_ = nullptr;
}

bool Swapchain::recreate(uint32_t width, uint32_t height, bool vsync) {
    if (!device_ || !device_->hasSurface()) return false;
    VkDevice dev = device_->device();
    VkPhysicalDevice pd = device_->physicalDevice();
    VkSurfaceKHR surface = device_->surface();

    vkDeviceWaitIdle(dev);

    VkSurfaceCapabilitiesKHR caps{};
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surface, &caps));

    uint32_t formatCount = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &formatCount, nullptr));
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &formatCount, formats.data()));

    VkSurfaceFormatKHR chosen =
        formats.empty()
            ? VkSurfaceFormatKHR{preferredFormat_, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}
            : formats[0];
    // Prefer an 8-bit sRGB surface so the compositor treats our output correctly.
    bool found = false;
    for (const auto& f : formats) {
        if (f.format == preferredFormat_ && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen = f;
            found = true;
            break;
        }
    }
    if (!found) {
        for (const auto& f : formats) {
            if (f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR &&
                (f.format == VK_FORMAT_R8G8B8A8_UNORM || f.format == VK_FORMAT_B8G8R8A8_UNORM)) {
                chosen = f;
                found = true;
                break;
            }
        }
    }
    if (!found && !formats.empty()) chosen = formats[0];

    uint32_t presentModeCount = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(pd, surface, &presentModeCount, nullptr));
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(pd, surface, &presentModeCount,
                                                       presentModes.data()));
    auto hasMode = [&](VkPresentModeKHR m) {
        return std::find(presentModes.begin(), presentModes.end(), m) != presentModes.end();
    };
    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
    if (!vsync) {
        if (hasMode(VK_PRESENT_MODE_MAILBOX_KHR)) presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
        else if (hasMode(VK_PRESENT_MODE_IMMEDIATE_KHR)) presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    }

    VkExtent2D extent{};
    if (caps.currentExtent.width != 0xFFFFFFFFu) {
        extent = caps.currentExtent;
    } else {
        extent.width = std::clamp(width ? width : 1280u, caps.minImageExtent.width,
                                  caps.maxImageExtent.width);
        extent.height = std::clamp(height ? height : 720u, caps.minImageExtent.height,
                                   caps.maxImageExtent.height);
    }
    if (extent.width == 0 || extent.height == 0) {
        R2_WARN("swapchain extent is zero; deferring recreation");
        needsRecreate_ = true;
        return false;
    }

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0) imageCount = std::min(imageCount, caps.maxImageCount);

    VkCompositeAlphaFlagBitsKHR composite = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if (!(caps.supportedCompositeAlpha & composite)) {
        for (VkCompositeAlphaFlagBitsKHR c :
             {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
              VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
              VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR}) {
            if (caps.supportedCompositeAlpha & c) {
                composite = c;
                break;
            }
        }
    }

    VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    info.surface = surface;
    info.minImageCount = imageCount;
    info.imageFormat = chosen.format;
    info.imageColorSpace = chosen.colorSpace;
    info.imageExtent = extent;
    info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.preTransform = caps.currentTransform;
    info.compositeAlpha = composite;
    info.presentMode = presentMode;
    info.clipped = VK_TRUE;
    info.oldSwapchain = swapchain_;

    VkSwapchainKHR newSwapchain = VK_NULL_HANDLE;
    VkResult r = vkCreateSwapchainKHR(dev, &info, nullptr, &newSwapchain);
    if (r != VK_SUCCESS) {
        R2_ERROR("vkCreateSwapchainKHR failed: ", vk::resultString(r));
        needsRecreate_ = true;
        return false;
    }
    destroyResources();
    swapchain_ = newSwapchain;
    format_ = chosen.format;
    colorSpace_ = chosen.colorSpace;
    extent_ = extent;
    presentMode_ = presentMode;
    vsync_ = vsync;

    uint32_t count = 0;
    VK_CHECK(vkGetSwapchainImagesKHR(dev, swapchain_, &count, nullptr));
    images_.resize(count);
    VK_CHECK(vkGetSwapchainImagesKHR(dev, swapchain_, &count, images_.data()));
    createImageViews();

    renderFinished_.resize(images_.size(), VK_NULL_HANDLE);
    VkSemaphoreCreateInfo semInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (auto& s : renderFinished_) VK_CHECK(vkCreateSemaphore(dev, &semInfo, nullptr, &s));

    needsRecreate_ = false;
    R2_INFO("swapchain: ", extent_.width, "x", extent_.height, " ", vk::formatName(format_),
            " x", images_.size(), " ",
            presentMode == VK_PRESENT_MODE_FIFO_KHR ? "FIFO(vsync)"
            : presentMode == VK_PRESENT_MODE_MAILBOX_KHR ? "MAILBOX"
                                                         : "IMMEDIATE");
    return true;
}

void Swapchain::createImageViews() {
    views_.resize(images_.size(), VK_NULL_HANDLE);
    for (size_t i = 0; i < images_.size(); ++i) {
        views_[i] = createImageView(*device_, images_[i], format_, VK_IMAGE_ASPECT_COLOR_BIT,
                                    VK_IMAGE_VIEW_TYPE_2D, 1, 1, "swapchain view");
    }
}

bool Swapchain::acquire(uint32_t frameIndex, VkSemaphore signal, uint32_t& imageIndex) {
    if (!swapchain_) return false;
    VkResult r = vkAcquireNextImageKHR(device_->device(), swapchain_, 2000000000ull, signal,
                                       VK_NULL_HANDLE, &imageIndex);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) {
        needsRecreate_ = true;
        return false;
    }
    if (r == VK_TIMEOUT || r == VK_NOT_READY) {
        R2_WARN("vkAcquireNextImageKHR timed out");
        return false;
    }
    if (r != VK_SUCCESS) {
        R2_ERROR("vkAcquireNextImageKHR failed: ", vk::resultString(r));
        needsRecreate_ = true;
        return false;
    }
    return true;
}

bool Swapchain::present(uint32_t frameIndex, uint32_t imageIndex) {
    if (!swapchain_) return false;
    VkPresentInfoKHR info{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &renderFinished_[imageIndex];
    info.swapchainCount = 1;
    info.pSwapchains = &swapchain_;
    info.pImageIndices = &imageIndex;
    VkResult r = vkQueuePresentKHR(device_->graphicsQueue().handle, &info);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) {
        needsRecreate_ = true;
        return false;
    }
    if (r != VK_SUCCESS) {
        R2_ERROR("vkQueuePresentKHR failed: ", vk::resultString(r));
        needsRecreate_ = true;
        return false;
    }
    return true;
}

}  // namespace room2::gfx
