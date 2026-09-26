// room2 - swapchain management with automatic recreation.
#pragma once

#include <vector>

#include "gfx/resources.hpp"

namespace room2::gfx {

class Swapchain {
public:
    Swapchain() = default;
    ~Swapchain();

    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    // `width`/`height` of 0 means "use the surface's current extent".
    bool init(Device& device, uint32_t width, uint32_t height, bool vsync,
              VkFormat preferredFormat = VK_FORMAT_B8G8R8A8_UNORM);
    void shutdown();

    // Recreates the swapchain (after a resize or VK_ERROR_OUT_OF_DATE_KHR).
    bool recreate(uint32_t width, uint32_t height, bool vsync);

    // Acquires the next image. Returns false when the swapchain is out of date and
    // must be recreated (the caller should skip the frame).
    bool acquire(uint32_t frameIndex, VkSemaphore signal, uint32_t& imageIndex);
    // Presents; returns false when the swapchain must be recreated.
    bool present(uint32_t frameIndex, uint32_t imageIndex);

    bool needsRecreate() const { return needsRecreate_; }
    void setNeedsRecreate() { needsRecreate_ = true; }

    // Semaphore that must be signalled by the submission whose image is being
    // presented. It is owned by the swapchain (one per image) because a semaphore
    // waited on by vkQueuePresentKHR must not be re-signalled until the presentation
    // engine has consumed it.
    VkSemaphore renderFinishedSemaphore(uint32_t imageIndex) const {
        return imageIndex < renderFinished_.size() ? renderFinished_[imageIndex] : VK_NULL_HANDLE;
    }

    VkSwapchainKHR handle() const { return swapchain_; }
    VkFormat format() const { return format_; }
    VkColorSpaceKHR colorSpace() const { return colorSpace_; }
    VkExtent2D extent() const { return extent_; }
    uint32_t imageCount() const { return static_cast<uint32_t>(images_.size()); }
    VkImage image(uint32_t i) const { return images_[i]; }
    VkImageView view(uint32_t i) const { return views_[i]; }
    bool vsync() const { return vsync_; }
    VkPresentModeKHR presentMode() const { return presentMode_; }

private:
    void destroyResources();
    void createImageViews();

    Device* device_ = nullptr;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_B8G8R8A8_UNORM;
    VkColorSpaceKHR colorSpace_ = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkExtent2D extent_{0, 0};
    VkPresentModeKHR presentMode_ = VK_PRESENT_MODE_FIFO_KHR;
    bool vsync_ = true;
    VkFormat preferredFormat_ = VK_FORMAT_B8G8R8A8_UNORM;
    std::vector<VkImage> images_;
    std::vector<VkImageView> views_;
    // One "render finished" semaphore per swapchain image so that a frame never
    // waits on a semaphore that is still in flight.
    std::vector<VkSemaphore> renderFinished_;
    bool needsRecreate_ = false;
};

}  // namespace room2::gfx
