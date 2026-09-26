// room2 - immediate-mode UI overlay: HUD text, crosshair and panels.
// Text is drawn from a procedurally rasterised single-channel font atlas, so the
// project still ships no asset files.
#pragma once

#include <string>
#include <vector>

#include "gfx/resources.hpp"
#include "render/renderer.hpp"

namespace room2::render {

class UiContext {
public:
    bool init(gfx::Device& device);
    void shutdown();

    // The font atlas (R8) is exposed so the renderer can bind it to the UI pass.
    const gfx::Image& atlas() const { return atlas_; }

    // --- drawing (pixel coordinates, origin at the top-left) --------------
    void clear() { list_.clear(); }
    void rect(Vec2 min, Vec2 max, Vec4 color);
    void rectOutline(Vec2 min, Vec2 max, float thickness, Vec4 color);
    void line(Vec2 a, Vec2 b, float thickness, Vec4 color);
    void text(Vec2 topLeft, const std::string& text, Vec4 color, float scale = 1.0f);
    void textCentered(Vec2 center, const std::string& text, Vec4 color, float scale = 1.0f);
    void textRight(Vec2 rightCenter, const std::string& text, Vec4 color, float scale = 1.0f);
    void crosshair(Vec2 center, float radius, float gap, float thickness, Vec4 color);

    float textWidth(const std::string& text, float scale) const;
    // Cap height of a glyph drawn at the given scale.
    float textHeight(float scale) const { return 8.0f * scale; }
    float lineHeight(float scale) const { return 13.0f * scale; }

    UiDrawList& list() { return list_; }
    const UiDrawList& list() const { return list_; }

private:
    UiDrawList list_;
    gfx::Device* device_ = nullptr;
    gfx::Image atlas_;
    VkSampler sampler_ = VK_NULL_HANDLE;
};

}  // namespace room2::render
