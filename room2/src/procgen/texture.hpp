// room2 - procedural texture generation.
// All textures are synthesised on the CPU at load time (physically-based channels:
// base colour, tangent-space normal, and a packed occlusion/roughness/metallic map).
// Everything is tileable so the room surfaces can repeat without visible seams.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/math.hpp"

namespace room2::procgen {

// Linear-light RGBA float image. Base-colour data must already be linear; the
// uploader writes it into an 8-bit sRGB image (values are converted on pack).
struct TextureData {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<float> pixels;   // width*height*4, RGBA

    void resize(uint32_t w, uint32_t h, Vec4 fill = Vec4(0, 0, 0, 1)) {
        width = w;
        height = h;
        pixels.assign(static_cast<size_t>(w) * h * 4, 0.0f);
        for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
            pixels[i * 4 + 0] = fill.x;
            pixels[i * 4 + 1] = fill.y;
            pixels[i * 4 + 2] = fill.z;
            pixels[i * 4 + 3] = fill.w;
        }
    }
    bool valid() const { return width > 0 && height > 0 && pixels.size() == size_t(width) * height * 4; }
    Vec4 texel(uint32_t x, uint32_t y) const {
        const float* p = &pixels[(static_cast<size_t>(y) * width + x) * 4];
        return {p[0], p[1], p[2], p[3]};
    }
    void setTexel(uint32_t x, uint32_t y, Vec4 c) {
        float* p = &pixels[(static_cast<size_t>(y) * width + x) * 4];
        p[0] = c.x; p[1] = c.y; p[2] = c.z; p[3] = c.w;
    }
    // Wraps out-of-range coordinates.
    Vec4 sample(int x, int y) const {
        int w = static_cast<int>(width), h = static_cast<int>(height);
        x = ((x % w) + w) % w;
        y = ((y % h) + h) % h;
        return texel(static_cast<uint32_t>(x), static_cast<uint32_t>(y));
    }
    Vec4 bilinear(float u, float v) const;
};

// One material's worth of maps.
struct TextureSet {
    TextureData baseColor;   // linear RGB, A = 1
    TextureData normal;      // RGB = tangent-space normal encoded to [0,1], A = 1
    TextureData orm;         // R = ambient occlusion, G = roughness, B = metallic
    bool hasNormal = false;
    bool hasOrm = false;

    uint32_t width() const { return baseColor.width; }
    uint32_t height() const { return baseColor.height; }
    // Frees the pixel data of maps that are not needed.
    void dropNormal() {
        normal = TextureData{};
        hasNormal = false;
    }
};

// Fills a flat normal map (0.5, 0.5, 1.0) of the given size.
TextureData flatNormal(uint32_t size);
// Encodes a height field (in metres, or any consistent unit) into a tangent-space
// normal map using central differences. `strength` scales the slope.
TextureData normalFromHeight(const std::vector<float>& height, uint32_t width, uint32_t heightPx,
                             float strength);
// Derives an AO map from a height field using a cheap horizon-based approximation.
TextureData aoFromHeight(const std::vector<float>& height, uint32_t width, uint32_t heightPx,
                         float strength, int radius = 8);
// Builds the packed ORM map from separate float buffers (each width*height).
TextureData packOrm(const std::vector<float>& ao, const std::vector<float>& roughness,
                    const std::vector<float>& metallic, uint32_t width, uint32_t heightPx);
// Converts an image to 8-bit sRGB RGBA bytes (for the base-colour map).
std::vector<uint8_t> toSrgb8(const TextureData& src);
// Converts an image to 8-bit linear RGBA bytes (for normal / ORM maps).
std::vector<uint8_t> toLinear8(const TextureData& src);
// Downsamples by repeated 2x box filtering (for building mip chains on the CPU).
TextureData downsample(const TextureData& src);

// ---------------------------------------------------------------- materials
// Every generator returns a tileable set at `size` x `size` texels. `texelsPerMetre`
// tells the generator how the texture will be mapped so detail is physically scaled.
struct SurfaceParams {
    uint32_t seed = 1234;
    float texelsPerMetre = 512.0f;   // mapping density used for feature sizing
};

TextureSet plasterWall(uint32_t size, const SurfaceParams& params = {});
TextureSet paintedCeiling(uint32_t size, const SurfaceParams& params = {});
TextureSet concreteFloor(uint32_t size, const SurfaceParams& params = {});
TextureSet woodPlanks(uint32_t size, const SurfaceParams& params = {});
TextureSet woodTableTop(uint32_t size, const SurfaceParams& params = {});
TextureSet varnishedWood(uint32_t size, const SurfaceParams& params = {});
// Blued/oxidised steel, as used on the HK USP slide and barrel.
TextureSet bluedSteel(uint32_t size, const SurfaceParams& params = {});
// Black polymer frame with moulded stippling.
TextureSet blackPolymer(uint32_t size, const SurfaceParams& params = {});
// Rubberised grip panels with a chequered pattern.
TextureSet gripPanel(uint32_t size, const SurfaceParams& params = {});
TextureSet brass(uint32_t size, const SurfaceParams& params = {});
// Almost perfectly smooth glass with faint smudges and micro-scratches.
TextureSet glassSurface(uint32_t size, const SurfaceParams& params = {});
TextureSet paintedMetalWhite(uint32_t size, const SurfaceParams& params = {});
TextureSet lampEmissive(uint32_t size, const SurfaceParams& params = {});
TextureSet fabric(uint32_t size, const SurfaceParams& params = {});
TextureSet cardboard(uint32_t size, const SurfaceParams& params = {});

// ---------------------------------------------------------------- utility
// Writes a TextureData to a PNG (used by the --dump-textures verification mode).
// Returns false on I/O failure. Only RGBA8 PNGs are written (no external deps).
bool writePng(const std::string& path, const TextureData& image, bool srgb = true);
// Writes an 8-bit grey PNG from a float buffer.
bool writePngGrey(const std::string& path, const std::vector<float>& data, uint32_t width,
                  uint32_t heightPx);

}  // namespace room2::procgen
