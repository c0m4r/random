#include "render/ui.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "core/log.hpp"

namespace room2::render {

// ============================================================ UiDrawList
void UiDrawList::addQuad(Vec2 min, Vec2 max, Vec4 color, Vec2 uvMin, Vec2 uvMax, bool textured) {
    if (!batches.empty() && batches.back().useTexture == textured) {
        batches.back().indexCount += 6;
    } else {
        Batch batch;
        batch.firstIndex = static_cast<uint32_t>(indices.size());
        batch.indexCount = 6;
        batch.useTexture = textured;
        batches.push_back(batch);
    }
    const uint32_t base = static_cast<uint32_t>(vertices.size());
    vertices.push_back({Vec2(min.x, min.y), Vec2(uvMin.x, uvMin.y), color});
    vertices.push_back({Vec2(max.x, min.y), Vec2(uvMax.x, uvMin.y), color});
    vertices.push_back({Vec2(max.x, max.y), Vec2(uvMax.x, uvMax.y), color});
    vertices.push_back({Vec2(min.x, max.y), Vec2(uvMin.x, uvMax.y), color});
    indices.insert(indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
}

void UiDrawList::addTexturedQuad(Vec2 min, Vec2 max, Vec2 uvMin, Vec2 uvMax, Vec4 color) {
    addQuad(min, max, color, uvMin, uvMax, true);
}

// ============================================================ stroke font
namespace {

// Each glyph is a list of polylines inside a 5 x 8 design box (x right, y up,
// baseline at y = 0). A point of (-1,-1) starts a new polyline, which avoids the
// preprocessor's inability to pass a braced comma list as a single macro argument.
// Characters that are not listed render blank.
struct Glyph {
    const float* points;   // x0,y0, x1,y1, ... concatenated, (-1,-1) = polyline break
    uint32_t count;        // number of points
};

#define R2_GLYPH(name, ...)                                                   \
    static const float name##_pts[] = {__VA_ARGS__};                          \
    static const Glyph name{name##_pts, sizeof(name##_pts) / sizeof(float) / 2}

// clang-format off
R2_GLYPH(gA, 0,0, 0,5, 2.5f,8, 5,5, 5,0, -1,-1, 0,3.2f, 5,3.2f);
R2_GLYPH(gB, 0,0, 0,8, 3,8, 3.9f,7.3f, 3.9f,5.2f, 3,4.5f, 0,4.5f, -1,-1, 3,4.5f, 4.3f,3.8f, 4.3f,0.8f, 3,0, 0,0);
R2_GLYPH(gC, 5,6.4f, 4,7.8f, 2.2f,8, 0.7f,6.6f, 0,4, 0.7f,1.4f, 2.2f,0, 4,0.2f, 5,1.6f);
R2_GLYPH(gD, 0,0, 0,8, 2.6f,8, 4.3f,6.6f, 5,4, 4.3f,1.4f, 2.6f,0, 0,0);
R2_GLYPH(gE, 5,8, 0,8, 0,0, 5,0, -1,-1, 0,4.2f, 3.5f,4.2f);
R2_GLYPH(gF, 5,8, 0,8, 0,0, -1,-1, 0,4.2f, 3.4f,4.2f);
R2_GLYPH(gG, 5,6.4f, 4,7.8f, 2.2f,8, 0.7f,6.6f, 0,4, 0.7f,1.4f, 2.2f,0, 4,0.2f, 5,1.6f, 5,3.3f, 2.7f,3.3f);
R2_GLYPH(gH, 0,0, 0,8, -1,-1, 5,0, 5,8, -1,-1, 0,4.2f, 5,4.2f);
R2_GLYPH(gI, 2.5f,0, 2.5f,8, -1,-1, 1.2f,8, 3.8f,8, -1,-1, 1.2f,0, 3.8f,0);
R2_GLYPH(gJ, 4.8f,8, 4.8f,1.7f, 3.7f,0.2f, 2,0, 0.7f,1.1f, 0.4f,2.2f, -1,-1, 2.4f,8, 5,8);
R2_GLYPH(gK, 0,0, 0,8, -1,-1, 5,8, 0.3f,3.3f, -1,-1, 1.5f,4.6f, 5,0);
R2_GLYPH(gL, 0,8, 0,0, 5,0);
R2_GLYPH(gM, 0,0, 0,8, 2.5f,3.4f, 5,8, 5,0);
R2_GLYPH(gN, 0,0, 0,8, 5,0, 5,8);
R2_GLYPH(gO, 2.5f,8, 4.3f,6.6f, 5,4, 4.3f,1.4f, 2.5f,0, 0.7f,1.4f, 0,4, 0.7f,6.6f, 2.5f,8);
R2_GLYPH(gP, 0,0, 0,8, 3.2f,8, 4.5f,7.1f, 4.5f,5, 3.2f,4.2f, 0,4.2f);
R2_GLYPH(gQ, 2.5f,8, 4.3f,6.6f, 5,4, 4.3f,1.4f, 2.5f,0, 0.7f,1.4f, 0,4, 0.7f,6.6f, 2.5f,8, -1,-1, 3.1f,2.1f, 5.3f,-0.6f);
R2_GLYPH(gR, 0,0, 0,8, 3.2f,8, 4.5f,7.1f, 4.5f,5, 3.2f,4.2f, 0,4.2f, -1,-1, 2.4f,4.2f, 5,0);
R2_GLYPH(gS, 5,6.7f, 3.8f,7.9f, 1.8f,8, 0.7f,6.9f, 0.8f,5.6f, 2,4.7f, 3.6f,4.2f, 4.6f,3.2f, 4.4f,1.4f, 3,0.1f, 1.2f,0, 0,1.2f);
R2_GLYPH(gT, 0,8, 5,8, -1,-1, 2.5f,8, 2.5f,0);
R2_GLYPH(gU, 0,8, 0,1.8f, 1.4f,0.1f, 3.6f,0.1f, 5,1.8f, 5,8);
R2_GLYPH(gV, 0,8, 2.5f,0, 5,8);
R2_GLYPH(gW, 0,8, 1,0, 2.5f,5, 4,0, 5,8);
R2_GLYPH(gX, 0,0, 5,8, -1,-1, 0,8, 5,0);
R2_GLYPH(gY, 0,8, 2.5f,4.2f, 5,8, -1,-1, 2.5f,4.2f, 2.5f,0);
R2_GLYPH(gZ, 0,8, 5,8, 0,0, 5,0);
R2_GLYPH(g0, 2.5f,8, 4.3f,6.6f, 5,4, 4.3f,1.4f, 2.5f,0, 0.7f,1.4f, 0,4, 0.7f,6.6f, 2.5f,8, -1,-1, 0.9f,1.7f, 4.1f,6.3f);
R2_GLYPH(g1, 1.1f,6.3f, 2.8f,8, 2.8f,0, -1,-1, 1,0, 4.6f,0);
R2_GLYPH(g2, 0.2f,6.4f, 1.2f,7.8f, 3.4f,8, 4.7f,6.7f, 4.5f,5.2f, 0.2f,0, 5,0);
R2_GLYPH(g3, 0.3f,7, 1.6f,8, 3.6f,7.9f, 4.6f,6.7f, 4.2f,5.2f, 2.6f,4.4f, 4.4f,3.7f, 4.9f,2.2f, 3.8f,0.4f, 1.6f,0, 0.3f,1.1f);
R2_GLYPH(g4, 3.8f,0, 3.8f,8, 0.2f,2.6f, 5,2.6f);
R2_GLYPH(g5, 4.7f,8, 0.6f,8, 0.4f,4.6f, 2.4f,5.2f, 4,4.6f, 4.9f,3, 4.4f,1.2f, 3,0.1f, 1.2f,0.2f, 0.2f,1.2f);
R2_GLYPH(g6, 4.5f,6.9f, 3.4f,8, 1.9f,7.9f, 0.7f,6.3f, 0.2f,3.6f, 0.6f,1.4f, 2,0.1f, 3.6f,0.3f, 4.8f,1.6f, 4.7f,3.3f, 3.5f,4.5f, 1.9f,4.4f, 0.6f,3.2f);
R2_GLYPH(g7, 0.2f,8, 5,8, 1.9f,0);
R2_GLYPH(g8, 2.5f,4.4f, 1.1f,5.2f, 1,6.9f, 2.3f,8, 3.7f,7.9f, 4.6f,6.6f, 4.4f,5.2f, 3.1f,4.4f, 1.6f,4.3f, 0.5f,3.3f, 0.4f,1.4f, 1.8f,0.1f, 3.5f,0.2f, 4.8f,1.5f, 4.6f,3.3f, 3.4f,4.4f);
R2_GLYPH(g9, 4.6f,4.9f, 3.4f,3.6f, 1.8f,3.5f, 0.6f,4.7f, 0.5f,6.5f, 1.7f,7.9f, 3.4f,8, 4.5f,6.8f, 4.9f,4.5f, 4.4f,1.8f, 3.2f,0.2f, 1.6f,0.1f, 0.5f,1.1f);
R2_GLYPH(gDot, 2.4f,0.45f, 2.6f,0.45f);
R2_GLYPH(gComma, 2.7f,0.4f, 2.2f,-1.2f, 1.4f,-1.8f);
R2_GLYPH(gColon, 2.5f,0.5f, 2.5f,0.7f, -1,-1, 2.5f,4.6f, 2.5f,4.8f);
R2_GLYPH(gMinus, 0.6f,4, 4.4f,4);
R2_GLYPH(gPlus, 0.7f,4.2f, 4.3f,4.2f, -1,-1, 2.5f,1.8f, 2.5f,6.6f);
R2_GLYPH(gEquals, 0.7f,5.4f, 4.3f,5.4f, -1,-1, 0.7f,3, 4.3f,3);
R2_GLYPH(gSlash, 0.4f,0, 4.6f,8);
R2_GLYPH(gBackslash, 0.4f,8, 4.6f,0);
R2_GLYPH(gBang, 2.5f,8, 2.5f,2.2f, -1,-1, 2.5f,0.45f, 2.5f,0.6f);
R2_GLYPH(gQuestion, 0.5f,6.5f, 1.5f,7.9f, 3.4f,8, 4.6f,6.8f, 4.4f,5.4f, 2.6f,4.4f, 2.6f,2.4f, -1,-1, 2.6f,0.45f, 2.6f,0.6f);
R2_GLYPH(gPercent, 0.4f,0, 4.6f,8, -1,-1, 0.5f,6.1f, 1.5f,7.1f, 2.5f,6.1f, 1.5f,5.1f, 0.5f,6.1f, -1,-1, 2.5f,1.9f, 3.5f,2.9f, 4.5f,1.9f, 3.5f,0.9f, 2.5f,1.9f);
R2_GLYPH(gParenL, 3.6f,8.6f, 2.2f,6.6f, 1.7f,4, 2.2f,1.4f, 3.6f,-0.6f);
R2_GLYPH(gParenR, 1.4f,8.6f, 2.8f,6.6f, 3.3f,4, 2.8f,1.4f, 1.4f,-0.6f);
R2_GLYPH(gBracketL, 3.6f,8.4f, 2,8.4f, 2,-0.4f, 3.6f,-0.4f);
R2_GLYPH(gBracketR, 1.4f,8.4f, 3,8.4f, 3,-0.4f, 1.4f,-0.4f);
R2_GLYPH(gUnderscore, 0,-1.4f, 5,-1.4f);
R2_GLYPH(gQuote, 1.8f,8, 1.6f,6.4f, -1,-1, 3.4f,8, 3.2f,6.4f);
R2_GLYPH(gStar, 2.5f,1.6f, 2.5f,7.4f, -1,-1, 0.5f,3, 4.5f,6, -1,-1, 0.5f,6, 4.5f,3);
R2_GLYPH(gLess, 4.4f,7, 0.6f,4, 4.4f,1);
R2_GLYPH(gGreater, 0.6f,7, 4.4f,4, 0.6f,1);
R2_GLYPH(gSpace, -1,-1);
// clang-format on
#undef R2_GLYPH

const Glyph* glyphFor(char c) {
    switch (c) {
        case 'A': return &gA; case 'B': return &gB; case 'C': return &gC; case 'D': return &gD;
        case 'E': return &gE; case 'F': return &gF; case 'G': return &gG; case 'H': return &gH;
        case 'I': return &gI; case 'J': return &gJ; case 'K': return &gK; case 'L': return &gL;
        case 'M': return &gM; case 'N': return &gN; case 'O': return &gO; case 'P': return &gP;
        case 'Q': return &gQ; case 'R': return &gR; case 'S': return &gS; case 'T': return &gT;
        case 'U': return &gU; case 'V': return &gV; case 'W': return &gW; case 'X': return &gX;
        case 'Y': return &gY; case 'Z': return &gZ;
        case 'a': return &gA; case 'b': return &gB; case 'c': return &gC; case 'd': return &gD;
        case 'e': return &gE; case 'f': return &gF; case 'g': return &gG; case 'h': return &gH;
        case 'i': return &gI; case 'j': return &gJ; case 'k': return &gK; case 'l': return &gL;
        case 'm': return &gM; case 'n': return &gN; case 'o': return &gO; case 'p': return &gP;
        case 'q': return &gQ; case 'r': return &gR; case 's': return &gS; case 't': return &gT;
        case 'u': return &gU; case 'v': return &gV; case 'w': return &gW; case 'x': return &gX;
        case 'y': return &gY; case 'z': return &gZ;
        case '0': return &g0; case '1': return &g1; case '2': return &g2; case '3': return &g3;
        case '4': return &g4; case '5': return &g5; case '6': return &g6; case '7': return &g7;
        case '8': return &g8; case '9': return &g9;
        case '.': return &gDot; case ',': return &gComma; case ':': return &gColon;
        case ';': return &gColon; case '-': return &gMinus; case '+': return &gPlus;
        case '=': return &gEquals; case '/': return &gSlash; case '\\': return &gBackslash;
        case '!': return &gBang; case '?': return &gQuestion; case '%': return &gPercent;
        case '(': return &gParenL; case ')': return &gParenR;
        case '[': return &gBracketL; case ']': return &gBracketR;
        case '_': return &gUnderscore; case '\'': return &gQuote; case '"': return &gQuote;
        case '*': return &gStar; case '<': return &gLess; case '>': return &gGreater;
        case ' ': return &gSpace;
        default: return nullptr;
    }
}

// Atlas layout: ASCII 32..126 in a 16 x 6 grid of cells.
constexpr uint32_t kCellW = 34;
constexpr uint32_t kCellH = 46;
constexpr uint32_t kGridCols = 16;
constexpr uint32_t kGridRows = 6;
constexpr uint32_t kAtlasW = kCellW * kGridCols;
constexpr uint32_t kAtlasH = kCellH * kGridRows;
constexpr float kDesignW = 5.0f;
constexpr float kDesignH = 8.0f;
constexpr float kPaddingX = 3.0f;
constexpr float kPaddingY = 6.0f;
constexpr float kStrokeHalfWidth = 0.42f;   // in design units

float distanceToSegment(float px, float py, float ax, float ay, float bx, float by) {
    const float dx = bx - ax, dy = by - ay;
    const float lenSq = dx * dx + dy * dy;
    float t = 0.0f;
    if (lenSq > 1e-9f) t = clamp(((px - ax) * dx + (py - ay) * dy) / lenSq, 0.0f, 1.0f);
    const float cx = ax + t * dx, cy = ay + t * dy;
    return std::sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy));
}

std::vector<uint8_t> rasteriseAtlas() {
    std::vector<uint8_t> atlas(static_cast<size_t>(kAtlasW) * kAtlasH, 0);
    const float scaleX = (kCellW - kPaddingX * 2.0f) / kDesignW;
    const float scaleY = (kCellH - kPaddingY * 2.0f) / kDesignH;
    const float scale = std::min(scaleX, scaleY);
    const float offsetX = (kCellW - kDesignW * scale) * 0.5f;
    const float offsetY = kCellH - kPaddingY;

    for (uint32_t code = 32; code <= 126; ++code) {
        const Glyph* glyph = glyphFor(static_cast<char>(code));
        const uint32_t cell = code - 32;
        const uint32_t cx = (cell % kGridCols) * kCellW;
        const uint32_t cy = (cell / kGridCols) * kCellH;

        // Supersample 3x3 for smooth edges.
        const int ss = 3;
        for (uint32_t y = 0; y < kCellH; ++y) {
            for (uint32_t x = 0; x < kCellW; ++x) {
                float coverage = 0.0f;
                for (int sy = 0; sy < ss; ++sy) {
                    for (int sx = 0; sx < ss; ++sx) {
                        const float fx = (static_cast<float>(x) + (sx + 0.5f) / ss - offsetX) / scale;
                        const float fy = (offsetY - (static_cast<float>(y) + (sy + 0.5f) / ss)) /
                                         scale;
                        float best = 1e9f;
                        if (glyph && glyph->count > 0) {
                            for (uint32_t p = 0; p + 1 < glyph->count; ++p) {
                                const float ax = glyph->points[p * 2];
                                const float ay = glyph->points[p * 2 + 1];
                                const float bx = glyph->points[(p + 1) * 2];
                                const float by = glyph->points[(p + 1) * 2 + 1];
                                // (-1,-1) marks a polyline break, not a real segment.
                                if (ax < -0.5f || bx < -0.5f) continue;
                                best = std::min(best, distanceToSegment(fx, fy, ax, ay, bx, by));
                            }
                        }
                        const float edge = kStrokeHalfWidth;
                        const float c = clamp((edge + 0.35f - best) / 0.7f, 0.0f, 1.0f);
                        coverage += c;
                    }
                }
                coverage /= static_cast<float>(ss * ss);
                atlas[static_cast<size_t>(cy + y) * kAtlasW + (cx + x)] =
                    static_cast<uint8_t>(clamp(coverage, 0.0f, 1.0f) * 255.0f + 0.5f);
            }
        }
    }
    return atlas;
}

}  // namespace

// ============================================================ UiContext
bool UiContext::init(gfx::Device& device) {
    device_ = &device;
    std::vector<uint8_t> data = rasteriseAtlas();

    gfx::ImageDesc desc;
    desc.format = VK_FORMAT_R8_UNORM;
    desc.extent = {kAtlasW, kAtlasH, 1};
    desc.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    desc.name = "font atlas";
    atlas_ = gfx::createImage(device, desc);

    gfx::UploadBatch batch(device);
    batch.image(atlas_, data.data(), data.size());
    batch.submit("font atlas");

    sampler_ = gfx::createSampler(device, VK_FILTER_LINEAR, VK_FILTER_LINEAR,
                                  VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                  VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 1.0f, false,
                                  VK_COMPARE_OP_LESS_OR_EQUAL, 0.0f, 0.0f, false,
                                  VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE, "font sampler");
    R2_INFO("UI font atlas: ", kAtlasW, "x", kAtlasH, " (", kGridCols, "x", kGridRows, " cells)");
    return true;
}

void UiContext::shutdown() {
    if (device_ && sampler_) vkDestroySampler(device_->device(), sampler_, nullptr);
    sampler_ = VK_NULL_HANDLE;
    if (device_) gfx::destroyImage(*device_, atlas_);
    device_ = nullptr;
}

void UiContext::rect(Vec2 min, Vec2 max, Vec4 color) {
    list_.addQuad(Vec2(std::round(min.x), std::round(min.y)),
                  Vec2(std::round(max.x), std::round(max.y)), color, Vec2(0.5f, 0.5f),
                  Vec2(0.5f, 0.5f), false);
}

void UiContext::rectOutline(Vec2 min, Vec2 max, float thickness, Vec4 color) {
    rect(min, Vec2(max.x, min.y + thickness), color);
    rect(Vec2(min.x, max.y - thickness), max, color);
    rect(min, Vec2(min.x + thickness, max.y), color);
    rect(Vec2(max.x - thickness, min.y), max, color);
}

void UiContext::line(Vec2 a, Vec2 b, float thickness, Vec4 color) {
    const Vec2 delta = b - a;
    const float len = length(delta);
    if (len < 1e-4f) return;
    const Vec2 n = Vec2(-delta.y, delta.x) / len * (thickness * 0.5f);
    const uint32_t base = static_cast<uint32_t>(list_.vertices.size());
    const Vec4 uv(0.5f, 0.5f, 0, 0);
    list_.vertices.push_back({a + n, Vec2(uv.x, uv.y), color});
    list_.vertices.push_back({b + n, Vec2(uv.x, uv.y), color});
    list_.vertices.push_back({b - n, Vec2(uv.x, uv.y), color});
    list_.vertices.push_back({a - n, Vec2(uv.x, uv.y), color});
    if (!list_.batches.empty() && !list_.batches.back().useTexture) {
        list_.batches.back().indexCount += 6;
    } else {
        UiDrawList::Batch batch;
        batch.firstIndex = static_cast<uint32_t>(list_.indices.size());
        batch.indexCount = 6;
        batch.useTexture = false;
        list_.batches.push_back(batch);
    }
    list_.indices.insert(list_.indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
}

// `scale` is the rendered cap height in eighths of the design box, so scale 1.0 draws
// 8-pixel-tall capitals. The atlas cell is taller than the design box (descenders and
// padding), so the quad is sized to keep the caps at exactly 8 * scale pixels.
void UiContext::text(Vec2 topLeft, const std::string& str, Vec4 color, float scale) {
    constexpr float kCapsInCell = 34.0f / static_cast<float>(kCellH);   // ink height fraction
    const float quadH = (8.0f * scale) / kCapsInCell;
    const float quadW = quadH * static_cast<float>(kCellW) / static_cast<float>(kCellH);
    const float advance = quadW;
    float pen = std::round(topLeft.x);
    const float top = std::round(topLeft.y);
    for (char c : str) {
        const int code = static_cast<int>(static_cast<unsigned char>(c));
        if (code >= 32 && code <= 126) {
            const uint32_t cell = static_cast<uint32_t>(code - 32);
            const float u0 = static_cast<float>((cell % kGridCols) * kCellW) / kAtlasW;
            const float v0 = static_cast<float>((cell / kGridCols) * kCellH) / kAtlasH;
            const float u1 = u0 + static_cast<float>(kCellW) / kAtlasW;
            const float v1 = v0 + static_cast<float>(kCellH) / kAtlasH;
            list_.addTexturedQuad(Vec2(pen, top), Vec2(pen + quadW, top + quadH),
                                  Vec2(u0, v0), Vec2(u1, v1), color);
        }
        pen += advance;
    }
}

void UiContext::textCentered(Vec2 center, const std::string& str, Vec4 color, float scale) {
    const float w = textWidth(str, scale);
    text(Vec2(center.x - w * 0.5f, center.y - textHeight(scale) * 0.5f), str, color, scale);
}

void UiContext::textRight(Vec2 rightCenter, const std::string& str, Vec4 color, float scale) {
    const float w = textWidth(str, scale);
    text(Vec2(rightCenter.x - w, rightCenter.y - textHeight(scale) * 0.5f), str, color, scale);
}

float UiContext::textWidth(const std::string& str, float scale) const {
    constexpr float kCapsInCell = 34.0f / static_cast<float>(kCellH);
    const float quadH = (8.0f * scale) / kCapsInCell;
    return static_cast<float>(str.size()) * quadH * static_cast<float>(kCellW) /
           static_cast<float>(kCellH);
}

void UiContext::crosshair(Vec2 center, float radius, float gap, float thickness, Vec4 color) {
    const float cx = std::round(center.x), cy = std::round(center.y);
    const float t = std::max(1.0f, thickness);
    // Four ticks with a clear centre.
    rect(Vec2(cx - radius, cy - t * 0.5f), Vec2(cx - gap, cy + t * 0.5f), color);
    rect(Vec2(cx + gap, cy - t * 0.5f), Vec2(cx + radius, cy + t * 0.5f), color);
    rect(Vec2(cx - t * 0.5f, cy - radius), Vec2(cx + t * 0.5f, cy - gap), color);
    rect(Vec2(cx - t * 0.5f, cy + gap), Vec2(cx + t * 0.5f, cy + radius), color);
    rect(Vec2(cx - 1.0f, cy - 1.0f), Vec2(cx + 1.0f, cy + 1.0f), color);
}

}  // namespace room2::render
