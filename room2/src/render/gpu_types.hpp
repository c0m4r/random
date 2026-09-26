// room2 - structures shared between C++ and the GLSL shaders.
//
// IMPORTANT: every member of every struct below is 16-byte aligned (vec4 / mat4 /
// uvec4 only). This keeps std430 and std140 layouts identical, so the same struct
// can be bound as either a uniform or a storage buffer without layout surprises.
// `shaders/common.glsl` mirrors these definitions exactly - keep them in sync.
#pragma once

#include <cstdint>

#include "core/math.hpp"

namespace room2::render {

// ---------------------------------------------------------------- material
struct GpuMaterial {
    Vec4 baseColorFactor{1, 1, 1, 1};
    Vec4 emissiveFactor{0, 0, 0, 0};      // rgb = emissive, a = unused
    Vec4 attenuationColor{1, 1, 1, 0};    // rgb = tint, a = thickness (metres)
    // x = base colour, y = normal, z = ORM, w = emissive texture slots.
    uint32_t texBaseColor = 0;
    uint32_t texNormal = 1;
    uint32_t texOrm = 0;
    uint32_t texEmissive = 0xFFFFFFFFu;
    // x = metallic, y = roughness, z = normalScale, w = occlusionStrength
    Vec4 params0{0.0f, 0.5f, 1.0f, 1.0f};
    // x = transmission, y = ior, z = alpha, w = materialKind (0 opaque, 1 glass)
    Vec4 params1{0.0f, 1.5f, 1.0f, 0.0f};
};
static_assert(sizeof(GpuMaterial) == 96, "GpuMaterial must stay 16-byte aligned");

// ---------------------------------------------------------------- instance
struct GpuInstance {
    Mat4 model;
    Mat4 normalMatrix;
    // x = material index, y = userData, z = flags, w = unused
    uint32_t materialIndex = 0;
    uint32_t userData = 0;
    uint32_t flags = 0;
    uint32_t pad = 0;
};
static_assert(sizeof(GpuInstance) == 144, "GpuInstance must stay 16-byte aligned");

enum InstanceFlags : uint32_t {
    INSTANCE_CASTS_SHADOW = 1u << 0,
    INSTANCE_TRANSPARENT = 1u << 1,
    INSTANCE_DYNAMIC = 1u << 2,
    INSTANCE_EMISSIVE = 1u << 3,
};

// ---------------------------------------------------------------- light
inline constexpr uint32_t kMaxLights = 8;
// Shadow-casting lights use a single 2D perspective map (a spot frustum), which avoids
// all cube-map face conventions and gives better texel density for the room.
inline constexpr uint32_t kMaxShadowLights = 2;
inline constexpr uint32_t kShadowCubeFaces = 1;   // retained for layout compatibility

struct GpuLight {
    Vec4 positionRange{0, 0, 0, 20.0f};      // xyz = position, w = range
    Vec4 colorIntensity{1, 1, 1, 1};         // rgb = colour, a = intensity
    Vec4 directionRadius{0, -1, 0, 0.05f};   // xyz = direction, w = radius
    Vec4 shadowParams{0, 0, 0, 1};           // x = innerCos, y = outerCos, z = type, w = shadowIndex+1
    Vec4 shadowNearFar{0.05f, 30.0f, 0.0f, 0.0f};
    // Shadow view-projection; only entry 0 is used now that shadows are 2D spot maps.
    Mat4 shadowViewProj[6];
};
static_assert(sizeof(GpuLight) == 64 * 6 + 80, "GpuLight layout drift");

// ---------------------------------------------------------------- globals
// Per-frame uniform data, bound as a storage buffer at set 0 binding 0.
struct GpuGlobals {
    Mat4 view;
    Mat4 projection;
    Mat4 viewProjection;
    Mat4 invViewProjection;
    Mat4 prevViewProjection;
    Vec4 cameraPosition{0, 0, 0, 1};     // xyz = world position, w = 1/far
    Vec4 viewportSize{1920, 1080, 1.0f / 1920.0f, 1.0f / 1080.0f};
    Vec4 envCeiling{0.35f, 0.34f, 0.32f, 0.35f};   // rgb = bounce colour, a = intensity
    Vec4 envWall{0.22f, 0.20f, 0.17f, 0.0f};
    Vec4 envFloor{0.10f, 0.07f, 0.05f, 0.0f};
    Vec4 ambientTint{1.0f, 0.97f, 0.92f, 0.0f};
    // x = time seconds, y = frame index, z = light count, w = exposure
    Vec4 misc{0, 0, 0, 1.0f};
    // x = jitter x, y = jitter y, z = previous jitter x, w = previous jitter y
    Vec4 jitter{0, 0, 0, 0};
    // x = near, y = far, z = shadow texel size, w = shadow bias
    Vec4 shadowParams{0.05f, 30.0f, 1.0f / 1024.0f, 0.0015f};
    // x = ao strength, y = bloom strength, z = taa blend, w = fovY
    Vec4 postParams{1.0f, 0.06f, 0.1f, 1.0f};
    Vec4 renderSize{1920, 1080, 1.0f / 1920.0f, 1.0f / 1080.0f};
};

// ---------------------------------------------------------------- shader constants
inline constexpr uint32_t kDescriptorSetGlobals = 0;
inline constexpr uint32_t kDescriptorSetTextures = 1;
inline constexpr uint32_t kDescriptorSetPass = 2;

inline constexpr uint32_t kBindingGlobals = 0;
inline constexpr uint32_t kBindingInstances = 1;
inline constexpr uint32_t kBindingMaterials = 2;
inline constexpr uint32_t kBindingLights = 3;
inline constexpr uint32_t kBindingBindlessTextures = 0;
inline constexpr uint32_t kMaxBindlessTextures = 512;

}  // namespace room2::render
