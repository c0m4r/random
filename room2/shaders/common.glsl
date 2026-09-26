// room2 - shared GLSL declarations. Mirrors src/render/gpu_types.hpp exactly.
#ifndef ROOM2_COMMON_GLSL
#define ROOM2_COMMON_GLSL

#extension GL_EXT_nonuniform_qualifier : require

const float R2_PI = 3.141592653589793;
const float R2_INV_PI = 0.3183098861837907;
const float R2_EPS = 1e-6;

// ---------------------------------------------------------------- structs
struct GpuMaterial {
    vec4 baseColorFactor;
    vec4 emissiveFactor;
    vec4 attenuationColor;   // rgb tint, a = thickness
    uvec4 texIndices;        // baseColor, normal, orm, emissive
    vec4 params0;            // metallic, roughness, normalScale, occlusionStrength
    vec4 params1;            // transmission, ior, alpha, materialKind
};

struct GpuInstance {
    mat4 model;
    mat4 normalMatrix;
    uvec4 misc;              // materialIndex, userData, flags, pad
};

struct GpuLight {
    vec4 positionRange;
    vec4 colorIntensity;
    vec4 directionRadius;
    vec4 shadowParams;       // innerCos, outerCos, type, shadowIndex+1
    vec4 shadowNearFar;
    mat4 shadowViewProj[6];
};

struct GpuGlobals {
    mat4 view;
    mat4 projection;
    mat4 viewProjection;
    mat4 invViewProjection;
    mat4 prevViewProjection;
    vec4 cameraPosition;
    vec4 viewportSize;
    vec4 envCeiling;
    vec4 envWall;
    vec4 envFloor;
    vec4 ambientTint;
    vec4 misc;               // time, frameIndex, lightCount, exposure
    vec4 jitter;             // jitter.xy, prevJitter.xy
    vec4 shadowParams;       // near, far, texelSize, bias
    vec4 postParams;         // aoStrength, bloomStrength, taaBlend, fovY
    vec4 renderSize;
};

// ---------------------------------------------------------------- bindings
layout(set = 0, binding = 0, std140) uniform GlobalsBlock {
    GpuGlobals g;
};

layout(set = 0, binding = 1, std430) readonly buffer InstanceBlock {
    GpuInstance instances[];
};

layout(set = 0, binding = 2, std430) readonly buffer MaterialBlock {
    GpuMaterial materials[];
};

layout(set = 0, binding = 3, std430) readonly buffer LightBlock {
    GpuLight lights[];
};

#define R2_INSTANCE_FLAG_SHADOW 1u
#define R2_INSTANCE_FLAG_TRANSPARENT 2u
#define R2_INSTANCE_FLAG_DYNAMIC 4u
#define R2_INSTANCE_FLAG_EMISSIVE 8u

// ---------------------------------------------------------------- math helpers
float r2_luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

float r2_saturate(float x) { return clamp(x, 0.0, 1.0); }
vec3 r2_saturate(vec3 x) { return clamp(x, vec3(0.0), vec3(1.0)); }

// Reconstructs the world-space position from a depth buffer value.
//
// This deliberately avoids a matrix inverse: the view matrix already carries the
// camera basis, and with the projection's known shape the ray through a pixel can be
// rebuilt from the basis, the field of view, the aspect ratio and the near/far range.
// That is exact, cheaper, and immune to any inverse-projection convention mismatch.
// `viewDepth` is the LINEAR distance along the camera's view axis (in metres), which
// the G-buffer stores directly. Reconstructing it from the [0,1] depth buffer instead
// would amplify any depth-buffer noise by ~(far/near) and is not numerically usable.
vec3 r2_worldFromDepth(vec2 uv, float viewDepth) {
    float tanHalf = tan(g.postParams.w * 0.5);
    float aspect = g.viewportSize.x / max(g.viewportSize.y, 1.0);
    vec2 ndc = uv * 2.0 - 1.0;
    // Rows of the view matrix are the camera basis vectors in world space.
    vec3 right = vec3(g.view[0].x, g.view[1].x, g.view[2].x);
    vec3 up = vec3(g.view[0].y, g.view[1].y, g.view[2].y);
    vec3 forward = -vec3(g.view[0].z, g.view[1].z, g.view[2].z);
    // Vulkan's clip-space Y is inverted relative to the view-space Y axis.
    vec3 dir = normalize(right * (ndc.x * tanHalf * aspect) + up * (-ndc.y * tanHalf) + forward);
    float along = dot(dir, forward);
    return g.cameraPosition.xyz + dir * (viewDepth / max(along, 1e-6));
}

// Linearises a [0,1] Vulkan depth value into view-space distance. Only used where the
// G-buffer's linear depth is unavailable (for example on a shadow map).
float r2_linearDepth(float depth, float nearPlane, float farPlane) {
    return (nearPlane * farPlane) / max(farPlane - depth * (farPlane - nearPlane), 1e-9);
}

// Octahedral normal encoding (2 x 16 bit is plenty; we store in a float target).
vec2 r2_octEncode(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z) + 1e-9);
    vec2 e = n.xy;
    if (n.z < 0.0) {
        e = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    }
    return e;
}
vec3 r2_octDecode(vec2 e) {
    vec3 n = vec3(e.xy, 1.0 - abs(e.x) - abs(e.y));
    float t = max(-n.z, 0.0);
    n.x += n.x >= 0.0 ? -t : t;
    n.y += n.y >= 0.0 ? -t : t;
    return normalize(n);
}

// ---------------------------------------------------------------- colour
// ACES filmic tone mapping (Narkowicz's fit, sufficient for a real-time pipeline).
vec3 r2_aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return r2_saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

vec3 r2_acesFitted(vec3 color) {
    // Stephen Hill's fit with the ACES RRT/ODT matrices.
    const mat3 inputMatrix = mat3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777);
    const mat3 outputMatrix = mat3(
         1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602);
    color = inputMatrix * color;
    vec3 a = color * (color + 0.0245786) - 0.000090537;
    vec3 b = color * (0.983729 * color + 0.4329510) + 0.238081;
    color = a / b;
    return r2_saturate(outputMatrix * color);
}

vec3 r2_srgbEncode(vec3 c) {
    return mix(c * 12.92, 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055,
               step(vec3(0.0031308), c));
}
vec3 r2_srgbDecode(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}

// ---------------------------------------------------------------- sampling
// Bindless material textures. Slot 0 is a 1x1 white texture, slot 1 a flat normal,
// slot 2 a 1x1 black texture, so an "unset" slot always samples something sensible.
layout(set = 1, binding = 0) uniform sampler2D uTextures[];

vec4 r2_sampleBaseColor(uint slot, vec2 uv, vec2 ddx, vec2 ddy) {
    return textureGrad(uTextures[nonuniformEXT(slot)], uv, ddx, ddy);
}
vec4 r2_sampleBaseColorLod(uint slot, vec2 uv, float lod) {
    return textureLod(uTextures[nonuniformEXT(slot)], uv, lod);
}
vec4 r2_sampleOrm(uint slot, vec2 uv, vec2 ddx, vec2 ddy) {
    return textureGrad(uTextures[nonuniformEXT(slot)], uv, ddx, ddy);
}
vec4 r2_sampleNormalTex(uint slot, vec2 uv, vec2 ddx, vec2 ddy) {
    return textureGrad(uTextures[nonuniformEXT(slot)], uv, ddx, ddy);
}
vec4 r2_sampleEmissive(uint slot, vec2 uv, vec2 ddx, vec2 ddy) {
    return textureGrad(uTextures[nonuniformEXT(slot)], uv, ddx, ddy);
}

// Perturbation of a world-space normal by a tangent-space normal map.
vec3 r2_applyNormalMap(vec4 sampled, vec3 N, vec3 T, vec3 B, float scale) {
    vec3 n = sampled.xyz * 2.0 - 1.0;
    n.xy *= scale;
    n = normalize(n);
    return normalize(mat3(T, B, N) * n);
}

// ---------------------------------------------------------------- BRDF
float r2_distributionGGX(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(R2_PI * d * d, 1e-7);
}

float r2_geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float r2_geometrySmith(float NdotV, float NdotL, float roughness) {
    return r2_geometrySchlickGGX(NdotV, roughness) * r2_geometrySchlickGGX(NdotL, roughness);
}

vec3 r2_fresnelSchlick(float cosTheta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 r2_fresnelSchlickRoughness(float cosTheta, vec3 f0, float roughness) {
    vec3 fr = max(vec3(1.0 - roughness), f0);
    return f0 + (fr - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Full metallic-roughness BRDF evaluation for a single light direction.
vec3 r2_brdf(vec3 N, vec3 V, vec3 L, vec3 albedo, float metallic, float roughness, vec3 f0) {
    vec3 H = normalize(V + L);
    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    float NdotV = clamp(dot(N, V), 1e-4, 1.0);
    if (NdotL <= 0.0) return vec3(0.0);
    float NdotH = clamp(dot(N, H), 0.0, 1.0);
    float VdotH = clamp(dot(V, H), 0.0, 1.0);

    float D = r2_distributionGGX(NdotH, roughness);
    float G = r2_geometrySmith(NdotV, NdotL, roughness);
    vec3 F = r2_fresnelSchlick(VdotH, f0);

    vec3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 1e-6);
    vec3 kd = (vec3(1.0) - F) * (1.0 - metallic);
    return (kd * albedo * R2_INV_PI + specular) * NdotL;
}

// Environment BRDF approximation (Karis' analytic fit) for split-sum IBL.
vec2 r2_envBrdfApprox(float NdotV, float roughness) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
    return vec2(-1.04, 1.04) * a004 + r.zw;
}

// ---------------------------------------------------------------- misc
// Interleaved gradient noise, used for dithering and stochastic sampling.
float r2_ign(vec2 pixel) {
    return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

// A stable per-pixel random sequence.
float r2_hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// GGX importance sampling around +Z (used by the procedural reflection probe).
vec3 r2_importanceSampleGGX(vec2 xi, vec3 N, float roughness) {
    float a = roughness * roughness;
    float phi = 2.0 * R2_PI * xi.x;
    float cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    vec3 H = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    vec3 up = abs(N.z) < 0.999 ? vec3(0, 0, 1) : vec3(1, 0, 0);
    vec3 tangentX = normalize(cross(up, N));
    vec3 tangentY = cross(N, tangentX);
    return normalize(tangentX * H.x + tangentY * H.y + N * H.z);
}

// Interpolates the procedural room irradiance from the three analytic bounce colours.
// `N` is the world-space shading normal; the result approximates the room's
// multi-bounce ambient light without any precomputation.
vec3 r2_roomAmbient(vec3 N, float ao) {
    float up = N.y;
    vec3 fromCeiling = g.envCeiling.rgb;
    vec3 fromWall = g.envWall.rgb;
    vec3 fromFloor = g.envFloor.rgb;
    vec3 ambient;
    if (up >= 0.0) {
        ambient = mix(fromWall, fromCeiling, up * up);
    } else {
        ambient = mix(fromWall, fromFloor, up * up);
    }
    // Horizon-weighted blend so vertical surfaces still pick up some ceiling light.
    float horizon = 1.0 - abs(up);
    ambient = mix(ambient, mix(fromWall, fromCeiling * 0.6 + fromFloor * 0.4, 0.5), horizon * 0.35);
    return ambient * g.envCeiling.a * g.ambientTint.rgb * ao;
}

#endif  // ROOM2_COMMON_GLSL
